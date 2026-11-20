#!/usr/bin/env python3
"""
Hybrid MPI-CUDA stencil and shared-memory false-sharing laboratory.

Run on one NVIDIA T4 node with four MPI ranks:
    mpirun -np 4 python hybrid_mpi_cuda_cache.py YOUR_GITHUB_USERNAME

Important:
- This is a reference implementation of the mechanisms specified in the lab PDF.
- The PDF's starter source is truncated after the beginning of init_hybrid_context
  and does not specify the exact PDE/source/boundary formula. This implementation
  therefore states its numerical assumptions in the accompanying README.
- Hardware-dependent outputs must be collected on the target T4/MPI environment.
"""
import os
import sys
import json
import math
import time
import hashlib
import platform
import subprocess
import re
import ctypes
from pathlib import Path

import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2
GLOBAL_NX = 512
GLOBAL_NY = 512
LOCAL_NX = GLOBAL_NX // 2
LOCAL_NY = GLOBAL_NY // 2
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15
DX = 1.0 / (GLOBAL_NX + 1)
KAPPA_BASE = 0.10
KAPPA_VARIATION = 0.05


@cuda.jit
def pack_halos_kernel(u, send_top, send_bottom, send_left, send_right):
    """Pack four 256-element faces into contiguous 1D device buffers."""
    q = cuda.grid(1)
    if q < LOCAL_NX:
        send_top[q] = u[1, q + 1]
        send_bottom[q] = u[LOCAL_NX, q + 1]
        send_left[q] = u[q + 1, 1]
        send_right[q] = u[q + 1, LOCAL_NY]


@cuda.jit
def unpack_halos_kernel(u, recv_top, recv_bottom, recv_left, recv_right,
                        has_up, has_down, has_left, has_right):
    """Unpack received faces; physical outer boundaries use zero normal flux."""
    q = cuda.grid(1)
    if q < LOCAL_NX:
        if has_up:
            u[0, q + 1] = recv_top[q]
        else:
            u[0, q + 1] = u[1, q + 1]
        if has_down:
            u[LOCAL_NX + 1, q + 1] = recv_bottom[q]
        else:
            u[LOCAL_NX + 1, q + 1] = u[LOCAL_NX, q + 1]
        if has_left:
            u[q + 1, 0] = recv_left[q]
        else:
            u[q + 1, 0] = u[q + 1, 1]
        if has_right:
            u[q + 1, LOCAL_NY + 1] = recv_right[q]
        else:
            u[q + 1, LOCAL_NY + 1] = u[q + 1, LOCAL_NY]


@cuda.jit
def tiled_stencil_kernel(u, kappa, source, out, dt, dx, region):
    """
    Conservative variable-coefficient 5-point diffusion update.
    region=0: all cells; region=1: deep interior [2..255,2..255];
    region=2: outer one-cell border of the local 256x256 interior.
    """
    tile = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = cuda.blockIdx.x * TILE_DIM + tx + 1
    i = cuda.blockIdx.y * TILE_DIM + ty + 1

    if i <= LOCAL_NX and j <= LOCAL_NY:
        tile[ty + 1, tx + 1] = u[i, j]
        if ty == 0:
            tile[0, tx + 1] = u[i - 1, j]
        if ty == TILE_DIM - 1:
            tile[TILE_DIM + 1, tx + 1] = u[i + 1, j]
        if tx == 0:
            tile[ty + 1, 0] = u[i, j - 1]
        if tx == TILE_DIM - 1:
            tile[ty + 1, TILE_DIM + 1] = u[i, j + 1]
    cuda.syncthreads()

    if i <= LOCAL_NX and j <= LOCAL_NY:
        is_border = (i == 1 or i == LOCAL_NX or j == 1 or j == LOCAL_NY)
        if region == 1 and is_border:
            return
        if region == 2 and not is_border:
            return

        c = tile[ty + 1, tx + 1]
        n = tile[ty, tx + 1]
        s = tile[ty + 2, tx + 1]
        w = tile[ty + 1, tx]
        e = tile[ty + 1, tx + 2]

        kc = kappa[i, j]
        kn = 2.0 * kc * kappa[i - 1, j] / (kc + kappa[i - 1, j])
        ks = 2.0 * kc * kappa[i + 1, j] / (kc + kappa[i + 1, j])
        kw = 2.0 * kc * kappa[i, j - 1] / (kc + kappa[i, j - 1])
        ke = 2.0 * kc * kappa[i, j + 1] / (kc + kappa[i, j + 1])
        inv_dx2 = 1.0 / (dx * dx)
        diffusion = (kn * (n - c) + ks * (s - c) +
                     kw * (w - c) + ke * (e - c)) * inv_dx2
        out[i, j] = c + dt * (diffusion + source[i, j] -
                              LAMBDA_DAMP * c)


@njit(nogil=True)
def host_shm_accumulate(shm_view, index, iterations):
    """Repeated in-place read-modify-write on one shared uint64 slot."""
    for _ in range(iterations):
        shm_view[index] = shm_view[index] + np.uint64(1)


def get_gpu_uuid():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8")
    return out.strip().splitlines()[0].strip()


def detect_cpu_line_size():
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


def init_hybrid_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}")

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{os.cpu_count()}"
    line_bytes = detect_cpu_line_size()

    digest = hashlib.sha256(
        f"{github_user.strip()}::{cpu_sig}::{gpu_uuid}".encode()).digest()
    instance_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)

    cart = comm.Create_cart(dims=(2, 2), periods=(False, False), reorder=False)
    cart_rank = cart.Get_rank()
    coords = cart.Get_coords(cart_rank)
    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
    node_rank = node_comm.Get_rank()
    node_size = node_comm.Get_size()
    if node_size != 4:
        raise RuntimeError(
            f"All four ranks must share one node for this benchmark; node size={node_size}")

    # Allocate storage on node rank 0 only, with 63 extra bytes for alignment.
    nslots = 64
    nbytes = nslots * np.dtype(np.uint64).itemsize
    alloc_bytes = nbytes + 63 if node_rank == 0 else 0
    win = MPI.Win.Allocate_shared(alloc_bytes, disp_unit=1, comm=node_comm)
    buf, _ = win.Shared_query(0)
    raw_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    offset = (-raw_addr) % 64
    shm_view = np.ndarray((nslots,), dtype=np.uint64, buffer=buf, offset=offset)
    if node_rank == 0:
        shm_view[:] = 0
    node_comm.Barrier()

    return {
        "comm": comm, "rank": rank, "size": size, "cart": cart,
        "coords": coords, "up": up, "down": down, "left": left, "right": right,
        "node_comm": node_comm, "node_rank": node_rank, "node_size": node_size,
        "win": win, "shm_view": shm_view, "gpu_uuid": gpu_uuid,
        "cpu_signature": cpu_sig, "cpu_line_bytes": line_bytes,
        "instance_seed": instance_seed,
    }


def initialize_fields(ctx):
    """Build rank-local padded arrays using global coordinates only."""
    r0, r1 = ctx["coords"]
    gx = r0 * LOCAL_NX + np.arange(1, LOCAL_NX + 1)
    gy = r1 * LOCAL_NY + np.arange(1, LOCAL_NY + 1)
    x = gx[:, None] * DX
    y = gy[None, :] * DX
    u = np.zeros((LOCAL_NX + 2, LOCAL_NY + 2), dtype=np.float64)
    kappa = np.full_like(u, KAPPA_BASE)
    source = np.zeros_like(u)
    u[1:-1, 1:-1] = np.sin(np.pi * x) * np.sin(np.pi * y)
    kappa[1:-1, 1:-1] = KAPPA_BASE + KAPPA_VARIATION * (
        0.5 + 0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y))
    # Initialize local physical ghost values; MPI halo exchange fills internal faces.
    u[0, 1:-1] = u[1, 1:-1]
    u[-1, 1:-1] = u[-2, 1:-1]
    u[1:-1, 0] = u[1:-1, 1]
    u[1:-1, -1] = u[1:-1, -2]
    kappa[:, :] = np.maximum(kappa, KAPPA_BASE)
    return u, kappa, source


def make_device_state(u_host, kappa_host, source_host):
    u = cuda.to_device(u_host)
    kappa = cuda.to_device(kappa_host)
    source = cuda.to_device(source_host)
    out = cuda.device_array_like(u)
    return u, kappa, source, out


def make_buffers():
    d = [cuda.device_array(LOCAL_NX, dtype=np.float64) for _ in range(4)]
    h_send = [cuda.pinned_array(LOCAL_NX, dtype=np.float64) for _ in range(4)]
    h_recv = [cuda.pinned_array(LOCAL_NX, dtype=np.float64) for _ in range(4)]
    return d, h_send, h_recv


def exchange_halos(u, buffers, ctx, stream):
    d_send, h_send, h_recv = buffers
    pack_halos_kernel[16, 256, stream](
        u, d_send[0], d_send[1], d_send[2], d_send[3])
    for q in range(4):
        d_send[q].copy_to_host(h_send[q], stream=stream)
    stream.synchronize()

    cart = ctx["cart"]
    reqs = []
    # Receive top from up's bottom, bottom from down's top,
    # left from left's right, and right from right's left.
    if ctx["up"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(h_recv[0], source=ctx["up"], tag=1))
    if ctx["down"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(h_recv[1], source=ctx["down"], tag=0))
    if ctx["left"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(h_recv[2], source=ctx["left"], tag=3))
    if ctx["right"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(h_recv[3], source=ctx["right"], tag=2))

    if ctx["up"] != MPI.PROC_NULL:
        reqs.append(cart.Isend(h_send[0], dest=ctx["up"], tag=0))
    if ctx["down"] != MPI.PROC_NULL:
        reqs.append(cart.Isend(h_send[1], dest=ctx["down"], tag=1))
    if ctx["left"] != MPI.PROC_NULL:
        reqs.append(cart.Isend(h_send[2], dest=ctx["left"], tag=2))
    if ctx["right"] != MPI.PROC_NULL:
        reqs.append(cart.Isend(h_send[3], dest=ctx["right"], tag=3))
    MPI.Request.Waitall(reqs)

    for q in range(4):
        d_send[q].copy_to_device(h_recv[q], stream=stream)
    unpack_halos_kernel[1, 256, stream](
        u, d_send[0], d_send[1], d_send[2], d_send[3],
        ctx["up"] != MPI.PROC_NULL, ctx["down"] != MPI.PROC_NULL,
        ctx["left"] != MPI.PROC_NULL, ctx["right"] != MPI.PROC_NULL)
    stream.synchronize()


def launch_stencil(u, kappa, source, out, dt, region, stream):
    tiled_stencil_kernel[(16, 16), (16, 16), stream](
        u, kappa, source, out, dt, DX, region)


def run_serialized(ctx, u0, kappa0, source0, dt, steps, record_energy=False):
    u, kappa, source, out = make_device_state(u0, kappa0, source0)
    buffers = make_buffers()
    stream = cuda.stream()
    energies = {}
    for step in range(steps):
        exchange_halos(u, buffers, ctx, stream)
        launch_stencil(u, kappa, source, out, dt, 0, stream)
        stream.synchronize()
        u, out = out, u
        if record_energy and (step + 1) in (100, 200, 300):
            host = u.copy_to_host()
            local = float(np.sum(host[1:-1, 1:-1] ** 2))
            energies[step + 1] = ctx["cart"].allreduce(local, op=MPI.SUM)
    if record_energy:
        host = u.copy_to_host()
        local = float(np.sum(host[1:-1, 1:-1] ** 2))
        energies[0] = ctx["cart"].allreduce(
            float(np.sum(u0[1:-1, 1:-1] ** 2)), op=MPI.SUM)
    return u.copy_to_host(), energies


def run_overlapped(ctx, u0, kappa0, source0, dt, steps, record_energy=False):
    u, kappa, source, out = make_device_state(u0, kappa0, source0)
    buffers = make_buffers()
    stream_interior = cuda.stream()
    stream_boundary = cuda.stream()
    energies = {}
    for step in range(steps):
        # Pack and stage outgoing faces, then post nonblocking MPI.
        d_send, h_send, h_recv = buffers
        pack_halos_kernel[1, 256, stream_boundary](
            u, d_send[0], d_send[1], d_send[2], d_send[3])
        for q in range(4):
            d_send[q].copy_to_host(h_send[q], stream=stream_boundary)
        stream_boundary.synchronize()

        cart = ctx["cart"]
        reqs = []
        if ctx["up"] != MPI.PROC_NULL:
            reqs.append(cart.Irecv(h_recv[0], source=ctx["up"], tag=1))
        if ctx["down"] != MPI.PROC_NULL:
            reqs.append(cart.Irecv(h_recv[1], source=ctx["down"], tag=0))
        if ctx["left"] != MPI.PROC_NULL:
            reqs.append(cart.Irecv(h_recv[2], source=ctx["left"], tag=3))
        if ctx["right"] != MPI.PROC_NULL:
            reqs.append(cart.Irecv(h_recv[3], source=ctx["right"], tag=2))
        if ctx["up"] != MPI.PROC_NULL:
            reqs.append(cart.Isend(h_send[0], dest=ctx["up"], tag=0))
        if ctx["down"] != MPI.PROC_NULL:
            reqs.append(cart.Isend(h_send[1], dest=ctx["down"], tag=1))
        if ctx["left"] != MPI.PROC_NULL:
            reqs.append(cart.Isend(h_send[2], dest=ctx["left"], tag=2))
        if ctx["right"] != MPI.PROC_NULL:
            reqs.append(cart.Isend(h_send[3], dest=ctx["right"], tag=3))

        # Deep interior does not depend on newly received halo values.
        launch_stencil(u, kappa, source, out, dt, 1, stream_interior)
        MPI.Request.Waitall(reqs)

        for q in range(4):
            d_send[q].copy_to_device(h_recv[q], stream=stream_boundary)
        unpack_halos_kernel[1, 256, stream_boundary](
            u, d_send[0], d_send[1], d_send[2], d_send[3],
            ctx["up"] != MPI.PROC_NULL, ctx["down"] != MPI.PROC_NULL,
            ctx["left"] != MPI.PROC_NULL, ctx["right"] != MPI.PROC_NULL)
        launch_stencil(u, kappa, source, out, dt, 2, stream_boundary)
        stream_interior.synchronize()
        stream_boundary.synchronize()
        u, out = out, u

        if record_energy and (step + 1) in (100, 200, 300):
            host = u.copy_to_host()
            local = float(np.sum(host[1:-1, 1:-1] ** 2))
            energies[step + 1] = cart.allreduce(local, op=MPI.SUM)
    if record_energy:
        energies[0] = ctx["cart"].allreduce(
            float(np.sum(u0[1:-1, 1:-1] ** 2)), op=MPI.SUM)
    return u.copy_to_host(), energies


def audit_ptx():
    dummy = cuda.to_device(np.zeros((LOCAL_NX + 2, LOCAL_NY + 2), dtype=np.float64))
    k = cuda.to_device(np.full_like(dummy.copy_to_host(), 0.1))
    s = cuda.to_device(np.zeros_like(dummy.copy_to_host()))
    o = cuda.device_array_like(dummy)
    launch_stencil(dummy, k, s, o, 1e-8, 0, cuda.stream())
    cuda.synchronize()
    asm = tiled_stencil_kernel.inspect_asm()
    if isinstance(asm, dict):
        asm_text = "\n".join(asm.values())
    else:
        asm_text = str(asm)
    counts = {
        "st.shared.f64": len(re.findall(r"\bst\.shared\.f64\b", asm_text)),
        "ld.shared.f64": len(re.findall(r"\bld\.shared\.f64\b", asm_text)),
        "bar.sync": len(re.findall(r"\bbar\.sync\b", asm_text)),
    }
    return counts, asm_text


def audit_llvm():
    host_shm_accumulate(np.zeros(1, dtype=np.uint64), 0, 1)
    sig = host_shm_accumulate.signatures[0]
    llvm = host_shm_accumulate.inspect_llvm(sig)
    lines = [line.strip() for line in llvm.splitlines()
             if re.search(r"\b(load|store)\b", line)]
    return {"llvm_load_store_instruction_lines": len(lines),
            "llvm_load_lines": sum(" load " in f" {x} " for x in lines),
            "llvm_store_lines": sum(" store " in f" {x} " for x in lines)}, llvm


def benchmark_shm_strides(ctx):
    node_comm = ctx["node_comm"]
    view = ctx["shm_view"]
    results = {}
    for stride in (1, 8, 16):
        index = ctx["node_rank"] * stride
        view[index] = 0
        node_comm.Barrier()
        t0 = MPI.Wtime()
        host_shm_accumulate(view, index, SHM_ITERS)
        node_comm.Barrier()
        elapsed = MPI.Wtime() - t0
        max_elapsed = node_comm.allreduce(elapsed, op=MPI.MAX)
        expected = np.uint64(SHM_ITERS)
        ok = bool(view[index] == expected)
        all_ok = node_comm.allreduce(int(ok), op=MPI.MIN)
        results[str(stride)] = {
            "max_elapsed_seconds": float(max_elapsed),
            "updates_per_rank": SHM_ITERS,
            "verified": bool(all_ok),
            "final_values": [int(view[r * stride]) for r in range(4)]
        }
    return results


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: mpirun -np 4 python hybrid_mpi_cuda_cache.py GITHUB_USERNAME")
    github_user = sys.argv[1]
    ctx = init_hybrid_context(github_user)
    rank = ctx["rank"]
    u0, kappa, source = initialize_fields(ctx)

    kappa_max = float(ctx["cart"].allreduce(
        float(np.max(kappa[1:-1, 1:-1])), op=MPI.MAX))
    dt_max = DX * DX / (4.0 * kappa_max)
    dt = 0.20 * dt_max

    ctx["cart"].Barrier()
    t0 = MPI.Wtime()
    serial_field, serial_energy = run_serialized(
        ctx, u0, kappa, source, dt, STENCIL_STEPS, record_energy=True)
    ctx["cart"].Barrier()
    serial_ms = (MPI.Wtime() - t0) * 1000.0

    ctx["cart"].Barrier()
    t0 = MPI.Wtime()
    overlap_field, overlap_energy = run_overlapped(
        ctx, u0, kappa, source, dt, STENCIL_STEPS, record_energy=True)
    ctx["cart"].Barrier()
    overlap_ms = (MPI.Wtime() - t0) * 1000.0

    local_diff = float(np.max(np.abs(serial_field - overlap_field)))
    max_diff = ctx["cart"].allreduce(local_diff, op=MPI.MAX)
    energy_check = {
        str(step): float(overlap_energy[step]) for step in (0, 100, 200, 300)
    }

    ptx_counts, ptx_text = audit_ptx()
    llvm_counts, llvm_text = audit_llvm()
    shm_results = benchmark_shm_strides(ctx)

    if rank == 0:
        manifest = {
            "github_username": github_user,
            "instance_seed": int(ctx["instance_seed"]),
            "cpu_signature": ctx["cpu_signature"],
            "cpu_line_bytes": int(ctx["cpu_line_bytes"]),
            "gpu_uuid": ctx["gpu_uuid"],
            "gpu_name": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                encoding="utf-8").strip().splitlines()[0],
            "mpi_ranks": 4,
            "global_shape": [GLOBAL_NX, GLOBAL_NY],
            "local_shape_with_ghosts": [LOCAL_NX + 2, LOCAL_NY + 2],
            "steps": STENCIL_STEPS,
            "dx": DX,
            "kappa_max": kappa_max,
            "dt": dt,
            "dt_max": dt_max,
            "cfl_ratio_dt_over_dtmax": dt / dt_max,
            "gpu_serialized_ms": serial_ms,
            "gpu_overlapped_ms": overlap_ms,
            "max_diff_serialized_vs_overlapped": max_diff,
            "global_energy_sum_of_squares": energy_check,
            "ptx_instruction_counts": ptx_counts,
            "llvm_audit_counts": llvm_counts,
            "shared_memory_stride_benchmarks": shm_results,
            "assumptions": {
                "pde": "u_t = div(kappa grad(u)) + source - lambda*u",
                "source": "zero source field",
                "physical_boundary": "zero-normal-gradient (ghost equals adjacent interior)",
                "energy_metric": "global sum of u squared; expected to dissipate",
                "stencil": "conservative 5-point variable-coefficient diffusion"
            }
        }
        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open("ptx_audit.txt", "w", encoding="utf-8") as f:
            f.write(ptx_text)
        with open("llvm_audit.txt", "w", encoding="utf-8") as f:
            f.write(llvm_text)
        print(json.dumps(manifest, indent=2))
        if max_diff >= 1e-12:
            raise RuntimeError(f"Serialized/overlapped mismatch: {max_diff}")
        if not all(v["verified"] for v in shm_results.values()):
            raise RuntimeError("Shared-memory accumulation verification failed")

    ctx["cart"].Barrier()
    ctx["win"].Free()
    ctx["node_comm"].Free()
    ctx["cart"].Free()


if __name__ == "__main__":
    main()
