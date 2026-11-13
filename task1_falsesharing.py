
import argparse
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag

M_TOTAL = 24_000_000  # 2.4e7 total read-modify-write steps (spec)
MASK = 0xFFFFFFFFFFFFFFFF
MULT = 6364136223846793005


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    # Python 3.13+: track=False avoids resource-tracker noise for attached blocks.
    try:
        shm = shared_memory.SharedMemory(name=shm_name, track=False)
    except TypeError:
        shm = shared_memory.SharedMemory(name=shm_name)
    nslots = len(shm.buf) // 8
    arr = (ctypes.c_uint64 * nslots).from_buffer(shm.buf)
    acc = arr[slot_idx]
    for i in range(iters):
        acc = (acc * MULT + (i | 1)) & MASK
        arr[slot_idx] = acc  # store to the (possibly shared) cache line
    del arr  # release buffer export before closing
    shm.close()


def make_runner(p: int, stride: int, m_total: int):
    """Return (fn, cleanup). The shared block is allocated once per config,
    zeroed at the start of every run; process spawn cost is inside the timed
    region (it is constant per p and small relative to multi-second runs)."""
    needed = ((p - 1) * stride + 1) * 8
    size = max(4096, ((needed + 4095) // 4096) * 4096)
    shm = shared_memory.SharedMemory(create=True, size=size)
    iters = m_total // p

    def fn() -> None:
        shm.buf[:] = bytes(len(shm.buf))
        procs = [mp.Process(target=fs_worker, args=(shm.name, t * stride, iters))
                 for t in range(p)]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()

    def cleanup() -> None:
        shm.close()
        shm.unlink()

    return fn, cleanup


def bench(p: int, stride: int, m_total: int):
    fn, cleanup = make_runner(p, stride, m_total)
    try:
        return run_benchmark(fn, warmup_runs=2, timed_runs=7)
    finally:
        cleanup()


def karp_flatt(sp: float, p: int) -> float:
    return ((1.0 / sp) - (1.0 / p)) / (1.0 - 1.0 / p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True, help="numeric Student ID (hash seed)")
    ap.add_argument("--pphys", type=int, required=True, help="physical core count")
    ap.add_argument("--quick", action="store_true", help="use M=2.4e6 for a dry run")
    args = ap.parse_args()

    m_total = M_TOTAL // 10 if args.quick else M_TOTAL
    pphys = args.pphys
    plog = mp.cpu_count()
    out = []

    def emit(line: str = "") -> None:
        print(line, flush=True)
        out.append(line)

    # ---------------- Table 1A: stride sweep at p = Pphys ----------------
    strides = [1, 2, 4, 8, 16, 32]
    emit(f"## Table 1A: Stride sensitivity (p = {pphys}, M = {m_total:,})")
    emit("| stride | byte offset | median (ms) | IQR (ms) | throughput (Mops/s) |")
    emit("|---|---|---|---|---|")
    t1a = {}
    for s in strides:
        st = bench(pphys, s, m_total)
        t1a[s] = st.median_ms
        mops = (m_total / (st.median_ms / 1e3)) / 1e6
        emit(f"| {s} | {s * 8} | {st.median_ms:.2f} | {st.iqr_ms:.2f} | {mops:.2f} |")
    emit()

    # ---------------- Table 1B: process-scaling sweep ----------------
    ps = sorted({q for q in (1, 2, 4, 8, 16, 32, 64) if q <= plog} | {pphys, plog})
    series = {}
    for s in (1, 16):
        series[s] = {p: bench(p, s, m_total).median_ms for p in ps}

    for s in (1, 16):
        emit(f"## Table 1B: Karp-Flatt series, stride = {s} ({s * 8} B)")
        emit("| p | T_p (ms) | S_p | E_p | e(p) |")
        emit("|---|---|---|---|---|")
        t1 = series[s][1]
        for p in ps:
            tp = series[s][p]
            sp = t1 / tp
            ep = sp / p
            e = f"{karp_flatt(sp, p):.4f}" if p >= 2 else "n/a"
            emit(f"| {p} | {tp:.2f} | {sp:.3f} | {ep:.3f} | {e} |")
        emit()

    # ---------------- Verification hash ----------------
    tag = emit_verify_tag("PHASE1", args.id, t1a[1], t1a[16])
    emit(tag)

    with open("phase1_results.md", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n(saved to phase1_results.md)")


if __name__ == "__main__":
    main()