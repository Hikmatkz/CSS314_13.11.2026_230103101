import multiprocessing as mp
import platform
import sys
import time

from harness import emit_verify_tag


STUDENT_ID = 230103101
CLOCK_READS = 2_000_000


def measure_clock_latency():

    start = time.perf_counter_ns()

    for _ in range(CLOCK_READS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_ns = end - start
    average_ns = total_ns / CLOCK_READS

    return total_ns, average_ns


if __name__ == "__main__":

    print("=" * 60)
    print("PHASE 0: SYSTEM INTERROGATION")
    print("=" * 60)

    print("Student ID:", STUDENT_ID)
    print("CPU architecture:", platform.machine())
    print("Operating system:", platform.platform())
    print("Python version:", sys.version)
    print("Multiprocessing method:", mp.get_start_method())
    print("Logical CPU count:", mp.cpu_count())

    print("\nCLOCK CALIBRATION")

    total_ns, average_ns = measure_clock_latency()

    print("Clock reads:", CLOCK_READS)
    print("Total measurement time (ns):", total_ns)
    print(f"Average clock read latency (ns): {average_ns:.4f}")

    print("\nVERIFICATION")

    tag = emit_verify_tag(
        task_id="PHASE0",
        student_seed=STUDENT_ID,
        primary_metric=average_ns,
        secondary_metric=float(CLOCK_READS)
    )

    print(tag)

    print("=" * 60)
    print("PHASE 0 COMPLETED")
    print("=" * 60)

