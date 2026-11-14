"""Concurrency and resource limits validation script for Graph RAG Benchmark."""

import os
import sys
import threading
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import psutil
except ImportError:
    psutil = None

from graphrag_bench.config import get_settings
from graphrag_bench.observability import ConcurrencyLimiter


def test_concurrency_and_resource_limits() -> None:
    settings = get_settings()
    limiter = ConcurrencyLimiter.get_instance(settings)

    print("============================================================")
    print("  PHASE 9 CONCURRENCY & BACKPRESSURE STRESS TEST            ")
    print("============================================================")
    print(f"Configured MAX_CONCURRENT_GENERATIONS: {settings.max_concurrent_generations}")
    print(f"Configured MAX_CONCURRENT_BENCHMARKS:  {settings.max_concurrent_benchmarks}")

    if psutil:
        process = psutil.Process(os.getpid())
        init_mem_mb = process.memory_info().rss / (1024 * 1024)
        init_cpu = psutil.cpu_percent(interval=0.2)
        print(f"Initial Memory RSS: {init_mem_mb:.2f} MB | Initial CPU: {init_cpu:.1f}%")

    # 1. Test 2 Concurrent Generation Slots
    slot1_acquired = threading.Event()
    slot2_acquired = threading.Event()
    release_slots = threading.Event()

    def worker_gen(slot_event: threading.Event, name: str):
        with limiter.acquire_generation_slot(timeout=5.0):
            slot_event.set()
            print(f"  [{name}] Generation slot acquired. Active: {limiter.active_generations}")
            release_slots.wait(timeout=10.0)
            print(f"  [{name}] Generation slot releasing.")

    t1 = threading.Thread(target=worker_gen, args=(slot1_acquired, "Thread-1"))
    t2 = threading.Thread(target=worker_gen, args=(slot2_acquired, "Thread-2"))

    t1.start()
    t2.start()

    slot1_acquired.wait(timeout=5.0)
    slot2_acquired.wait(timeout=5.0)

    print(f"\nBoth slots active. Current active generations: {limiter.active_generations}")
    assert limiter.active_generations == 2, (
        f"Expected 2 active generations, got {limiter.active_generations}"
    )

    # 2. Test 3rd Slot Rejection / Timeout
    print("Attempting to acquire 3rd generation slot (should fail with TimeoutError)...")
    t0 = time.perf_counter()
    third_rejected = False
    try:
        with limiter.acquire_generation_slot(timeout=0.5):
            pass
    except TimeoutError as exc:
        third_rejected = True
        dur = (time.perf_counter() - t0) * 1000.0
        print(f"  => SUCCESS: 3rd slot correctly rejected after {dur:.2f} ms with error: {exc}")

    assert third_rejected, "3rd concurrent generation was NOT rejected!"

    # Release workers
    release_slots.set()
    t1.join()
    t2.join()

    print(f"After release, active generations: {limiter.active_generations}")
    assert limiter.active_generations == 0

    # 3. Test Benchmark Mutual Exclusion
    print("\nTesting Benchmark Suite Mutual Exclusion (max 1 run)...")
    bench_slot1 = threading.Event()
    bench_release = threading.Event()

    def worker_bench():
        with limiter.acquire_benchmark_slot(timeout=5.0):
            bench_slot1.set()
            bench_release.wait(timeout=10.0)

    tb = threading.Thread(target=worker_bench)
    tb.start()
    bench_slot1.wait(timeout=5.0)

    assert limiter.active_benchmarks == 1
    bench_second_rejected = False
    try:
        with limiter.acquire_benchmark_slot(timeout=0.2):
            pass
    except RuntimeError as exc:
        bench_second_rejected = True
        print(f"  => SUCCESS: 2nd benchmark correctly rejected with error: {exc}")

    assert bench_second_rejected, "2nd benchmark run was NOT rejected!"

    bench_release.set()
    tb.join()
    assert limiter.active_benchmarks == 0

    if psutil:
        final_mem_mb = process.memory_info().rss / (1024 * 1024)
        final_cpu = psutil.cpu_percent(interval=0.2)
        print(f"\nFinal Memory RSS: {final_mem_mb:.2f} MB | Final CPU: {final_cpu:.1f}%")
    print("PASS: Concurrency limiters and backpressure mechanisms verified!")


if __name__ == "__main__":
    test_concurrency_and_resource_limits()
