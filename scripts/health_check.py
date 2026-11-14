"""Production health check CLI entrypoint for container HEALTHCHECK and monitoring probes."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.production_health import (
    ProductionHealthMonitor,
    ProductionHealthState,
)


def main() -> None:
    monitor = ProductionHealthMonitor()
    report = monitor.run_all_checks()

    # Print summary
    print(f"Health Status: {report.overall_state.value}")
    print(f"Liveness:      {'OK' if report.liveness else 'FAIL'}")
    print(f"Readiness:     {'OK' if report.readiness else 'FAIL'}")
    print(f"Summary:       {report.summary}")

    # Return exit code:
    # 0 = HEALTHY or DEGRADED (operational)
    # 1 = UNAVAILABLE (critical failure)
    if report.overall_state == ProductionHealthState.UNAVAILABLE or not report.readiness:
        print("\nCRITICAL FAILURE: System is UNAVAILABLE.", file=sys.stderr)
        for check in report.checks:
            if check.state == ProductionHealthState.UNAVAILABLE:
                print(f"  - {check.name}: {check.message}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
