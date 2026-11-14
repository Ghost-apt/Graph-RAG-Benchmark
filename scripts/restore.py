"""Command-line utility to inspect, verify, and restore backups for Graph RAG Benchmark."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.backup_manager import BackupManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or restore Graph RAG Benchmark backups.")
    parser.add_argument("--list", action="store_true", help="List all available backups")
    parser.add_argument(
        "--verify", type=str, help="Verify SHA-256 integrity of specified backup ID"
    )
    parser.add_argument(
        "--preview", type=str, help="Dry-run preview of files in specified backup ID"
    )
    parser.add_argument("--restore", type=str, help="Restore the specified backup ID")
    parser.add_argument(
        "--target-dir", type=str, default=None, help="Alternate directory for restoration"
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Explicit confirmation required to restore"
    )

    args = parser.parse_args()
    mgr = BackupManager()

    if args.list:
        backups = mgr.list_backups()
        if not backups:
            print("No backups found in repository.")
            sys.exit(0)
        print(f"Found {len(backups)} backups:")
        for b in backups:
            print(
                f"- {b.backup_id} [{b.category.value}] | {b.timestamp} | {b.total_files} files ({b.total_bytes:,} bytes)"
            )
        sys.exit(0)

    if args.verify:
        valid, errors = mgr.verify_backup_integrity(args.verify)
        if valid:
            print(f"Integrity verified for {args.verify}: Archive and manifest hashes are valid.")
            sys.exit(0)
        else:
            print(f"Integrity check FAILED for {args.verify}:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    if args.preview:
        prev = mgr.preview_restore(args.preview)
        print(f"\nRestore Preview for {prev.backup_id}:")
        print(f"Valid Integrity: {prev.valid_integrity}")
        print(f"Category:        {prev.category.value}")
        print(f"Total Files:     {prev.total_files}")
        print(f"Total Bytes:     {prev.total_bytes:,} bytes")
        if prev.validation_errors:
            print("Validation Errors:")
            for e in prev.validation_errors:
                print(f"  - {e}")
        else:
            print("\nFiles to be restored:")
            for f in prev.files_to_restore[:20]:
                print(f"  + {f}")
            if len(prev.files_to_restore) > 20:
                print(f"  ... and {len(prev.files_to_restore) - 20} more files.")
        sys.exit(0)

    if args.restore:
        if not args.confirm:
            print("Error: You must provide --confirm to proceed with restore.", file=sys.stderr)
            sys.exit(1)

        target = Path(args.target_dir) if args.target_dir else None
        res = mgr.restore_backup(args.restore, target_dir=target, confirm=True)
        if res.success:
            print("\nRestore successful!")
            print(f"Restored {res.restored_files_count} files into {res.target_dir}")
            if res.pre_restore_snapshot_id:
                print(f"Pre-restore safety snapshot created: {res.pre_restore_snapshot_id}")
            print(f"Duration: {res.duration_ms:.2f} ms")
            sys.exit(0)
        else:
            print(f"Restore failed: {res.error}", file=sys.stderr)
            sys.exit(1)

    parser.print_help()


if __name__ == "__main__":
    main()
