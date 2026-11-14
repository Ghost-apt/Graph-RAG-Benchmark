"""Command-line utility to trigger on-demand backups for Graph RAG Benchmark."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.backup_manager import BackupCategory, BackupManager


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a secure backup of Graph RAG Benchmark data."
    )
    parser.add_argument(
        "--category",
        choices=["canonical", "benchmark", "full"],
        default="full",
        help="Data scope to include in backup (default: full)",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Optional descriptive label for this backup",
    )
    args = parser.parse_args()

    mgr = BackupManager()
    category = BackupCategory(args.category)
    print(f"Starting {category.value.upper()} backup...")
    result = mgr.create_backup(category=category, description=args.description)

    if result.success and result.manifest:
        print("\nBackup Created Successfully!")
        print(f"Backup ID:    {result.backup_id}")
        print(f"Path:         {result.backup_path}")
        print(f"Total Files:  {result.manifest.total_files}")
        print(f"Total Bytes:  {result.manifest.total_bytes:,} bytes")
        print(f"Duration:     {result.duration_ms:.2f} ms")
        print(f"Archive Hash: {result.manifest.archive_checksum}")
        sys.exit(0)
    else:
        print(f"\nBackup Failed: {result.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
