"""Cryptographic backup and recovery management subsystem for Graph RAG Benchmark.

Provides:
1. Category-isolated and full system backups (canonical data, benchmark data, config).
2. Cryptographic SHA-256 file manifest generation and integrity verification.
3. Safe archive extraction with path traversal protection (Zip/Tar Slip defense).
4. Restore preview (dry-run) and isolated target directory restoration.
5. Pre-restore state snapshots to prevent unrecoverable data loss.
"""

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.logger import get_logger

logger = get_logger("backup_manager")


class BackupCategory(str, Enum):
    """Categorized data backup targets."""

    CANONICAL = "canonical"
    BENCHMARK = "benchmark"
    CONFIG = "config"
    FULL = "full"


class FileChecksum(BaseModel):
    """Integrity record for a single archived file."""

    relative_path: str = Field(description="Relative path within the backup archive")
    size_bytes: int = Field(description="File size in bytes")
    sha256: str = Field(description="Cryptographic SHA-256 hex digest")


class BackupManifest(BaseModel):
    """Cryptographic manifest stored in every backup directory."""

    backup_id: str = Field(description="Unique backup identifier")
    timestamp: str = Field(description="ISO-8601 creation timestamp")
    app_version: str = Field(description="Application version at backup time")
    category: BackupCategory = Field(description="Backup category scope")
    description: str = Field(default="", description="Operator notes or description")
    files: list[FileChecksum] = Field(default_factory=list, description="List of file checksums")
    total_files: int = Field(default=0, description="Total count of archived files")
    total_bytes: int = Field(default=0, description="Total uncompressed bytes")
    archive_checksum: str = Field(default="", description="SHA-256 hash of tar archive")


class BackupResult(BaseModel):
    """Outcome report for a backup operation."""

    success: bool
    backup_id: str
    backup_path: str
    category: BackupCategory
    manifest: BackupManifest | None = None
    duration_ms: float = 0.0
    error: str | None = None


class RestorePreview(BaseModel):
    """Dry-run preview of files that would be modified or created by restore."""

    backup_id: str
    valid_integrity: bool
    category: BackupCategory
    files_to_restore: list[str]
    total_files: int
    total_bytes: int
    validation_errors: list[str] = Field(default_factory=list)


class RestoreResult(BaseModel):
    """Outcome report for a restore operation."""

    success: bool
    backup_id: str
    target_dir: str
    restored_files_count: int
    pre_restore_snapshot_id: str | None = None
    duration_ms: float = 0.0
    error: str | None = None


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file reading in chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class BackupManager:
    """Orchestrates secure backup creation, manifest verification, and recovery."""

    def __init__(
        self,
        settings: Settings | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self.backup_root = (self.base_dir / self.settings.backup_dir).resolve()
        self.backup_root.mkdir(parents=True, exist_ok=True)

    def _resolve_paths_for_category(self, category: BackupCategory) -> list[tuple[Path, str]]:
        """Resolve list of (source_path, archive_prefix) for the specified category."""
        paths: list[tuple[Path, str]] = []

        if category in {BackupCategory.CANONICAL, BackupCategory.FULL}:
            paths.append((self.base_dir / self.settings.processed_dir, "data/processed"))
            paths.append((self.base_dir / self.settings.graph_dir, "data/knowledge_graph"))
            paths.append((self.base_dir / self.settings.chroma_path, "data/chroma"))
            paths.append((self.base_dir / self.settings.upload_dir, "data/uploads"))

        if category in {BackupCategory.BENCHMARK, BackupCategory.FULL}:
            paths.append((self.base_dir / self.settings.benchmark_dir, "data/phase8_benchmark"))

        return paths

    def create_backup(
        self,
        category: BackupCategory = BackupCategory.FULL,
        description: str = "",
    ) -> BackupResult:
        """Create a timestamped backup with SHA-256 manifest and tar archive."""
        t0 = time.perf_counter()
        timestamp_str = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        backup_id = f"backup_{category.value}_{timestamp_str}"
        target_backup_dir = self.backup_root / backup_id

        if target_backup_dir.exists():
            return BackupResult(
                success=False,
                backup_id=backup_id,
                backup_path=str(target_backup_dir),
                category=category,
                error=f"Backup destination {target_backup_dir} already exists.",
            )

        target_backup_dir.mkdir(parents=True, exist_ok=True)
        archive_path = target_backup_dir / "data.tar.gz"

        try:
            category_sources = self._resolve_paths_for_category(category)
            file_checksums: list[FileChecksum] = []
            total_bytes = 0

            # Write archive safely
            with tarfile.open(archive_path, "w:gz") as tar:
                for src_path, arc_prefix in category_sources:
                    if not src_path.exists():
                        continue

                    if src_path.is_file():
                        h = compute_sha256(src_path)
                        size = src_path.stat().st_size
                        tar.add(src_path, arcname=arc_prefix)
                        file_checksums.append(
                            FileChecksum(
                                relative_path=arc_prefix.replace("\\", "/"),
                                size_bytes=size,
                                sha256=h,
                            )
                        )
                        total_bytes += size
                    elif src_path.is_dir():
                        for root, _, files in os.walk(src_path):
                            for fname in files:
                                fpath = Path(root) / fname
                                rel_to_src = fpath.relative_to(src_path)
                                arcname = f"{arc_prefix}/{rel_to_src}".replace("\\", "/")
                                h = compute_sha256(fpath)
                                size = fpath.stat().st_size
                                tar.add(fpath, arcname=arcname)
                                file_checksums.append(
                                    FileChecksum(
                                        relative_path=arcname,
                                        size_bytes=size,
                                        sha256=h,
                                    )
                                )
                                total_bytes += size

                # Also archive sanitized configuration snapshot
                cfg_dict = self.settings.to_safe_dict()
                cfg_bytes = json.dumps(cfg_dict, indent=2).encode("utf-8")
                cfg_tarinfo = tarfile.TarInfo(name="config_snapshot.json")
                cfg_tarinfo.size = len(cfg_bytes)
                cfg_tarinfo.mtime = int(time.time())
                import io

                tar.addfile(cfg_tarinfo, io.BytesIO(cfg_bytes))
                cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
                file_checksums.append(
                    FileChecksum(
                        relative_path="config_snapshot.json",
                        size_bytes=len(cfg_bytes),
                        sha256=cfg_hash,
                    )
                )
                total_bytes += len(cfg_bytes)

            archive_hash = compute_sha256(archive_path)

            manifest = BackupManifest(
                backup_id=backup_id,
                timestamp=datetime.now(tz=UTC).isoformat(),
                app_version=self.settings.app_version,
                category=category,
                description=description,
                files=file_checksums,
                total_files=len(file_checksums),
                total_bytes=total_bytes,
                archive_checksum=archive_hash,
            )

            manifest_path = target_backup_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(manifest.model_dump_json(indent=2))

            dur = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "Backup %s completed in %.2f ms (%d files, %d bytes)",
                backup_id,
                dur,
                len(file_checksums),
                total_bytes,
            )
            return BackupResult(
                success=True,
                backup_id=backup_id,
                backup_path=str(target_backup_dir),
                category=category,
                manifest=manifest,
                duration_ms=round(dur, 2),
            )

        except Exception as exc:
            # Clean up aborted backup
            if target_backup_dir.exists():
                shutil.rmtree(target_backup_dir, ignore_errors=True)
            dur = (time.perf_counter() - t0) * 1000.0
            logger.error("Backup %s failed: %s", backup_id, exc)
            return BackupResult(
                success=False,
                backup_id=backup_id,
                backup_path=str(target_backup_dir),
                category=category,
                duration_ms=round(dur, 2),
                error=str(exc),
            )

    def list_backups(self) -> list[BackupManifest]:
        """Enumerate all available backups sorted newest first."""
        manifests: list[BackupManifest] = []
        if not self.backup_root.exists():
            return []

        for item in sorted(self.backup_root.iterdir(), reverse=True):
            if item.is_dir():
                manifest_file = item / "manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, encoding="utf-8") as f:
                            data = json.load(f)
                        manifests.append(BackupManifest.model_validate(data))
                    except Exception as exc:
                        logger.warning("Corrupted backup manifest in %s: %s", item, exc)
        return manifests

    def verify_backup_integrity(self, backup_id: str) -> tuple[bool, list[str]]:
        """Verify archive SHA-256 and internal manifest integrity."""
        backup_dir = self.backup_root / backup_id
        manifest_file = backup_dir / "manifest.json"
        archive_file = backup_dir / "data.tar.gz"

        errors: list[str] = []
        if not backup_dir.exists():
            return False, [f"Backup directory not found: {backup_id}"]
        if not manifest_file.exists():
            return False, ["Missing manifest.json in backup directory"]
        if not archive_file.exists():
            return False, ["Missing data.tar.gz archive in backup directory"]

        try:
            with open(manifest_file, encoding="utf-8") as f:
                manifest = BackupManifest.model_validate(json.load(f))
        except Exception as exc:
            return False, [f"Failed to parse manifest: {exc}"]

        actual_arc_hash = compute_sha256(archive_file)
        if actual_arc_hash != manifest.archive_checksum:
            errors.append(
                f"Archive SHA-256 mismatch! Expected {manifest.archive_checksum}, got {actual_arc_hash}"
            )

        # Inspect members for Zip/Tar slip path traversal
        try:
            with tarfile.open(archive_file, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.startswith(("/", "\\")) or ".." in Path(member.name).parts:
                        errors.append(
                            f"Security hazard: Archive contains traversal path '{member.name}'"
                        )
        except Exception as exc:
            errors.append(f"Failed to inspect tar members: {exc}")

        return (len(errors) == 0), errors

    def preview_restore(self, backup_id: str) -> RestorePreview:
        """Generate a dry-run preview of files that would be restored."""
        valid, errors = self.verify_backup_integrity(backup_id)
        backup_dir = self.backup_root / backup_id
        manifest_file = backup_dir / "manifest.json"

        if not manifest_file.exists():
            return RestorePreview(
                backup_id=backup_id,
                valid_integrity=False,
                category=BackupCategory.FULL,
                files_to_restore=[],
                total_files=0,
                total_bytes=0,
                validation_errors=errors or ["Missing manifest"],
            )

        with open(manifest_file, encoding="utf-8") as f:
            manifest = BackupManifest.model_validate(json.load(f))

        return RestorePreview(
            backup_id=backup_id,
            valid_integrity=valid,
            category=manifest.category,
            files_to_restore=[f.relative_path for f in manifest.files],
            total_files=manifest.total_files,
            total_bytes=manifest.total_bytes,
            validation_errors=errors,
        )

    def restore_backup(
        self,
        backup_id: str,
        target_dir: Path | None = None,
        confirm: bool = False,
        create_pre_snapshot: bool = True,
    ) -> RestoreResult:
        """Safely restore backup into target directory after integrity validation."""
        t0 = time.perf_counter()
        destination = (target_dir or self.base_dir).resolve()

        if not confirm:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                target_dir=str(destination),
                restored_files_count=0,
                error="Restore operation aborted: explicit confirmation (confirm=True) required.",
            )

        valid, errors = self.verify_backup_integrity(backup_id)
        if not valid:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                target_dir=str(destination),
                restored_files_count=0,
                error=f"Integrity check failed: {'; '.join(errors)}",
            )

        archive_file = self.backup_root / backup_id / "data.tar.gz"

        snapshot_id: str | None = None
        # Create pre-restore snapshot if restoring into operational base directory
        if create_pre_snapshot and destination == self.base_dir:
            try:
                snap_res = self.create_backup(
                    category=BackupCategory.FULL,
                    description=f"Pre-restore safety snapshot before applying {backup_id}",
                )
                if snap_res.success:
                    snapshot_id = snap_res.backup_id
            except Exception as exc:
                logger.warning("Could not create pre-restore snapshot: %s", exc)

        restored_count = 0
        staging_dir: Path | None = None
        try:
            # Stage extraction in a clean temporary directory to guarantee transactional isolation
            staging_parent = destination.parent if destination.parent.exists() else destination
            staging_dir = Path(tempfile.mkdtemp(prefix=".restore_staging_", dir=staging_parent))

            with tarfile.open(archive_file, "r:gz") as tar:
                for member in tar.getmembers():
                    # Double-check path traversal security
                    if member.name.startswith(("/", "\\")) or ".." in Path(member.name).parts:
                        raise ValueError(f"Malicious member path rejected: {member.name}")

                    # Resolve staged file path
                    staged_path = staging_dir / member.name
                    if member.isdir():
                        staged_path.mkdir(parents=True, exist_ok=True)
                    else:
                        staged_path.parent.mkdir(parents=True, exist_ok=True)
                        extracted = tar.extractfile(member)
                        if extracted:
                            with open(staged_path, "wb") as out_f:
                                shutil.copyfileobj(extracted, out_f)

            # Move verified staged files into target destination
            for staged_item in staging_dir.rglob("*"):
                if staged_item.is_file():
                    rel = staged_item.relative_to(staging_dir)
                    target_file = destination / rel
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(staged_item), str(target_file))
                    restored_count += 1

            if staging_dir and staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

            dur = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "Restored %d files from %s into %s in %.2f ms",
                restored_count,
                backup_id,
                destination,
                dur,
            )
            return RestoreResult(
                success=True,
                backup_id=backup_id,
                target_dir=str(destination),
                restored_files_count=restored_count,
                pre_restore_snapshot_id=snapshot_id,
                duration_ms=round(dur, 2),
            )

        except Exception as exc:
            if staging_dir and staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            dur = (time.perf_counter() - t0) * 1000.0
            logger.error("Restore of %s failed: %s", backup_id, exc)
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                target_dir=str(destination),
                restored_files_count=restored_count,
                pre_restore_snapshot_id=snapshot_id,
                duration_ms=round(dur, 2),
                error=str(exc),
            )
