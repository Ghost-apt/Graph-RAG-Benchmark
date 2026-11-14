"""Unit tests for Phase 9 backup creation, manifest verification, and disaster recovery."""

import shutil
import tempfile
from pathlib import Path

import pytest

from graphrag_bench.backup_manager import (
    BackupCategory,
    BackupManager,
)
from graphrag_bench.config import Settings


class TestBackupManager:
    """Test suite for cryptographic backup creation and disaster recovery."""

    @pytest.fixture
    def test_env(self):
        """Create an isolated temporary repository environment."""
        tmp_dir = Path(tempfile.mkdtemp())
        data_dir = tmp_dir / "data"
        processed_dir = data_dir / "processed"
        kg_dir = data_dir / "knowledge_graph"
        backup_dir = data_dir / "backups"

        processed_dir.mkdir(parents=True, exist_ok=True)
        kg_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy sample files
        (processed_dir / "manifest.json").write_text('{"status": "ok"}', encoding="utf-8")
        (kg_dir / "knowledge_graph.json").write_text(
            '{"entities": [], "relationships": []}', encoding="utf-8"
        )

        cfg = Settings(
            data_dir="data",
            processed_dir="data/processed",
            graph_dir="data/knowledge_graph",
            backup_dir="data/backups",
            _env_file=None,
        )

        mgr = BackupManager(settings=cfg, base_dir=tmp_dir)
        yield mgr, tmp_dir

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_create_and_verify_backup(self, test_env):
        """Verify backup creation, file hashes, and manifest generation."""
        mgr, _tmp_dir = test_env
        res = mgr.create_backup(category=BackupCategory.CANONICAL, description="Test backup")
        assert res.success is True
        assert res.manifest is not None
        assert res.manifest.total_files >= 2
        assert res.manifest.total_bytes > 0
        assert len(res.manifest.archive_checksum) == 64

        # Verify integrity
        valid, errors = mgr.verify_backup_integrity(res.backup_id)
        assert valid is True
        assert len(errors) == 0

    def test_corrupted_archive_detection(self, test_env):
        """Verify that tampering with an archive is caught by SHA-256 verification."""
        mgr, _tmp_dir = test_env
        res = mgr.create_backup(category=BackupCategory.CANONICAL)
        assert res.success is True

        archive_file = Path(res.backup_path) / "data.tar.gz"
        # Tamper with archive bytes
        with open(archive_file, "ab") as f:
            f.write(b"CORRUPTED_BYTES")

        valid, errors = mgr.verify_backup_integrity(res.backup_id)
        assert valid is False
        assert any("Archive SHA-256 mismatch" in e for e in errors)

    def test_preview_restore(self, test_env):
        """Verify restore preview (dry-run) without modifying disk."""
        mgr, _tmp_dir = test_env
        res = mgr.create_backup(category=BackupCategory.CANONICAL)
        prev = mgr.preview_restore(res.backup_id)
        assert prev.valid_integrity is True
        assert prev.total_files == res.manifest.total_files
        assert "config_snapshot.json" in prev.files_to_restore

    def test_restore_requires_explicit_confirmation(self, test_env):
        """Verify that restore rejects execution without confirm=True."""
        mgr, _tmp_dir = test_env
        res = mgr.create_backup(category=BackupCategory.CANONICAL)
        rest = mgr.restore_backup(res.backup_id, confirm=False)
        assert rest.success is False
        assert "explicit confirmation" in rest.error

    def test_restore_to_alternate_directory(self, test_env):
        """Verify restoring backup safely into an isolated destination directory."""
        mgr, tmp_dir = test_env
        res = mgr.create_backup(category=BackupCategory.CANONICAL)
        restore_target = tmp_dir / "restored_output"

        rest = mgr.restore_backup(
            res.backup_id,
            target_dir=restore_target,
            confirm=True,
            create_pre_snapshot=False,
        )
        assert rest.success is True
        assert rest.restored_files_count >= 2

        # Verify restored files exist
        assert (restore_target / "data/processed/manifest.json").exists()
        assert (restore_target / "data/knowledge_graph/knowledge_graph.json").exists()
        assert (restore_target / "config_snapshot.json").exists()
