"""Unit tests for Phase 9 production health monitoring and readiness subsystem."""

from graphrag_bench.production_health import (
    ProductionHealthMonitor,
    ProductionHealthReport,
    ProductionHealthState,
)


class TestProductionHealth:
    """Test suite for decoupled liveness and readiness evaluation."""

    def test_liveness_check(self):
        """Verify process liveness probe."""
        monitor = ProductionHealthMonitor()
        item = monitor.check_liveness()
        assert item.state == ProductionHealthState.HEALTHY
        assert "Process alive on PID" in item.message
        assert item.details["pid"] > 0

    def test_config_check(self):
        """Verify configuration validation check."""
        monitor = ProductionHealthMonitor()
        item = monitor.check_config()
        assert item.state == ProductionHealthState.HEALTHY
        assert "validated successfully" in item.message

    def test_chromadb_check(self):
        """Verify ChromaDB persistent accessibility and collection counting."""
        monitor = ProductionHealthMonitor()
        chroma_item, coll_item = monitor.check_chromadb()
        assert chroma_item.state == ProductionHealthState.HEALTHY
        assert coll_item.state == ProductionHealthState.HEALTHY
        assert coll_item.details["chunk_count"] >= 0

    def test_knowledge_graph_check(self):
        """Verify Knowledge Graph integrity and canonical counts."""
        monitor = ProductionHealthMonitor()
        item = monitor.check_knowledge_graph()
        assert item.state == ProductionHealthState.HEALTHY
        assert item.details["entities"] == 22
        assert item.details["relationships"] == 20

    def test_document_repository_check(self):
        """Verify document repository manifest check."""
        monitor = ProductionHealthMonitor()
        item = monitor.check_document_repository()
        assert item.state == ProductionHealthState.HEALTHY
        assert item.details["has_manifest"] is True

    def test_disk_space_check(self):
        """Verify storage disk space evaluation."""
        monitor = ProductionHealthMonitor()
        item = monitor.check_disk_space()
        assert item.state in {ProductionHealthState.HEALTHY, ProductionHealthState.DEGRADED}
        assert item.details["free_gb"] > 0

    def test_benchmark_storage_check(self):
        """Verify Phase 8 benchmark storage evaluation."""
        monitor = ProductionHealthMonitor()
        item = monitor.check_benchmark_storage()
        assert item.state == ProductionHealthState.HEALTHY

    def test_run_all_checks_aggregation(self):
        """Verify consolidated health report compilation."""
        monitor = ProductionHealthMonitor()
        report = monitor.run_all_checks()
        assert isinstance(report, ProductionHealthReport)
        assert len(report.checks) == 13
        assert report.liveness is True
        assert report.readiness is True
        assert report.overall_state in {
            ProductionHealthState.HEALTHY,
            ProductionHealthState.DEGRADED,
        }
