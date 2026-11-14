# Phase 9 Backup & Disaster Recovery Guide

**Project:** Graph RAG Benchmark  
**Current Version:** `0.8.0`  
**Target Version:** `0.9.0`  
**Phase:** 9 — Production Readiness & Deployment  
**Author:** Site Reliability Engineer & AI Infrastructure Architect  

---

## 1. Overview & Recovery Objectives

The **Graph RAG Benchmark** persistence layer holds both active operational data (ChromaDB vector indices, canonical knowledge graphs, ingested documents) and historical benchmark datasets.

### Recovery Objectives:
- **Recovery Point Objective (RPO):** At every significant ingestion or evaluation milestone.
- **Recovery Time Objective (RTO):** $< 60\text{ seconds}$ to full service restoration from validated backup archives.
- **Integrity Guarantee:** Every archive contains a cryptographically signed SHA-256 manifest. Restores will abort immediately if any bit-level discrepancy or archive corruption is detected.

---

## 2. Backup Scopes & Storage Structure

Backups are created in `data/backups/` under timestamped directories:
`data/backups/backup_<category>_YYYYMMDD_HHMMSS/`

Inside each backup directory:
- `manifest.json`: JSON document containing backup ID, category, timestamp, app version, uncompressed byte count, file count, archive SHA-256 hash, and individual file hashes.
- `data.tar.gz`: Compressed tarball containing relative path structures.

### Supported Categories:
1. **`canonical`**:
   - `data/processed/` (manifest, chunks, documents)
   - `data/knowledge_graph/` (knowledge_graph.json, graphml, extractions.jsonl)
   - `data/chroma/` (ChromaDB sqlite & parquet embeddings)
   - `data/uploads/` (raw uploaded files)
2. **`benchmark`**:
   - `data/phase8_benchmark/` (corpus, chunks, ground truth, dedicated chroma, dedicated graph, historical runs)
3. **`full`**:
   - All canonical data + all benchmark data + sanitized configuration snapshot.

---

## 3. Operational CLI Procedures

### 3.1 Creating Backups
```bash
# Full system backup (default)
python scripts/backup.py --category full --description "Weekly operational full backup"

# Canonical production data only
python scripts/backup.py --category canonical --description "Pre-ingestion canonical snapshot"

# Benchmark corpus & historical evaluation runs only
python scripts/backup.py --category benchmark --description "Post-benchmark run evaluation snapshot"
```

### 3.2 Listing & Inspecting Backups
```bash
# List all existing backups
python scripts/restore.py --list
```

### 3.3 Verifying Backup Integrity
```bash
# Verify checksums and TarSlip path safety without extracting
python scripts/restore.py --verify backup_full_20261114_140000
```

### 3.4 Dry-Run Restore Preview
```bash
# Preview what files would be restored without touching disk
python scripts/restore.py --preview backup_full_20261114_140000
```

### 3.5 Restoring to Alternate Directory (Testing / Staging)
```bash
# Restore safely to a temporary directory
python scripts/restore.py --restore backup_full_20261114_140000 --target-dir /tmp/test_restore --confirm
```

### 3.6 Restoring In-Place to Operational Repository
```bash
# Restores in-place; automatically takes a pre-restore safety snapshot first
python scripts/restore.py --restore backup_full_20261114_140000 --confirm
```

---

## 4. Disaster Recovery Scenarios & Mitigations

| Disaster Scenario | Impact | Recovery Procedure |
| :--- | :--- | :--- |
| **Accidental Ingestion Overwrite** | Canonical chunks or ChromaDB modified | Identify latest canonical backup -> Run `python scripts/restore.py --restore <id> --confirm` |
| **Host Disk Corruption** | SQLite or JSON files unreadable | Reinstall Python environment -> Restore full backup -> Run `python scripts/health_check.py` |
| **Benchmark Run Accidental Deletion** | Historical results missing | Restore benchmark category: `python scripts/restore.py --restore <bench_id> --confirm` |
| **TarSlip Traversal Attack Attempt** | Attacker tampers with archive archive paths | Built-in `verify_backup_integrity` rejects any archive containing `..` or leading `/` |
