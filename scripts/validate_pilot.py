"""Automated Validation CLI for Phase 10 Pilot Deployment.

Validates Role-Based Access Control, user registry lifecycle, audit logging,
credential sanitization, and session governance without running a browser.

Usage:
    python scripts/validate_pilot.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure src is on Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.access_control import (
    ROLE_PERMISSIONS,
    Permission,
    PermissionDeniedError,
    Role,
    has_permission,
    require_permission,
)
from graphrag_bench.auth_manager import AuthManager
from graphrag_bench.config import Settings
from graphrag_bench.user_audit import UserAuditLogger, sanitize_audit_details
from graphrag_bench.user_registry import UserRegistry, UserStatus


def validate_rbac() -> bool:
    """Verify Role-Based Access Control matrix and enforcement."""
    print("  [1/5] Testing Role-Based Access Control matrix...")

    # Admin must have all permissions
    admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
    assert len(admin_perms) == len(Permission), "Admin must have all permissions"

    # Researcher must have query and benchmark but NOT users_manage or operations_restore
    assert has_permission(Role.RESEARCHER, Permission.RETRIEVAL_EXECUTE)
    assert has_permission(Role.RESEARCHER, Permission.ANSWERS_GENERATE)
    assert has_permission(Role.RESEARCHER, Permission.BENCHMARKS_EXECUTE)
    assert not has_permission(Role.RESEARCHER, Permission.USERS_MANAGE)
    assert not has_permission(Role.RESEARCHER, Permission.OPERATIONS_RESTORE)

    # Viewer must have read-only permissions
    assert has_permission(Role.VIEWER, Permission.RETRIEVAL_EXECUTE)
    assert has_permission(Role.VIEWER, Permission.DOCUMENTS_READ)
    assert not has_permission(Role.VIEWER, Permission.DOCUMENTS_UPLOAD)
    assert not has_permission(Role.VIEWER, Permission.DOCUMENTS_DELETE)
    assert not has_permission(Role.VIEWER, Permission.OPERATIONS_BACKUP)

    # Test decorator
    @require_permission(Permission.USERS_MANAGE)
    def admin_only_func(role: Role) -> str:
        return "admin_success"

    assert admin_only_func(Role.ADMIN) == "admin_success"
    try:
        admin_only_func(Role.RESEARCHER)
        assert False, "Should have raised PermissionDeniedError"
    except PermissionDeniedError:
        pass

    print("        ✅ RBAC hierarchy, permissions, and decorators verified.")
    return True


def validate_user_registry(temp_dir: Path) -> bool:
    """Verify user registry CRUD, allowlisting, and bootstrap."""
    print("  [2/5] Testing User Registry and allowlist governance...")
    reg_path = temp_dir / "users.json"
    settings = Settings(
        user_registry_path=str(reg_path),
        admin_bootstrap_email="pilot.admin@organization.com",
    )
    registry = UserRegistry(registry_path=reg_path, settings=settings)

    # Test bootstrap admin identification
    assert registry.is_bootstrap_admin("pilot.admin@organization.com")
    assert not registry.is_bootstrap_admin("evaluator@organization.com")

    # Register admin
    admin = registry.register_user(
        email="pilot.admin@organization.com",
        name="Pilot Admin",
        role=Role.ADMIN,
        status=UserStatus.ACTIVE,
    )
    assert admin.role == Role.ADMIN
    assert admin.status == UserStatus.ACTIVE

    # Register researcher
    res = registry.register_user(
        email="evaluator.one@organization.com",
        name="Evaluator One",
        role=Role.RESEARCHER,
        status=UserStatus.ACTIVE,
    )
    assert res.role == Role.RESEARCHER

    # Update role
    updated_res = registry.update_user_role(res.user_id, Role.VIEWER)
    assert updated_res.role == Role.VIEWER

    # Suspend user
    suspended_res = registry.update_user_status(res.user_id, UserStatus.SUSPENDED)
    assert suspended_res.status == UserStatus.SUSPENDED

    # Persistence verification
    registry_reloaded = UserRegistry(registry_path=reg_path, settings=settings)
    assert len(registry_reloaded.list_users()) == 2
    reloaded_admin = registry_reloaded.get_user_by_email("pilot.admin@organization.com")
    assert reloaded_admin is not None and reloaded_admin.role == Role.ADMIN

    print("        ✅ User registry CRUD, persistence, and state updates verified.")
    return True


def validate_audit_logging(temp_dir: Path) -> bool:
    """Verify JSONL attributable audit logging and credential redaction."""
    print("  [3/5] Testing Attributable Audit Logging and sanitization...")
    audit_path = temp_dir / "test_user_audit.jsonl"
    audit_logger = UserAuditLogger(log_path=audit_path)

    # Log clean event
    e1 = audit_logger.log_event(
        user_id="user-123",
        email="evaluator@test.com",
        role=Role.RESEARCHER.value,
        action="QUERY_EXECUTION",
        status="SUCCESS",
        details={"mode": "hybrid", "query_length": 42},
    )
    assert e1.action == "QUERY_EXECUTION"

    # Log event with sensitive payload to test sanitization
    e2 = audit_logger.log_event(
        user_id="user-123",
        email="evaluator@test.com",
        role=Role.RESEARCHER.value,
        action="AUTH_LOGIN",
        status="SUCCESS",
        details={
            "token": "secret-jwt-123",
            "password": "my-password",
            "access_token": "bearer-abc",
            "normal_key": "safe_value",
        },
    )
    assert e2.details["token"] == "[REDACTED]"
    assert e2.details["password"] == "[REDACTED]"
    assert e2.details["access_token"] == "[REDACTED]"
    assert e2.details["normal_key"] == "safe_value"

    # Query events back
    events = audit_logger.get_recent_events(limit=10)
    assert len(events) == 2
    assert events[0].action == "AUTH_LOGIN"  # Latest first
    assert events[1].action == "QUERY_EXECUTION"

    # Verify summary stats
    stats = audit_logger.get_summary_statistics()
    assert stats["total_events"] == 2
    assert stats["actions"]["AUTH_LOGIN"] == 1
    assert stats["actions"]["QUERY_EXECUTION"] == 1

    print("        ✅ Audit logging append, query, and zero-leakage redaction verified.")
    return True


def validate_sanitization_helper() -> bool:
    """Verify recursive sanitization helper."""
    print("  [4/5] Testing payload sanitization recursion...")
    dirty = {
        "auth_header": "Bearer xyz",
        "nested": {
            "api_key": "12345",
            "user": "alice",
            "deep": [{"client_secret": "shh"}, {"id": 10}],
        },
    }
    clean = sanitize_audit_details(dirty)
    assert clean["auth_header"] == "[REDACTED]"
    assert clean["nested"]["api_key"] == "[REDACTED]"
    assert clean["nested"]["user"] == "alice"
    assert clean["nested"]["deep"][0]["client_secret"] == "[REDACTED]"
    assert clean["nested"]["deep"][1]["id"] == 10
    print("        ✅ Recursive sanitization verified.")
    return True


def validate_auth_manager_dev_mode(temp_dir: Path) -> bool:
    """Verify AuthManager dev persona switching and default users."""
    print("  [5/5] Testing AuthManager in dev persona mode...")
    reg_path = temp_dir / "dev_users.json"
    audit_path = temp_dir / "dev_audit.jsonl"
    settings = Settings(
        auth_mode="dev",
        user_registry_path=str(reg_path),
        user_audit_log_path=str(audit_path),
        admin_bootstrap_email="dev.admin@graphrag.local",
    )
    mgr = AuthManager(settings=settings)

    user = mgr.get_current_user()
    assert user is not None
    assert user.role == Role.ADMIN
    assert user.email == "dev.admin@graphrag.local"

    print("        ✅ AuthManager headless dev persona verified.")
    return True


def main() -> None:
    print("=" * 65)
    print("🔬 GRAPH RAG BENCHMARK - PILOT VALIDATION TEST RUNNER")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as td:
        t_path = Path(td)
        assert validate_rbac()
        assert validate_user_registry(t_path)
        assert validate_audit_logging(t_path)
        assert validate_sanitization_helper()
        assert validate_auth_manager_dev_mode(t_path)

    print("=" * 65)
    print("🎉 ALL PILOT VALIDATION CHECKS PASSED (5/5) — ZERO FAILURES")
    print("=" * 65)


if __name__ == "__main__":
    main()
