"""Unit tests for Phase 10 Role-Based Access Control (RBAC).

Tests:
- Role and Permission enums
- Least-privilege matrix
- Permission verification functions
- Decorator enforcement and exception raising
"""

from __future__ import annotations

import pytest

from graphrag_bench.access_control import (
    ROLE_PERMISSIONS,
    Permission,
    PermissionDeniedError,
    Role,
    check_permission,
    has_permission,
    normalize_permission,
    normalize_role,
    require_permission,
)


class DummyUser:
    """Mock user object for testing decorator user extraction."""

    def __init__(self, role: Role, user_id: str = "usr_test123") -> None:
        self.role = role
        self.user_id = user_id


class TestAccessControl:
    """RBAC validation suite."""

    def test_roles_and_normalization(self) -> None:
        """Verify role values and case-insensitive normalization."""
        assert Role.ADMIN == "admin"
        assert Role.RESEARCHER == "researcher"
        assert Role.VIEWER == "viewer"

        assert normalize_role("ADMIN") == Role.ADMIN
        assert normalize_role("Researcher") == Role.RESEARCHER
        assert normalize_role("viewer") == Role.VIEWER

        with pytest.raises(ValueError, match="Invalid role"):
            normalize_role("superuser")

    def test_permissions_count_and_normalization(self) -> None:
        """Verify all 15 granular permissions exist and normalize properly."""
        expected_permissions = [
            Permission.DOCUMENTS_READ,
            Permission.DOCUMENTS_UPLOAD,
            Permission.DOCUMENTS_DELETE,
            Permission.RETRIEVAL_EXECUTE,
            Permission.ANSWERS_GENERATE,
            Permission.ANSWERS_EXPORT,
            Permission.BENCHMARKS_READ,
            Permission.BENCHMARKS_EXECUTE,
            Permission.BENCHMARKS_EXPORT,
            Permission.OPERATIONS_READ,
            Permission.OPERATIONS_BACKUP,
            Permission.OPERATIONS_RESTORE,
            Permission.USERS_READ,
            Permission.USERS_MANAGE,
            Permission.SETTINGS_MANAGE,
        ]
        assert len(Permission) == 15
        for p in expected_permissions:
            assert normalize_permission(p.value) == p

    def test_role_permissions_matrix(self) -> None:
        """Verify least-privilege matrix integrity."""
        # Admin must have all 15 permissions
        assert len(ROLE_PERMISSIONS[Role.ADMIN]) == 15

        # Researcher must have retrieval, benchmark, read/upload documents, operations.read
        # but NOT operations.restore, operations.backup, users.manage, settings.manage
        res_perms = ROLE_PERMISSIONS[Role.RESEARCHER]
        assert Permission.RETRIEVAL_EXECUTE in res_perms
        assert Permission.ANSWERS_GENERATE in res_perms
        assert Permission.BENCHMARKS_EXECUTE in res_perms
        assert Permission.DOCUMENTS_UPLOAD in res_perms
        assert Permission.DOCUMENTS_DELETE not in res_perms
        assert Permission.OPERATIONS_RESTORE not in res_perms
        assert Permission.USERS_MANAGE not in res_perms

        # Viewer must be strictly read-only
        view_perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.DOCUMENTS_READ in view_perms
        assert Permission.RETRIEVAL_EXECUTE in view_perms
        assert Permission.ANSWERS_GENERATE in view_perms
        assert Permission.BENCHMARKS_READ in view_perms
        assert Permission.DOCUMENTS_UPLOAD not in view_perms
        assert Permission.DOCUMENTS_DELETE not in view_perms
        assert Permission.OPERATIONS_BACKUP not in view_perms

    def test_has_and_check_permission(self) -> None:
        """Verify check_permission raises PermissionDeniedError on denied actions."""
        # Granted actions
        assert has_permission(Role.ADMIN, Permission.USERS_MANAGE)
        check_permission(Role.ADMIN, Permission.USERS_MANAGE)

        # Denied actions
        assert not has_permission(Role.VIEWER, Permission.DOCUMENTS_UPLOAD)
        with pytest.raises(PermissionDeniedError) as exc_info:
            check_permission(Role.VIEWER, Permission.DOCUMENTS_UPLOAD, user_id="usr_viewer")

        err = exc_info.value
        assert err.permission == Permission.DOCUMENTS_UPLOAD
        assert err.role == Role.VIEWER
        assert err.user_id == "usr_viewer"
        assert "lacks required permission" in str(err)

    def test_require_permission_decorator_with_role(self) -> None:
        """Verify decorator with positional Role argument."""

        @require_permission(Permission.USERS_MANAGE)
        def manage_something(role: Role) -> str:
            return "success"

        assert manage_something(Role.ADMIN) == "success"

        with pytest.raises(PermissionDeniedError):
            manage_something(Role.RESEARCHER)

        with pytest.raises(PermissionDeniedError):
            manage_something(Role.VIEWER)

    def test_require_permission_decorator_with_user_object(self) -> None:
        """Verify decorator with DummyUser object having role attribute."""

        @require_permission(Permission.DOCUMENTS_UPLOAD)
        def upload_doc(user: DummyUser) -> str:
            return "uploaded"

        admin_user = DummyUser(role=Role.ADMIN)
        researcher_user = DummyUser(role=Role.RESEARCHER)
        viewer_user = DummyUser(role=Role.VIEWER)

        assert upload_doc(admin_user) == "uploaded"
        assert upload_doc(researcher_user) == "uploaded"

        with pytest.raises(PermissionDeniedError):
            upload_doc(viewer_user)

    def test_require_permission_decorator_missing_context(self) -> None:
        """Verify decorator raises when no user or role is supplied."""

        @require_permission(Permission.RETRIEVAL_EXECUTE)
        def unprotected_call() -> str:
            return "unreachable"

        with pytest.raises(PermissionDeniedError, match="No role or user context"):
            unprotected_call()
