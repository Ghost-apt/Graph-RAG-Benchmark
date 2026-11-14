"""Role-Based Access Control (RBAC) and Permission Enforcement Engine.

Provides:
1. Canonical user roles (ADMIN, RESEARCHER, VIEWER).
2. Granular functional permissions for all application operations.
3. Least-privilege role-to-permission mappings.
4. Server-side permission check functions and decorators.
"""

from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any


class Role(str, Enum):
    """Canonical user roles for Graph RAG Benchmark."""

    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Granular operation permissions."""

    # Document management
    DOCUMENTS_READ = "documents.read"
    DOCUMENTS_UPLOAD = "documents.upload"
    DOCUMENTS_DELETE = "documents.delete"

    # Retrieval and Question Answering
    RETRIEVAL_EXECUTE = "retrieval.execute"
    ANSWERS_GENERATE = "answers.generate"
    ANSWERS_EXPORT = "answers.export"

    # Comparative Benchmarks
    BENCHMARKS_READ = "benchmarks.read"
    BENCHMARKS_EXECUTE = "benchmarks.execute"
    BENCHMARKS_EXPORT = "benchmarks.export"

    # Production Operations & Maintenance
    OPERATIONS_READ = "operations.read"
    OPERATIONS_BACKUP = "operations.backup"
    OPERATIONS_RESTORE = "operations.restore"

    # User & Identity Management
    USERS_READ = "users.read"
    USERS_MANAGE = "users.manage"

    # Configuration Management
    SETTINGS_MANAGE = "settings.manage"


# Explicit least-privilege permission matrix
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: set(Permission),
    Role.RESEARCHER: {
        Permission.DOCUMENTS_READ,
        Permission.DOCUMENTS_UPLOAD,
        Permission.RETRIEVAL_EXECUTE,
        Permission.ANSWERS_GENERATE,
        Permission.ANSWERS_EXPORT,
        Permission.BENCHMARKS_READ,
        Permission.BENCHMARKS_EXECUTE,
        Permission.BENCHMARKS_EXPORT,
        Permission.OPERATIONS_READ,
    },
    Role.VIEWER: {
        Permission.DOCUMENTS_READ,
        Permission.RETRIEVAL_EXECUTE,
        Permission.ANSWERS_GENERATE,
        Permission.ANSWERS_EXPORT,
        Permission.BENCHMARKS_READ,
        Permission.BENCHMARKS_EXPORT,
    },
}


class PermissionDeniedError(Exception):
    """Raised when an authenticated principal attempts an unauthorized operation."""

    def __init__(
        self,
        permission: Permission | str,
        role: Role | str,
        user_id: str | None = None,
        message: str | None = None,
    ) -> None:
        self.permission = permission
        self.role = role
        self.user_id = user_id
        msg = (
            message
            or f"Access denied: User '{user_id or 'anonymous'}' with role '{role}' "
            f"lacks required permission '{permission}'."
        )
        super().__init__(msg)


def normalize_role(role_val: Role | str) -> Role:
    """Normalize input string to Role enum."""
    if isinstance(role_val, Role):
        return role_val
    try:
        return Role(str(role_val).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"Invalid role '{role_val}'. Valid roles: {[r.value for r in Role]}"
        ) from exc


def normalize_permission(perm_val: Permission | str) -> Permission:
    """Normalize input string to Permission enum."""
    if isinstance(perm_val, Permission):
        return perm_val
    try:
        return Permission(str(perm_val).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"Invalid permission '{perm_val}'. Valid permissions: {[p.value for p in Permission]}"
        ) from exc


def has_permission(role: Role | str, permission: Permission | str) -> bool:
    """Check whether a given role holds a specific permission."""
    try:
        r = normalize_role(role)
        p = normalize_permission(permission)
    except ValueError:
        return False
    return p in ROLE_PERMISSIONS.get(r, set())


def check_permission(
    role: Role | str,
    permission: Permission | str,
    user_id: str | None = None,
) -> None:
    """Enforce permission check, raising PermissionDeniedError if not granted."""
    r = normalize_role(role)
    p = normalize_permission(permission)
    if not has_permission(r, p):
        raise PermissionDeniedError(permission=p, role=r, user_id=user_id)


def require_permission(permission: Permission | str) -> Callable:
    """Decorator for server-side function authorization enforcement.

    Expects decorated function to receive a 'user' or 'role' parameter, or
    evaluates against the thread-local context.
    """
    p = normalize_permission(permission)

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check user keyword argument
            user = kwargs.get("user")
            role = kwargs.get("role")
            user_id = None

            if user is not None:
                if hasattr(user, "role"):
                    role = user.role
                if hasattr(user, "user_id"):
                    user_id = user.user_id
            elif role is None and args:
                # First positional arg could be Role directly, or user/self with role attribute
                first_arg = args[0]
                if isinstance(first_arg, Role):
                    role = first_arg
                elif hasattr(first_arg, "role"):
                    role = first_arg.role
                if hasattr(first_arg, "user_id"):
                    user_id = first_arg.user_id

            if role is None:
                raise PermissionDeniedError(
                    permission=p,
                    role="none",
                    user_id=user_id,
                    message=f"Authorization failure: No role or user context provided for protected operation requiring '{p.value}'.",
                )

            check_permission(role=role, permission=p, user_id=user_id)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
