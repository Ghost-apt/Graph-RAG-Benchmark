"""User Registry, Allowlist Management, and Admin Bootstrap Subsystem.

Provides:
1. UserAccount model and UserStatus lifecycle states.
2. Atomic JSON-backed user persistence.
3. Explicit allowlist enforcement (no unrestricted public self-registration).
4. Secure first-admin bootstrap logic based on verified OIDC claims.
5. Role assignment, account suspension, and access revocation.
"""

import hashlib
import json
import threading
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from graphrag_bench.access_control import Role, normalize_role
from graphrag_bench.atomic_io import atomic_write_json
from graphrag_bench.config import Settings, get_settings
from graphrag_bench.logger import get_logger

logger = get_logger("user_registry")


class UserStatus(str, Enum):
    """User account lifecycle status."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"


class UserAccount(BaseModel):
    """Authenticated user account record."""

    user_id: str = Field(description="Unique deterministic user identifier")
    email: str = Field(description="Normalized email address")
    name: str = Field(default="", description="Display name")
    role: Role = Field(default=Role.VIEWER, description="Assigned RBAC role")
    status: UserStatus = Field(default=UserStatus.ACTIVE, description="Account status")
    issuer: str = Field(default="", description="OIDC identity provider issuer URL")
    subject: str = Field(default="", description="OIDC subject identifier")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Account creation timestamp",
    )
    last_login_at: str | None = Field(default=None, description="Last authenticated login")
    notes: str = Field(default="", description="Administrative notes")

    @property
    def is_active(self) -> bool:
        return self.status == UserStatus.ACTIVE


def generate_user_id(issuer: str, subject: str, email: str) -> str:
    """Generate a deterministic, opaque user identifier."""
    seed = f"{issuer.strip().lower()}::{subject.strip()}::{email.strip().lower()}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"usr_{digest}"


class UserRegistry:
    """Thread-safe persistent registry managing authorized user accounts."""

    _instance: "UserRegistry | None" = None
    _lock = threading.Lock()

    def __init__(
        self,
        registry_file: Path | str | None = None,
        registry_path: Path | str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        target_path = registry_path or registry_file or self.settings.user_registry_path
        self.registry_path = Path(target_path).resolve()

        self._file_lock = threading.RLock()
        self._users: dict[str, UserAccount] = {}
        self._ensure_storage()
        self._load()

    @classmethod
    def get_instance(cls, registry_file: Path | str | None = None) -> "UserRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(registry_file)
            return cls._instance

    def _ensure_storage(self) -> None:
        """Ensure the parent storage directory exists."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            with self._file_lock:
                atomic_write_json(self.registry_path, {"users": []})

    def _load(self) -> None:
        """Load user accounts from persistent storage."""
        with self._file_lock:
            if not self.registry_path.exists():
                self._users = {}
                return

            try:
                with open(self.registry_path, encoding="utf-8") as f:
                    data = json.load(f)
                users_list = data.get("users", [])
                self._users = {u["user_id"]: UserAccount.model_validate(u) for u in users_list}
            except Exception as exc:
                logger.error("Failed to load user registry from %s: %s", self.registry_path, exc)
                self._users = {}

    def _save(self) -> None:
        """Persist user accounts atomically to disk."""
        with self._file_lock:
            data = {"users": [u.model_dump() for u in self._users.values()]}
            atomic_write_json(self.registry_path, data)

    def count(self) -> int:
        """Return total number of registered users."""
        with self._file_lock:
            return len(self._users)

    def list_users(self) -> list[UserAccount]:
        """List all registered users sorted by creation date."""
        with self._file_lock:
            return sorted(self._users.values(), key=lambda u: u.created_at)

    def get_user(self, user_id: str) -> UserAccount | None:
        """Retrieve user account by ID."""
        with self._file_lock:
            return self._users.get(user_id)

    def get_user_by_email(self, email: str) -> UserAccount | None:
        """Retrieve user account by normalized email."""
        norm_email = email.strip().lower()
        with self._file_lock:
            for u in self._users.values():
                if u.email.lower() == norm_email:
                    return u
        return None

    def get_user_by_oidc(self, issuer: str, subject: str) -> UserAccount | None:
        """Retrieve user by exact OIDC issuer and subject pair."""
        norm_iss = issuer.strip().lower()
        sub = subject.strip()
        with self._file_lock:
            for u in self._users.values():
                if u.issuer.lower() == norm_iss and u.subject == sub:
                    return u
        return None

    def create_user(
        self,
        email: str,
        name: str = "",
        role: Role | str = Role.VIEWER,
        issuer: str = "",
        subject: str = "",
        status: UserStatus = UserStatus.ACTIVE,
        notes: str = "",
    ) -> UserAccount:
        """Register a new user in the allowlist."""
        norm_email = email.strip().lower()
        existing = self.get_user_by_email(norm_email)
        if existing:
            raise ValueError(
                f"User with email '{email}' already registered with ID '{existing.user_id}'."
            )

        r = normalize_role(role)
        uid = generate_user_id(issuer, subject, norm_email)
        now_str = datetime.now(UTC).isoformat()

        account = UserAccount(
            user_id=uid,
            email=norm_email,
            name=name.strip(),
            role=r,
            status=status,
            issuer=issuer.strip(),
            subject=subject.strip(),
            created_at=now_str,
            notes=notes.strip(),
        )

        with self._file_lock:
            self._users[uid] = account
            self._save()

        logger.info("Registered user %s (%s) with role %s", uid, norm_email, r.value)
        return account

    def update_role(self, user_id: str, new_role: Role | str, actor: str = "admin") -> UserAccount:
        """Update a user's assigned RBAC role."""
        account = self.get_user(user_id)
        if not account:
            raise ValueError(f"User '{user_id}' not found.")

        r = normalize_role(new_role)
        old_role = account.role
        account.role = r

        with self._file_lock:
            self._users[user_id] = account
            self._save()

        logger.info("User %s role updated from %s to %s by %s", user_id, old_role, r.value, actor)
        return account

    def update_status(
        self, user_id: str, new_status: UserStatus, actor: str = "admin"
    ) -> UserAccount:
        """Update account status (e.g. suspend, activate, revoke)."""
        account = self.get_user(user_id)
        if not account:
            raise ValueError(f"User '{user_id}' not found.")

        old_status = account.status
        account.status = new_status

        with self._file_lock:
            self._users[user_id] = account
            self._save()

        logger.info(
            "User %s status updated from %s to %s by %s",
            user_id,
            old_status,
            new_status.value,
            actor,
        )
        return account

    def delete_user(self, user_id: str, actor: str = "admin") -> bool:
        """Permanently remove a user from the allowlist."""
        with self._file_lock:
            if user_id not in self._users:
                return False
            del self._users[user_id]
            self._save()

        logger.info("User %s deleted by %s", user_id, actor)
        return True

    def is_bootstrap_admin(self, email: str) -> bool:
        """Check if an email qualifies for automatic bootstrap as initial administrator.

        Strictly requires ADMIN_BOOTSTRAP_EMAIL to be configured and matched.
        """
        if not self.settings.admin_bootstrap_email:
            return False
        normalized = email.strip().lower()
        return self.settings.admin_bootstrap_email.strip().lower() == normalized

    def register_user(self, *args: Any, **kwargs: Any) -> UserAccount:
        """Alias for create_user."""
        return self.create_user(*args, **kwargs)

    def update_user_role(self, *args: Any, **kwargs: Any) -> UserAccount:
        """Alias for update_role."""
        return self.update_role(*args, **kwargs)

    def update_user_status(self, *args: Any, **kwargs: Any) -> UserAccount:
        """Alias for update_status."""
        return self.update_status(*args, **kwargs)

    def record_login(self, user_id: str) -> None:
        """Record login timestamp for user."""
        from datetime import UTC, datetime

        account = self.get_user(user_id)
        if account:
            account.last_login_at = datetime.now(UTC).isoformat()
            with self._file_lock:
                self._users[user_id] = account
                self._save()

    def authenticate_or_provision(
        self,
        claims: dict[str, Any],
        settings: Settings | None = None,
    ) -> tuple[UserAccount | None, str | None]:
        """Authenticate an incoming OIDC identity claim against the allowlist.

        Returns:
            (UserAccount, None) on successful authentication.
            (None, error_message) on rejection (unregistered, suspended, unverified).
        """
        cfg = settings or self.settings
        email = str(claims.get("email") or "").strip().lower()
        sub = str(claims.get("sub") or "").strip()
        iss = str(claims.get("iss") or "").strip().lower()
        name = str(claims.get("name") or claims.get("preferred_username") or email).strip()

        if not email:
            return None, "Authentication failed: Identity token is missing required 'email' claim."

        # Check email verification claim if required
        if cfg.require_email_verified:
            email_verified = claims.get("email_verified")
            # Some providers don't send email_verified (or send True as boolean/string)
            if email_verified is not None and email_verified is False:
                return (
                    None,
                    f"Authentication rejected: Email '{email}' is not verified with identity provider.",
                )

        # Check existing registration by OIDC sub/iss or email
        user = self.get_user_by_oidc(iss, sub) if (iss and sub) else None
        if not user:
            user = self.get_user_by_email(email)

        # Admin Bootstrap Condition:
        if cfg.admin_bootstrap_email:
            qualifies_bootstrap = cfg.admin_bootstrap_email.strip().lower() == email
        else:
            qualifies_bootstrap = self.count() == 0

        if not user and qualifies_bootstrap:
            # Bootstrap first admin
            logger.info("Bootstrapping initial administrator account for %s", email)
            user = self.create_user(
                email=email,
                name=name,
                role=Role.ADMIN,
                issuer=iss,
                subject=sub,
                status=UserStatus.ACTIVE,
                notes="Initial administrator bootstrap",
            )
            return user, None

        if not user:
            # Unregistered user rejected by default
            return None, (
                f"Access Denied: Account '{email}' is not on the invited pilot allowlist. "
                "Please contact your administrator to request access."
            )

        # Check account status
        if user.status == UserStatus.SUSPENDED:
            return None, f"Access Denied: Account '{email}' has been suspended by an administrator."

        if user.status == UserStatus.PENDING:
            return None, f"Access Denied: Account '{email}' is pending administrator approval."

        # Update OIDC linkage if not previously recorded
        if (not user.issuer or not user.subject) and (iss and sub):
            user.issuer = iss
            user.subject = sub

        # Update last login timestamp
        user.last_login_at = datetime.now(UTC).isoformat()
        if name and not user.name:
            user.name = name

        with self._file_lock:
            self._users[user.user_id] = user
            self._save()

        return user, None
