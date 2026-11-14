"""Unit tests for Phase 10 User Registry and Allowlist Management.

Tests:
- UserAccount model validation
- Deterministic user ID generation
- Registry persistence and thread-safe locking
- Allowlist checking and rejection of uninvited accounts
- OIDC claim authentication and initial administrator bootstrap
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphrag_bench.access_control import Role
from graphrag_bench.config import Settings
from graphrag_bench.user_registry import (
    UserAccount,
    UserRegistry,
    UserStatus,
    generate_user_id,
)


class TestUserRegistry:
    """User registry test suite."""

    def test_user_account_model(self) -> None:
        """Verify UserAccount model schema and defaults."""
        user = UserAccount(
            user_id="usr_abc123",
            email="researcher@test.com",
            name="Test Researcher",
            role=Role.RESEARCHER,
        )
        assert user.user_id == "usr_abc123"
        assert user.email == "researcher@test.com"
        assert user.status == UserStatus.ACTIVE
        assert user.created_at is not None
        assert user.last_login_at is None

    def test_generate_user_id_determinism(self) -> None:
        """Verify user ID generation is deterministic and properly formatted."""
        id1 = generate_user_id(
            issuer="https://accounts.google.com",
            subject="sub_123456",
            email="user@example.com",
        )
        id2 = generate_user_id(
            issuer="https://accounts.google.com",
            subject="sub_123456",
            email="user@example.com",
        )
        assert id1 == id2
        assert id1.startswith("usr_")
        assert len(id1) == 20  # "usr_" + 16 chars

    def test_create_and_get_user(self, tmp_path: Path) -> None:
        """Verify creating users, normalized lookups, and duplicate prevention."""
        reg_file = tmp_path / "users.json"
        registry = UserRegistry(registry_file=reg_file)

        user = registry.create_user(
            email="Alice.Smith@Example.COM",
            name="Alice Smith",
            role=Role.RESEARCHER,
            notes="Initial pilot tester",
        )
        assert user.email == "alice.smith@example.com"
        assert registry.count() == 1

        # Lookup by normalized email
        found = registry.get_user_by_email("alice.smith@example.com")
        assert found is not None
        assert found.user_id == user.user_id

        # Lookup by uppercase email
        found_upper = registry.get_user_by_email("ALICE.SMITH@EXAMPLE.COM")
        assert found_upper is not None

        # Duplicate email prevention
        with pytest.raises(ValueError, match="already registered"):
            registry.create_user(
                email="alice.smith@example.com",
                role=Role.VIEWER,
            )

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        """Verify JSON storage is safely written and reloaded."""
        reg_file = tmp_path / "users.json"
        reg1 = UserRegistry(registry_file=reg_file)
        user1 = reg1.create_user(email="user1@test.com", role=Role.ADMIN)
        reg1.create_user(email="user2@test.com", role=Role.RESEARCHER)
        assert reg1.count() == 2

        # Create new instance pointing to same file
        reg2 = UserRegistry(registry_file=reg_file)
        assert reg2.count() == 2
        loaded1 = reg2.get_user(user1.user_id)
        assert loaded1 is not None
        assert loaded1.email == "user1@test.com"
        assert loaded1.role == Role.ADMIN

    def test_update_role_and_status(self, tmp_path: Path) -> None:
        """Verify role updates, account suspension, and reactivation."""
        reg_file = tmp_path / "users.json"
        registry = UserRegistry(registry_file=reg_file)

        user = registry.create_user(email="bob@test.com", role=Role.VIEWER)
        assert user.role == Role.VIEWER
        assert user.status == UserStatus.ACTIVE

        # Update role
        updated_role = registry.update_role(user.user_id, Role.RESEARCHER)
        assert updated_role.role == Role.RESEARCHER

        # Suspend user
        suspended = registry.update_status(user.user_id, UserStatus.SUSPENDED)
        assert suspended.status == UserStatus.SUSPENDED

        # Reactivate user
        active = registry.update_status(user.user_id, UserStatus.ACTIVE)
        assert active.status == UserStatus.ACTIVE

    def test_delete_user(self, tmp_path: Path) -> None:
        """Verify user deletion from registry."""
        reg_file = tmp_path / "users.json"
        registry = UserRegistry(registry_file=reg_file)
        user = registry.create_user(email="charlie@test.com", role=Role.VIEWER)
        assert registry.count() == 1

        assert registry.delete_user(user.user_id)
        assert registry.count() == 0
        assert registry.get_user(user.user_id) is None

    def test_bootstrap_admin_with_configured_email(self, tmp_path: Path) -> None:
        """Verify that configured admin_bootstrap_email is provisioned as ADMIN."""
        reg_file = tmp_path / "users.json"
        settings = Settings(
            user_registry_path=str(reg_file),
            admin_bootstrap_email="bootstrap.admin@corp.org",
        )
        registry = UserRegistry(registry_file=reg_file, settings=settings)

        assert registry.is_bootstrap_admin("bootstrap.admin@corp.org")
        assert not registry.is_bootstrap_admin("regular.user@corp.org")

        claims = {
            "email": "bootstrap.admin@corp.org",
            "name": "Bootstrap Admin",
            "sub": "sub_bootstrap",
            "iss": "https://accounts.google.com",
            "email_verified": True,
        }
        user, err = registry.authenticate_or_provision(claims, settings=settings)
        assert err is None
        assert user is not None
        assert user.role == Role.ADMIN
        assert user.status == UserStatus.ACTIVE

    def test_allowlist_rejection_for_unregistered_user(self, tmp_path: Path) -> None:
        """Verify uninvited users are rejected when bootstrap email does not match."""
        reg_file = tmp_path / "users.json"
        settings = Settings(
            user_registry_path=str(reg_file),
            admin_bootstrap_email="bootstrap.admin@corp.org",
        )
        registry = UserRegistry(registry_file=reg_file, settings=settings)

        claims = {
            "email": "intruder@outside.org",
            "name": "Intruder",
            "sub": "sub_intruder",
            "iss": "https://accounts.google.com",
            "email_verified": True,
        }
        user, err = registry.authenticate_or_provision(claims, settings=settings)
        assert user is None
        assert "not on the invited pilot allowlist" in str(err)

    def test_rejection_for_suspended_user(self, tmp_path: Path) -> None:
        """Verify suspended allowlisted users cannot authenticate."""
        reg_file = tmp_path / "users.json"
        registry = UserRegistry(registry_file=reg_file)

        # Pre-register user as suspended
        registry.create_user(
            email="suspended@test.com",
            role=Role.RESEARCHER,
            status=UserStatus.SUSPENDED,
        )

        claims = {
            "email": "suspended@test.com",
            "sub": "sub_susp",
            "iss": "https://test.auth",
            "email_verified": True,
        }
        auth_user, err = registry.authenticate_or_provision(claims)
        assert auth_user is None
        assert "suspended" in str(err).lower()

    def test_rejection_for_unverified_email(self, tmp_path: Path) -> None:
        """Verify unverified emails are rejected when require_email_verified is True."""
        reg_file = tmp_path / "users.json"
        settings = Settings(
            user_registry_path=str(reg_file),
            require_email_verified=True,
        )
        registry = UserRegistry(registry_file=reg_file, settings=settings)

        claims = {
            "email": "unverified@test.com",
            "sub": "sub_unv",
            "iss": "https://test.auth",
            "email_verified": False,
        }
        auth_user, err = registry.authenticate_or_provision(claims, settings=settings)
        assert auth_user is None
        assert "not verified" in str(err)
