"""Unit tests for Phase 10 AuthManager session lifecycle and authentication bridge.

Tests:
- Auth mode configuration
- Dev persona default account provisioning
- Inactivity session timeout calculation
- Logout event logging
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from graphrag_bench.access_control import Role
from graphrag_bench.auth_manager import AuthManager
from graphrag_bench.config import Settings
from graphrag_bench.user_audit import UserAuditLogger
from graphrag_bench.user_registry import UserRegistry


class TestAuthManager:
    """AuthManager test suite."""

    def test_auth_mode_property(self, tmp_path: Path) -> None:
        """Verify auth mode property normalization."""
        settings = Settings(
            auth_mode=" OIDC ",
            user_registry_path=str(tmp_path / "u.json"),
            user_audit_log_path=str(tmp_path / "a.jsonl"),
        )
        mgr = AuthManager(settings=settings)
        assert mgr.auth_mode == "oidc"

    def test_dev_mode_provisions_default_user(self, tmp_path: Path) -> None:
        """Verify dev mode automatically provisions and returns local admin account."""
        reg_file = tmp_path / "users.json"
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="dev",
            admin_bootstrap_email="dev.admin@local.test",
            user_registry_path=str(reg_file),
            user_audit_log_path=str(audit_file),
        )
        registry = UserRegistry(registry_file=reg_file, settings=settings)
        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, registry=registry, audit_logger=audit)

        user = mgr.get_current_user()
        assert user is not None
        assert user.email == "dev.admin@local.test"
        assert user.role == Role.ADMIN
        assert registry.count() == 1

    def test_session_timeout_detection(self, tmp_path: Path) -> None:
        """Verify session timeout detection against configured inactivity window."""
        reg_file = tmp_path / "users.json"
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="dev",
            session_timeout_minutes=30,
            user_registry_path=str(reg_file),
            user_audit_log_path=str(audit_file),
        )
        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, audit_logger=audit)

        # Mock streamlit session_state
        mock_session_state = {
            "_auth_login_time": time.time() - (35 * 60),  # 35 minutes ago
            "_current_user": mgr.get_current_user(),
        }

        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = mock_session_state
            is_expired = mgr.check_session_timeout()
            assert is_expired is True

            # Verify timeout audit event was logged
            events = audit.get_recent_events(limit=5)
            assert any(e.action == "AUTH_SESSION_EXPIRED" for e in events)

    def test_session_not_timed_out(self, tmp_path: Path) -> None:
        """Verify active session within timeout window is not expired."""
        reg_file = tmp_path / "users.json"
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="dev",
            session_timeout_minutes=60,
            user_registry_path=str(reg_file),
            user_audit_log_path=str(audit_file),
        )
        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, audit_logger=audit)

        mock_session_state = {
            "_auth_login_time": time.time() - (10 * 60),  # 10 minutes ago
            "_current_user": mgr.get_current_user(),
        }

        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = mock_session_state
            is_expired = mgr.check_session_timeout()
            assert is_expired is False

    def test_logout_logging(self, tmp_path: Path) -> None:
        """Verify logout logs AUTH_LOGOUT event."""
        reg_file = tmp_path / "users.json"
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="dev",
            user_registry_path=str(reg_file),
            user_audit_log_path=str(audit_file),
        )
        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, audit_logger=audit)

        mock_session_state = {
            "_current_user": mgr.get_current_user(),
        }

        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.logout = MagicMock()
            mock_st.rerun = MagicMock()

            mgr.logout()

            events = audit.get_recent_events(limit=5)
            assert any(e.action == "AUTH_LOGOUT" for e in events)

    def test_dev_persona_login_in_dev_mode(self, tmp_path: Path) -> None:
        """Verify dev persona switcher functions properly in dev mode."""
        reg_file = tmp_path / "users.json"
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="dev",
            user_registry_path=str(reg_file),
            user_audit_log_path=str(audit_file),
        )
        registry = UserRegistry(registry_file=reg_file, settings=settings)
        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, registry=registry, audit_logger=audit)

        mock_session_state: dict = {}
        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = mock_session_state
            user = mgr.set_dev_user("researcher@test.local", "Researcher Alice", Role.RESEARCHER)
            assert user.email == "researcher@test.local"
            assert user.role == Role.RESEARCHER
            assert mock_session_state["_current_user"].email == "researcher@test.local"

    def test_dev_persona_rejected_in_production_modes(self, tmp_path: Path) -> None:
        """Verify dev persona switching is strictly rejected in OIDC and none modes."""
        audit_file = tmp_path / "audit.jsonl"

        # 1. Reject in OIDC mode
        settings_oidc = Settings(
            auth_mode="oidc",
            user_registry_path=str(tmp_path / "u1.json"),
            user_audit_log_path=str(audit_file),
        )
        audit = UserAuditLogger(log_path=audit_file)
        mgr_oidc = AuthManager(settings=settings_oidc, audit_logger=audit)

        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = {}
            import pytest

            with pytest.raises(PermissionError, match="Development persona"):
                mgr_oidc.set_dev_user("hacker@test.local", "Malicious Dev", Role.ADMIN)

        # 2. Reject in none mode
        settings_none = Settings(
            auth_mode="none",
            user_registry_path=str(tmp_path / "u2.json"),
            user_audit_log_path=str(audit_file),
        )
        mgr_none = AuthManager(settings=settings_none, audit_logger=audit)
        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = {}
            with pytest.raises(PermissionError, match="Development persona"):
                mgr_none.set_dev_user("user@test.local", "Standard User", Role.ADMIN)

        # Verify audit rejection event logged
        events = audit.get_recent_events(limit=5)
        assert any(e.action == "AUTH_DEV_REJECTED" and e.status == "DENIED" for e in events)

    def test_missing_auth_config_denies_protected_access(self, tmp_path: Path) -> None:
        """Verify production fails closed when OIDC configuration is missing."""
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="oidc",
            user_registry_path=str(tmp_path / "u.json"),
            user_audit_log_path=str(audit_file),
        )
        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, audit_logger=audit)

        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = {}
            mock_st.stop = MagicMock()
            mock_st.error = MagicMock()
            mock_st.secrets = {}  # No auth secrets

            user = mgr.authenticate_streamlit()
            assert user is None
            assert mock_st.stop.called
            assert mock_st.error.called

            events = audit.get_recent_events(limit=5)
            assert any(
                e.action == "AUTH_FAILURE"
                and e.details.get("reason") == "missing_oidc_configuration"
                for e in events
            )

    def test_unapproved_oidc_user_denied(self, tmp_path: Path) -> None:
        """Verify non-allowlisted OIDC users are denied access."""
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="oidc",
            user_registry_path=str(tmp_path / "u.json"),
            user_audit_log_path=str(audit_file),
        )
        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, audit_logger=audit)

        class MockUser:
            email = "stranger@external.org"
            name = "Unknown Stranger"
            sub = "sub_unknown_999"
            iss = "https://accounts.google.com"
            email_verified = True
            is_logged_in = True

        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = {}
            mock_st.user = MockUser()
            mock_st.stop = MagicMock()
            mock_st.error = MagicMock()
            mock_st.secrets = {
                "auth": {
                    "client_id": "test_cid",
                    "client_secret": "test_csec",
                    "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
                }
            }

            user = mgr.authenticate_streamlit()
            assert user is None
            assert mock_st.stop.called
            assert mock_st.error.called

            events = audit.get_recent_events(limit=5)
            assert any(
                e.action == "AUTH_FAILURE" and e.details.get("reason") == "not_allowlisted"
                for e in events
            )

    def test_suspended_user_denied(self, tmp_path: Path) -> None:
        """Verify registered but suspended users are denied access."""
        reg_file = tmp_path / "users.json"
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="oidc",
            user_registry_path=str(reg_file),
            user_audit_log_path=str(audit_file),
        )
        registry = UserRegistry(registry_file=reg_file, settings=settings)
        # Register and suspend user
        from graphrag_bench.user_registry import UserStatus

        suspended_user = registry.register_user(
            email="suspended@company.com",
            name="Suspended Person",
            role=Role.RESEARCHER,
            status=UserStatus.SUSPENDED,
        )
        assert suspended_user.status == UserStatus.SUSPENDED

        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, registry=registry, audit_logger=audit)

        class MockUser:
            email = "suspended@company.com"
            name = "Suspended Person"
            sub = "sub_suspended_1"
            email_verified = True
            is_logged_in = True

        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = {}
            mock_st.user = MockUser()
            mock_st.stop = MagicMock()
            mock_st.error = MagicMock()
            mock_st.secrets = {
                "auth": {
                    "client_id": "test_cid",
                    "client_secret": "test_csec",
                    "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
                }
            }

            user = mgr.authenticate_streamlit()
            assert user is None
            assert mock_st.stop.called

            events = audit.get_recent_events(limit=5)
            assert any(
                e.action == "AUTH_FAILURE" and e.details.get("reason") == "account_suspended"
                for e in events
            )

    def test_oidc_issuer_and_subject_validation(self, tmp_path: Path) -> None:
        """Verify OIDC issuer mismatch and missing subject are rejected."""
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="oidc",
            oidc_issuer="https://trusted-idp.example.com",
            user_registry_path=str(tmp_path / "u.json"),
            user_audit_log_path=str(audit_file),
        )
        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, audit_logger=audit)

        # 1. Missing subject claim
        class MissingSubUser:
            email = "user@example.com"
            name = "No Sub User"
            sub = None
            oid = None
            is_logged_in = True

        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = {}
            mock_st.user = MissingSubUser()
            mock_st.stop = MagicMock()
            mock_st.error = MagicMock()
            mock_st.secrets = {
                "auth": {
                    "client_id": "test_cid",
                    "client_secret": "test_csec",
                    "server_metadata_url": "https://trusted-idp.example.com/.well-known/openid-configuration",
                }
            }

            user = mgr.authenticate_streamlit()
            assert user is None
            assert mock_st.stop.called
            events = audit.get_recent_events(limit=5)
            assert any(
                e.action == "AUTH_FAILURE" and e.details.get("reason") == "missing_subject_claim"
                for e in events
            )

        # 2. Issuer mismatch
        class UntrustedIssuerUser:
            email = "user@example.com"
            name = "Untrusted Issuer User"
            sub = "sub_untrusted"
            iss = "https://untrusted-rogue-idp.com"
            is_logged_in = True

        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = {}
            mock_st.user = UntrustedIssuerUser()
            mock_st.stop = MagicMock()
            mock_st.error = MagicMock()
            mock_st.secrets = {
                "auth": {
                    "client_id": "test_cid",
                    "client_secret": "test_csec",
                    "server_metadata_url": "https://trusted-idp.example.com/.well-known/openid-configuration",
                }
            }

            user = mgr.authenticate_streamlit()
            assert user is None
            assert mock_st.stop.called
            events = audit.get_recent_events(limit=5)
            assert any(
                e.action == "AUTH_FAILURE" and e.details.get("reason") == "issuer_mismatch"
                for e in events
            )

    def test_unverified_email_cannot_bootstrap_admin(self, tmp_path: Path) -> None:
        """Verify unverified email claim cannot bootstrap an administrator account."""
        reg_file = tmp_path / "users.json"
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="oidc",
            admin_bootstrap_email="claimed.admin@company.com",
            user_registry_path=str(reg_file),
            user_audit_log_path=str(audit_file),
        )
        registry = UserRegistry(registry_file=reg_file, settings=settings)
        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, registry=registry, audit_logger=audit)

        class UnverifiedAdminUser:
            email = "claimed.admin@company.com"
            name = "Impostor Admin"
            sub = "sub_impostor_123"
            email_verified = False  # Unverified claim!
            is_logged_in = True

        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = {}
            mock_st.user = UnverifiedAdminUser()
            mock_st.stop = MagicMock()
            mock_st.error = MagicMock()
            mock_st.secrets = {
                "auth": {
                    "client_id": "test_cid",
                    "client_secret": "test_csec",
                    "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
                }
            }

            user = mgr.authenticate_streamlit()
            assert user is None
            assert mock_st.stop.called
            assert registry.count() == 0  # No admin account created

            events = audit.get_recent_events(limit=5)
            assert any(
                e.action == "AUTH_FAILURE"
                and e.details.get("reason") == "unverified_admin_bootstrap_attempt"
                for e in events
            )

    def test_admin_bootstrap_strictly_requires_verified_identity(self, tmp_path: Path) -> None:
        """Verify admin bootstrap succeeds only with verified identity matching bootstrap email."""
        reg_file = tmp_path / "users.json"
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="oidc",
            admin_bootstrap_email="verified.admin@company.com",
            user_registry_path=str(reg_file),
            user_audit_log_path=str(audit_file),
        )
        registry = UserRegistry(registry_file=reg_file, settings=settings)
        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, registry=registry, audit_logger=audit)

        class VerifiedAdminUser:
            email = "verified.admin@company.com"
            name = "Genuine Admin"
            sub = "sub_genuine_admin"
            email_verified = True  # Strictly verified!
            is_logged_in = True

        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = {}
            mock_st.user = VerifiedAdminUser()
            mock_st.stop = MagicMock()
            mock_st.error = MagicMock()
            mock_st.secrets = {
                "auth": {
                    "client_id": "test_cid",
                    "client_secret": "test_csec",
                    "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
                }
            }

            user = mgr.authenticate_streamlit()
            assert user is not None
            assert user.email == "verified.admin@company.com"
            assert user.role == Role.ADMIN
            assert registry.count() == 1

            events = audit.get_recent_events(limit=5)
            assert any(e.action == "AUTH_LOGIN" and e.status == "SUCCESS" for e in events)

    def test_multi_user_session_isolation(self, tmp_path: Path) -> None:
        """Verify independent session isolation: state separation, no leakage, logout cleanup."""
        reg_file = tmp_path / "users.json"
        audit_file = tmp_path / "audit.jsonl"
        settings = Settings(
            auth_mode="dev",
            user_registry_path=str(reg_file),
            user_audit_log_path=str(audit_file),
        )
        registry = UserRegistry(registry_file=reg_file, settings=settings)
        audit = UserAuditLogger(log_path=audit_file)
        mgr = AuthManager(settings=settings, registry=registry, audit_logger=audit)

        # Session A: Researcher Alice
        session_a = {
            "_current_user": registry.register_user("alice@lab.org", "Alice", Role.RESEARCHER),
            "_auth_login_time": time.time(),
            "query_cache": {"q": "Alice private graph traversal query"},
        }

        # Session B: Viewer Bob
        session_b = {
            "_current_user": registry.register_user("bob@lab.org", "Bob", Role.VIEWER),
            "_auth_login_time": time.time(),
            "query_cache": {"q": "Bob public search"},
        }

        # Verify state separation
        assert session_a["_current_user"].email != session_b["_current_user"].email
        assert session_a["_current_user"].role == Role.RESEARCHER
        assert session_b["_current_user"].role == Role.VIEWER
        assert session_a["query_cache"]["q"] != session_b["query_cache"]["q"]

        # Logout Session A
        with patch("graphrag_bench.auth_manager.st") as mock_st:
            mock_st.session_state = session_a
            mock_st.logout = MagicMock()
            mock_st.rerun = MagicMock()
            mgr.logout()

            # Session A credentials removed
            assert "_current_user" not in session_a
            assert "_auth_login_time" not in session_a

        # Session B completely untouched and isolated
        assert session_b["_current_user"].email == "bob@lab.org"
        assert session_b["_current_user"].role == Role.VIEWER
        assert session_b["query_cache"]["q"] == "Bob public search"
