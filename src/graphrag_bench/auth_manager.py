"""Authentication and session management module for Graph RAG Benchmark.

Bridges Streamlit native OIDC authentication (st.login, st.user, st.logout)
with the local UserRegistry, AccessControl (RBAC), and UserAuditLogger.
Supports fail-closed production OIDC, dev multi-user switching, and headless test modes.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from graphrag_bench.access_control import Role
from graphrag_bench.config import Settings, get_settings
from graphrag_bench.logger import get_logger
from graphrag_bench.user_audit import UserAuditLogger
from graphrag_bench.user_registry import UserAccount, UserRegistry, UserStatus

logger = get_logger(__name__)

# Try importing Streamlit safely so non-Streamlit environments (tests, CLI) don't crash
try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    st = None
    HAS_STREAMLIT = False


class AuthManager:
    """Manages user authentication, session lifecycle, and Streamlit auth integration."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: UserRegistry | None = None,
        audit_logger: UserAuditLogger | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or UserRegistry(settings=self.settings)
        self.audit_logger = audit_logger or UserAuditLogger(settings=self.settings)

    @property
    def auth_mode(self) -> str:
        """Normalized authentication mode."""
        return self.settings.auth_mode.lower().strip()

    def is_oidc_configured(self) -> bool:
        """Check whether OpenID Connect secrets and parameters are properly configured."""
        if HAS_STREAMLIT and st is not None:
            with contextlib.suppress(Exception):
                if hasattr(st, "secrets") and "auth" in st.secrets:
                    auth = st.secrets["auth"]
                    cid = str(auth.get("client_id", "")).strip()
                    csec = str(auth.get("client_secret", "")).strip()
                    meta = str(auth.get("server_metadata_url", "")).strip()
                    endpoints = (
                        str(auth.get("authorization_endpoint", "")).strip()
                        and str(auth.get("token_endpoint", "")).strip()
                    )
                    if cid and csec and (meta or endpoints):
                        return True

        secrets_file = Path(".streamlit/secrets.toml")
        if secrets_file.exists():
            with contextlib.suppress(Exception):
                with open(secrets_file, "rb") as f:
                    secrets = tomllib.load(f)
                auth = secrets.get("auth", {})
                cid = str(auth.get("client_id", "")).strip()
                csec = str(auth.get("client_secret", "")).strip()
                meta = str(auth.get("server_metadata_url", "")).strip()
                endpoints = (
                    str(auth.get("authorization_endpoint", "")).strip()
                    and str(auth.get("token_endpoint", "")).strip()
                )
                if cid and csec and (meta or endpoints):
                    return True

        return False

    def get_current_user(self) -> UserAccount | None:
        """Retrieve the currently authenticated user in the current execution context."""
        if self.auth_mode == "oidc":
            # In OIDC mode, fail-closed outside of valid authenticated session state
            if not HAS_STREAMLIT or st is None:
                return None
            with contextlib.suppress(Exception):
                user_data = st.session_state.get("_current_user")
                if isinstance(user_data, UserAccount):
                    return user_data
                elif isinstance(user_data, dict):
                    return UserAccount(**user_data)
            return None

        if not HAS_STREAMLIT or st is None:
            # Headless or test environment: check dev/none modes
            if self.auth_mode in ("none", "dev"):
                return self._get_default_dev_user()
            return None

        # Check Streamlit session_state safely
        with contextlib.suppress(Exception):
            if st.session_state.get("_current_user"):
                user_data = st.session_state["_current_user"]
                if isinstance(user_data, UserAccount):
                    return user_data
                elif isinstance(user_data, dict):
                    return UserAccount(**user_data)

        # In dev mode, return selected or default dev user
        if self.auth_mode in ("none", "dev"):
            dev_user = self._get_default_dev_user()
            with contextlib.suppress(Exception):
                st.session_state["_current_user"] = dev_user
            return dev_user

        return None

    def _get_default_dev_user(self, role: Role = Role.ADMIN) -> UserAccount:
        """Get or create a mock dev user for local development and headless testing."""
        if self.auth_mode == "oidc":
            raise PermissionError(
                "Default dev user provisioning is rejected in OIDC production mode."
            )
        dev_email = self.settings.admin_bootstrap_email or "admin@graphrag.local"
        user = self.registry.get_user_by_email(dev_email)
        if not user:
            user = self.registry.register_user(
                email=dev_email,
                name="Local Administrator",
                role=role,
                status=UserStatus.ACTIVE,
                notes="Automatic local dev account",
            )
        return user

    def set_dev_user(self, email: str, name: str, role: Role) -> UserAccount:
        """Switch current user in dev mode. Strictly prohibited in production modes."""
        if self.auth_mode != "dev":
            self.audit_logger.log_event(
                user_id="anonymous",
                email=email,
                role=role.value if isinstance(role, Role) else str(role),
                action="AUTH_DEV_REJECTED",
                status="DENIED",
                details={"mode": self.auth_mode, "reason": "dev_persona_prohibited_in_production"},
            )
            raise PermissionError(
                f"Development persona login is rejected when AUTH_MODE='{self.auth_mode}'. "
                "Development personas are strictly permitted only when AUTH_MODE='dev'."
            )

        if not HAS_STREAMLIT or st is None:
            raise RuntimeError("Streamlit session context required")

        user = self.registry.get_user_by_email(email)
        if not user:
            user = self.registry.register_user(
                email=email,
                name=name,
                role=role,
                status=UserStatus.ACTIVE,
                notes="Dev persona account",
            )
        elif user.role != role:
            user = self.registry.update_user_role(user.user_id, role)

        st.session_state["_current_user"] = user
        st.session_state["_auth_login_time"] = time.time()
        self.audit_logger.log_event(
            user_id=user.user_id,
            email=user.email,
            role=user.role.value,
            action="AUTH_LOGIN",
            status="SUCCESS",
            details={"mode": "dev_switch", "persona": name},
        )
        return user

    def check_session_timeout(self) -> bool:
        """Check if current session has timed out. Returns True if expired."""
        if not HAS_STREAMLIT or st is None:
            return False

        timeout_minutes = self.settings.session_timeout_minutes
        if timeout_minutes <= 0:
            return False  # Timeout disabled

        try:
            login_time = st.session_state.get("_auth_login_time")
        except Exception:
            return False

        if not login_time:
            return False

        elapsed_seconds = time.time() - login_time
        max_seconds = timeout_minutes * 60
        if elapsed_seconds > max_seconds:
            current_user = self.get_current_user()
            if current_user:
                self.audit_logger.log_event(
                    user_id=current_user.user_id,
                    email=current_user.email,
                    role=current_user.role.value,
                    action="AUTH_SESSION_EXPIRED",
                    status="SUCCESS",
                    details={
                        "elapsed_minutes": round(elapsed_seconds / 60, 1),
                        "timeout_minutes": timeout_minutes,
                    },
                )
            self._clear_session()
            return True
        return False

    def _clear_session(self) -> None:
        """Clear Streamlit session state credentials."""
        if HAS_STREAMLIT and st is not None:
            with contextlib.suppress(Exception):
                st.session_state.pop("_current_user", None)
                st.session_state.pop("_auth_login_time", None)
                st.session_state.pop("_auth_event_logged", None)

    def logout(self) -> None:
        """Log out the current user and reset session."""
        current_user = self.get_current_user()
        if current_user:
            self.audit_logger.log_event(
                user_id=current_user.user_id,
                email=current_user.email,
                role=current_user.role.value,
                action="AUTH_LOGOUT",
                status="SUCCESS",
                details={"mode": self.auth_mode},
            )

        self._clear_session()

        if HAS_STREAMLIT and st is not None:
            try:
                # Call native Streamlit logout if available
                if hasattr(st, "logout"):
                    st.logout()
            except Exception as e:
                logger.warning(f"st.logout() call encountered: {e}")
            st.rerun()

    def authenticate_streamlit(self) -> UserAccount | None:
        """Top-level authentication guard for the Streamlit app.

        Returns authenticated UserAccount if authorized.
        Halts execution via st.stop() or renders login screen if not authorized.
        """
        if not HAS_STREAMLIT or st is None:
            return self.get_current_user()

        # Check session timeout
        if self.check_session_timeout():
            st.warning("⏱️ Your session has expired due to inactivity. Please sign in again.")
            self.render_login_screen()
            st.stop()
            return None

        # Mode: None (single local user)
        if self.auth_mode == "none":
            return self.get_current_user()

        # Mode: Dev (Persona switcher)
        if self.auth_mode == "dev":
            user = self.get_current_user()
            if not user:
                user = self._get_default_dev_user()
                st.session_state["_current_user"] = user
                st.session_state["_auth_login_time"] = time.time()
            return user

        # Mode: OIDC (Native Streamlit OIDC)
        if self.auth_mode == "oidc":
            # Fail closed immediately if OIDC configuration is missing
            if not self.is_oidc_configured():
                st.error(
                    "🔒 **Access Denied**: OpenID Connect authentication is required (`AUTH_MODE=oidc`), "
                    "but identity provider configuration is missing in `.streamlit/secrets.toml`. "
                    "Protected access is denied."
                )
                self.audit_logger.log_event(
                    user_id="anonymous",
                    email="unknown",
                    role="NONE",
                    action="AUTH_FAILURE",
                    status="DENIED",
                    details={"mode": "oidc", "reason": "missing_oidc_configuration"},
                )
                st.stop()
                return None

            # Check native st.user
            st_user = getattr(st, "user", None)
            is_logged_in = getattr(st_user, "is_logged_in", False) if st_user else False

            if not is_logged_in:
                self.render_login_screen()
                st.stop()
                return None

            # User is logged in via OIDC, verify identity
            email = getattr(st_user, "email", None)
            name = getattr(st_user, "name", None) or (email.split("@")[0] if email else "User")
            sub = getattr(st_user, "sub", None) or getattr(st_user, "oid", None)
            iss = getattr(st_user, "iss", None)

            # Validate subject (sub)
            if not sub:
                st.error(
                    "Authentication Error: Identity provider token is missing required subject claim ('sub')."
                )
                self.audit_logger.log_event(
                    user_id="unknown",
                    email=email or "unknown",
                    role="NONE",
                    action="AUTH_FAILURE",
                    status="DENIED",
                    details={"reason": "missing_subject_claim"},
                )
                if hasattr(st, "logout"):
                    st.button("Try Again", on_click=st.logout)
                st.stop()
                return None

            # Validate issuer if configured
            expected_iss = getattr(self.settings, "oidc_issuer", None)
            if expected_iss and iss and iss.rstrip("/") != expected_iss.rstrip("/"):
                st.error(
                    "Authentication Error: Token issuer does not match configured trusted identity provider."
                )
                self.audit_logger.log_event(
                    user_id=sub,
                    email=email or "unknown",
                    role="NONE",
                    action="AUTH_FAILURE",
                    status="DENIED",
                    details={
                        "reason": "issuer_mismatch",
                        "expected": expected_iss,
                        "received": iss,
                    },
                )
                if hasattr(st, "logout"):
                    st.button("Try Again", on_click=st.logout)
                st.stop()
                return None

            if not email:
                st.error("Authentication Error: Identity provider did not return an email claim.")
                self.audit_logger.log_event(
                    user_id=sub,
                    email="unknown",
                    role="NONE",
                    action="AUTH_FAILURE",
                    status="DENIED",
                    details={"reason": "missing_email_claim"},
                )
                if hasattr(st, "logout"):
                    st.button("Try Again", on_click=st.logout)
                st.stop()
                return None

            # Verify email_verified claim
            email_verified = getattr(st_user, "email_verified", None)

            # Admin bootstrap claim with unverified email must be explicitly rejected
            if self.registry.is_bootstrap_admin(email) and email_verified is not True:
                st.error(
                    "Access Denied: Unverified email claims cannot bootstrap an administrator account."
                )
                self.audit_logger.log_event(
                    user_id=sub,
                    email=email,
                    role="NONE",
                    action="AUTH_FAILURE",
                    status="DENIED",
                    details={"reason": "unverified_admin_bootstrap_attempt"},
                )
                if hasattr(st, "logout"):
                    st.button("Sign Out", on_click=self.logout)
                st.stop()
                return None

            if self.settings.require_email_verified and email_verified is not True:
                st.error(
                    "Access Denied: Your email address has not been verified by your identity provider."
                )
                self.audit_logger.log_event(
                    user_id=sub,
                    email=email,
                    role="NONE",
                    action="AUTH_FAILURE",
                    status="DENIED",
                    details={"reason": "email_not_verified"},
                )
                if hasattr(st, "logout"):
                    st.button("Sign Out", on_click=self.logout)
                st.stop()
                return None

            # Check User Registry allowlist and status
            account = self.registry.get_user_by_email(email)

            if not account:
                # Check if this user qualifies for admin bootstrap
                if self.registry.is_bootstrap_admin(email):
                    # Admin bootstrap strictly requires verified identity
                    if email_verified is not True:
                        st.error(
                            "Access Denied: Unverified email claims cannot bootstrap an administrator account."
                        )
                        self.audit_logger.log_event(
                            user_id=sub,
                            email=email,
                            role="NONE",
                            action="AUTH_FAILURE",
                            status="DENIED",
                            details={"reason": "unverified_admin_bootstrap_attempt"},
                        )
                        if hasattr(st, "logout"):
                            st.button("Sign Out", on_click=self.logout)
                        st.stop()
                        return None

                    account = self.registry.register_user(
                        email=email,
                        name=name,
                        role=Role.ADMIN,
                        status=UserStatus.ACTIVE,
                        notes="Bootstrapped Initial Administrator",
                    )
                    logger.info(f"Bootstrapped initial admin: {email}")
                else:
                    # User not on allowlist
                    st.error(
                        f"🚫 **Access Restricted**: `{email}` is not authorized to access this pilot deployment.\n\n"
                        "Graph RAG Benchmark is currently operating in a controlled real-user pilot. "
                        "Please contact your administrator to request access."
                    )
                    self.audit_logger.log_event(
                        user_id=sub,
                        email=email,
                        role="NONE",
                        action="AUTH_FAILURE",
                        status="DENIED",
                        details={"reason": "not_allowlisted"},
                    )
                    if hasattr(st, "logout"):
                        st.button("Sign Out", on_click=self.logout)
                    st.stop()
                    return None

            # Check account status
            if account.status == UserStatus.SUSPENDED:
                st.error(
                    f"🚫 **Account Suspended**: Your account (`{email}`) has been suspended by an administrator."
                )
                self.audit_logger.log_event(
                    user_id=account.user_id,
                    email=email,
                    role=account.role.value,
                    action="AUTH_FAILURE",
                    status="DENIED",
                    details={"reason": "account_suspended"},
                )
                if hasattr(st, "logout"):
                    st.button("Sign Out", on_click=self.logout)
                st.stop()
                return None

            if account.status == UserStatus.PENDING:
                st.warning(
                    f"⏳ **Access Pending Approval**: Your account (`{email}`) is awaiting administrator review."
                )
                self.audit_logger.log_event(
                    user_id=account.user_id,
                    email=email,
                    role=account.role.value,
                    action="AUTH_FAILURE",
                    status="DENIED",
                    details={"reason": "account_pending_approval"},
                )
                if hasattr(st, "logout"):
                    st.button("Sign Out", on_click=self.logout)
                st.stop()
                return None

            # Store in session state
            st.session_state["_current_user"] = account
            if "_auth_login_time" not in st.session_state:
                st.session_state["_auth_login_time"] = time.time()

            # Record login audit event once per session
            if not st.session_state.get("_auth_event_logged"):
                self.registry.record_login(account.user_id)
                self.audit_logger.log_event(
                    user_id=account.user_id,
                    email=account.email,
                    role=account.role.value,
                    action="AUTH_LOGIN",
                    status="SUCCESS",
                    details={"provider": getattr(st_user, "provider", "oidc")},
                )
                st.session_state["_auth_event_logged"] = True

            return account

        # Fallback
        return self.get_current_user()

    def render_login_screen(self) -> None:
        """Render the welcome and sign-in page for unauthenticated visitors."""
        if not HAS_STREAMLIT or st is None:
            return

        st.markdown(
            """
            <div style="text-align: center; padding: 2.5rem 1rem 1rem 1rem;">
                <h1 style="font-size: 2.4rem; margin-bottom: 0.2rem;">🔬 Graph RAG Benchmark</h1>
                <p style="color: #64748b; font-size: 1.15rem; max-width: 650px; margin: 0 auto 1.5rem auto;">
                    Production-Grade Comparative Evaluation & Knowledge-Grounded Retrieval Platform
                </p>
                <div style="display: inline-block; padding: 0.25rem 0.8rem; border-radius: 9999px; background-color: #e0f2fe; color: #0369a1; font-weight: 600; font-size: 0.85rem; margin-bottom: 2rem;">
                    PILOT RELEASE v0.10.0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _col1, col2, _col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                ### 🔐 Secure Sign In
                This deployment is restricted to authorized pilot researchers and evaluators.
                Access is governed by enterprise Role-Based Access Control (RBAC) and attributable audit logging.
                """
            )

            if self.auth_mode == "dev":
                st.info("🛠️ **Development Mode Active**: Select a persona to test the platform:")
                persona = st.selectbox(
                    "Select Dev Persona",
                    options=[
                        "Admin (admin@graphrag.local) - Full Administrative Access",
                        "Researcher (researcher@graphrag.local) - Query & Benchmark Access",
                        "Viewer (viewer@graphrag.local) - Read-Only Dashboard Access",
                    ],
                    index=0,
                )
                if st.button(
                    "Sign In as Selected Persona", use_container_width=True, type="primary"
                ):
                    if "Admin" in persona:
                        self.set_dev_user("admin@graphrag.local", "Dev Administrator", Role.ADMIN)
                    elif "Researcher" in persona:
                        self.set_dev_user(
                            "researcher@graphrag.local", "Dev Researcher", Role.RESEARCHER
                        )
                    else:
                        self.set_dev_user("viewer@graphrag.local", "Dev Viewer", Role.VIEWER)
                    st.rerun()

            elif self.auth_mode == "oidc":
                st.markdown(
                    """
                    Click below to authenticate using your organization's OpenID Connect identity provider:
                    """
                )
                if hasattr(st, "login"):
                    if st.button(
                        "🔑 Sign in with Enterprise OIDC", use_container_width=True, type="primary"
                    ):
                        st.login()
                else:
                    st.warning("Streamlit native login is not enabled in this environment.")

            else:
                st.info(f"Current Auth Mode: `{self.auth_mode}`. Signing in automatically...")
                self._get_default_dev_user()
                st.rerun()

            st.markdown("---")
            st.caption(
                "🛡️ By signing in, you acknowledge that benchmark queries and system actions "
                "are logged for quality assurance and reproducibility. See the Data & Privacy Policy."
            )

    def render_user_badge(self, in_sidebar: bool = True) -> None:
        """Render active user indicator, role badge, dev persona switcher (if dev mode), and logout."""
        if not HAS_STREAMLIT or st is None:
            return

        user = self.get_current_user()
        if not user:
            return

        container = st.sidebar if in_sidebar else st

        # Role badge color formatting
        role_colors = {
            Role.ADMIN: ("#dc2626", "#fee2e2"),  # Red
            Role.RESEARCHER: ("#2563eb", "#dbeafe"),  # Blue
            Role.VIEWER: ("#16a34a", "#dcfce7"),  # Green
        }
        text_color, bg_color = role_colors.get(user.role, ("#475569", "#f1f5f9"))

        container.markdown(
            f"""
            <div style="padding: 0.6rem 0.8rem; border-radius: 8px; background-color: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.2); margin-bottom: 0.8rem;">
                <div style="font-size: 0.75rem; color: #64748b; font-weight: 500;">SIGNED IN AS</div>
                <div style="font-weight: 600; font-size: 0.9rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{user.name or user.email}</div>
                <div style="font-size: 0.75rem; color: #64748b; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{user.email}</div>
                <div style="margin-top: 0.4rem;">
                    <span style="background-color: {bg_color}; color: {text_color}; font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; text-transform: uppercase;">
                        {user.role.value}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # In dev mode, show persona switcher right in the sidebar
        if self.auth_mode == "dev":
            with container.expander("🔄 Switch Persona (Dev Mode)", expanded=False):
                col_a, col_b, col_c = st.columns(3)
                if col_a.button("Admin", key="btn_switch_admin", use_container_width=True):
                    self.set_dev_user("admin@graphrag.local", "Dev Administrator", Role.ADMIN)
                    st.rerun()
                if col_b.button("Researcher", key="btn_switch_res", use_container_width=True):
                    self.set_dev_user(
                        "researcher@graphrag.local", "Dev Researcher", Role.RESEARCHER
                    )
                    st.rerun()
                if col_c.button("Viewer", key="btn_switch_view", use_container_width=True):
                    self.set_dev_user("viewer@graphrag.local", "Dev Viewer", Role.VIEWER)
                    st.rerun()

        # Sign out button
        if self.auth_mode != "none" and container.button(
            "🚪 Sign Out", key="auth_btn_logout", use_container_width=True
        ):
            self.logout()
