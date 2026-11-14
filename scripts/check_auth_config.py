"""Validate Streamlit OIDC authentication configuration and secrets.

Usage:
    python scripts/check_auth_config.py [--strict]

Inspects .streamlit/secrets.toml for required OIDC parameters, validates syntax,
detects lingering placeholder credentials, and provides diagnostic guidance.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

# Ensure src is on Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.config import get_settings


def check_auth_configuration(strict: bool = False) -> int:
    """Validate auth configuration according to current settings."""
    settings = get_settings()
    auth_mode = settings.auth_mode.lower().strip()

    print("=" * 60)
    print("🔒 GRAPH RAG BENCHMARK - AUTHENTICATION CONFIG CHECK")
    print("=" * 60)
    print(f"Configured Auth Mode: {auth_mode.upper()}")

    if auth_mode in ("none", "dev"):
        print(f"ℹ️ Auth mode is '{auth_mode}'. External OIDC configuration is optional.")
        print("   Local user authentication and dev persona switching are enabled.")
        print("   Status: PASSED (Non-OIDC Mode)")
        return 0

    if auth_mode != "oidc":
        print(f"⚠️ Unrecognized auth mode '{auth_mode}'. Expected 'none', 'dev', or 'oidc'.")
        return 1

    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        print("❌ ERROR: Missing required secrets file: .streamlit/secrets.toml")
        print("\nRemediation:")
        print("1. Copy the template:")
        print("   copy .streamlit\\secrets.toml.example .streamlit\\secrets.toml")
        print("2. Edit .streamlit\\secrets.toml with your identity provider credentials:")
        print("   - client_id")
        print("   - client_secret")
        print("   - server_metadata_url (or authorization_endpoint & token_endpoint)")
        print("   - redirect_uri")
        return 1

    try:
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
    except Exception as e:
        print(f"❌ ERROR: Failed to parse .streamlit/secrets.toml syntax: {e}")
        return 1

    if "auth" not in secrets:
        print("❌ ERROR: Missing [auth] section in .streamlit/secrets.toml")
        print("   Refer to .streamlit/secrets.toml.example for schema structure.")
        return 1

    auth = secrets["auth"]
    required_keys = ["client_id", "client_secret"]
    missing = [k for k in required_keys if k not in auth or not str(auth[k]).strip()]
    if missing:
        print(f"❌ ERROR: Missing required auth keys in [auth]: {', '.join(missing)}")
        return 1

    # Check for metadata url or endpoints
    has_metadata = "server_metadata_url" in auth and str(auth["server_metadata_url"]).strip()
    has_endpoints = (
        "authorization_endpoint" in auth
        and "token_endpoint" in auth
        and str(auth["authorization_endpoint"]).strip()
        and str(auth["token_endpoint"]).strip()
    )

    if not (has_metadata or has_endpoints):
        print(
            "❌ ERROR: Must provide either 'server_metadata_url' or both 'authorization_endpoint' and 'token_endpoint'."
        )
        return 1

    # Check for lingering template placeholders
    client_id = str(auth.get("client_id", ""))
    client_secret = str(auth.get("client_secret", ""))

    placeholder_markers = ["your-", "YOUR_", "placeholder", "example.com", "00000000"]
    has_placeholders = any(
        m in client_id.lower() or m in client_secret.lower() for m in placeholder_markers
    )

    if has_placeholders:
        if strict:
            print(
                "❌ ERROR: Placeholder credentials detected in [auth] secrets under --strict mode."
            )
            return 1
        else:
            print("⚠️ WARNING: Detected placeholder strings in client credentials.")
            print(
                "   Ensure credentials are replaced with real OIDC credentials before public deployment."
            )

    print("✅ OpenID Connect configuration syntax valid:")
    print(f"   - Client ID: {client_id[:6]}... (length: {len(client_id)})")
    print(f"   - Client Secret: [Configured, length: {len(client_secret)}]")
    if has_metadata:
        print(f"   - Discovery URL: {auth['server_metadata_url']}")
    if "redirect_uri" in auth:
        print(f"   - Redirect URI: {auth['redirect_uri']}")

    print("\nStatus: PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Streamlit OIDC Authentication Config")
    parser.add_argument(
        "--strict", action="store_true", help="Fail if placeholder strings are found"
    )
    args = parser.parse_args()
    code = check_auth_configuration(strict=args.strict)
    sys.exit(code)


if __name__ == "__main__":
    main()
