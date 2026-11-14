# Graph RAG Benchmark — Phase 10 Authentication Architecture

**Document Version:** `1.0.0`  
**System Target Release:** `v0.10.0` (Phase 10 — Real-User Pilot Deployment & Identity Management)  
**Classification:** Technical Architecture Specification  
**Status:** Approved & Implemented  

---

## 1. Executive Summary

Graph RAG Benchmark Phase 10 transitions the platform from a local-only prototype into an enterprise-grade, identity-governed pilot application. This document details the end-to-end authentication architecture, protocol choices, security perimeters, session governance models, and provider integration semantics.

The authentication subsystem is designed to meet strict security standards:
1. **Zero-Trust Fail-Closed Security:** Unauthenticated, uninvited, or suspended visitors are blocked before accessing dashboard tabs, vector collections, knowledge graphs, or execution engines.
2. **Standardized Identity Federation:** Integrates directly with OpenID Connect (OIDC) Core 1.0 using the Authorization Code Flow with Proof Key for Code Exchange (PKCE).
3. **Defense-in-Depth Layering:** Authentication is separated into network perimeter termination (TLS reverse proxy), application session validation (Streamlit OIDC bridge), and server-side RBAC enforcement.
4. **Attributable Audit Logging:** Every authentication lifecycle event (`AUTH_LOGIN`, `AUTH_LOGOUT`, `AUTH_FAILURE`, `AUTH_SESSION_EXPIRED`) is logged to structured append-only JSONL with zero credential leakage.

---

## 2. Authentication Protocol Specification

### 2.1 Protocol Choice: OpenID Connect (OIDC) Core 1.0

The application leverages OpenID Connect (built atop OAuth 2.0), the industry-standard identity federation layer. 

```
+-----------------------------------------------------------------------------------+
|                                 OIDC Authorization Code Flow                       |
+-----------------------------------------------------------------------------------+

[ User Browser ]         [ Reverse Proxy (TLS) ]      [ Streamlit Pilot ]      [ Identity Provider (IdP) ]
       |                            |                         |                              |
       |--- 1. GET / -------------->|                         |                              |
       |                            |--- 2. Forward --------->|                              |
       |                            |                         |-- 3. Check Session Cookie -> |
       |                            |                         |   (Not Authenticated)        |
       |<-- 4. 302 Redirect to IdP Auth Endpoint -------------|                              |
       |                                                                                     |
       |--- 5. Authenticate with Credentials / MFA / SSO ----------------------------------->|
       |<-- 6. 302 Redirect with Auth Code (?code=XYZ&state=ABC) ----------------------------|
       |                                                                                     |
       |--- 7. GET /oauth2callback?code=XYZ&state=ABC -------->|                              |
       |                            |                         |-- 8. POST /token (Backend) ->|
       |                            |                         |<-- 9. ID Token + Access Token|
       |                            |                         |-- 10. Verify Signature & Claims
       |                            |                         |-- 11. Match Against Allowlist
       |                            |                         |-- 12. Set Encrypted Cookie   |
       |<-- 13. Render Pilot Dashboard with User Badge --------|                              |
```

### 2.2 Token Verification Semantics
Upon token retrieval, the authentication engine validates:
1. **Signature Integrity:** Cryptographically verified against the provider's JSON Web Key Set (JWKS).
2. **Issuer (`iss`):** Exact match against the trusted IdP discovery document.
3. **Audience (`aud`):** Matches the client ID configured in `.streamlit/secrets.toml`.
4. **Time Boundaries:**
   - `nbf` (Not Before): Token cannot be used before issued.
   - `exp` (Expiration): Token rejected if expired.
   - Clock skew tolerance: Maximum 60 seconds.
5. **Email Verification (`email_verified`):** When `REQUIRE_EMAIL_VERIFIED=true`, claims with `email_verified=false` are rejected with `AUTH_FAILURE`.

---

## 3. Deployment Modes

The platform supports three distinct authentication modes configured via `GRAPHRAG_AUTH_MODE` in `Settings`:

| Mode | Environment | Target Use Case | Identity Source |
| :--- | :--- | :--- | :--- |
| `dev` | Local Workstation | Development, UI testing, QA automation | Pre-configured persona switcher (`ADMIN`, `RESEARCHER`, `VIEWER`) |
| `oidc` | Production Pilot | Real-user pilot deployment, multi-user research | Enterprise IdP (Google, Azure AD / Entra, Okta, Auth0) |
| `none` | Air-Gapped / Test | Automated benchmark harnesses, headless scripts | Headless system administrator account |

### 3.1 Production Mode (`oidc`)
In production pilot mode, Streamlit native authentication (`st.login()`, `st.user`, `st.logout()`) manages the browser handshake. The application verifies incoming claims against the local `UserRegistry` allowlist. Unregistered emails are blocked at the welcome screen.

### 3.2 Development Mode (`dev`)
In development mode, no external internet connection or OAuth client credentials are required. An interactive persona switcher in the sidebar allows developers and QA engineers to switch instantly between Admin, Researcher, and Viewer personas, verifying RBAC enforcement across all 9 tabs.

---

## 4. Identity Provider (IdP) Integration Matrix

The pilot architecture is vendor-agnostic and supports any OpenID Connect certified provider. Configuration is managed securely via `.streamlit/secrets.toml`:

### 4.1 Google Workspace / Google Cloud Identity
- **Discovery URL:** `https://accounts.google.com/.well-known/openid-configuration`
- **Supported Scopes:** `openid`, `email`, `profile`
- **Subject Identifier:** Google Account Unique ID (`sub`)

```toml
[auth]
redirect_uri = "https://pilot.graphrag.yourdomain.com/oauth2callback"
cookie_secret = "CHANGE_ME_TO_A_64_CHAR_HEX_RANDOM_SECRET"
client_id = "your-google-client-id.apps.googleusercontent.com"
client_secret = "your-google-client-secret"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

### 4.2 Microsoft Entra ID (Azure AD)
- **Discovery URL:** `https://login.microsoftonline.com/{tenant-id}/v2.0/.well-known/openid-configuration`
- **Supported Scopes:** `openid`, `email`, `profile`
- **Subject Identifier:** Microsoft Object ID (`oid` / `sub`)

### 4.3 Okta / Auth0
- **Discovery URL:** `https://{your-domain}.okta.com/.well-known/openid-configuration`
- **Supported Scopes:** `openid`, `email`, `profile`

---

## 5. Session Lifecycle & Inactivity Management

Streamlit operates over persistent WebSocket connections. To maintain high security in a shared pilot environment:

```
[ Session State Initialized ] ---> [ Login Time Recorded ] ---> [ User Action in Tab ]
                                                                       |
+----------------------------------------------------------------------+
| Inactivity Check: (current_time - last_action_time) > SESSION_TIMEOUT_MINUTES?
+----------------------------------------------------------------------+
       |                                                  |
     [ YES ]                                            [ NO ]
       |                                                  |
  [ Log AUTH_SESSION_EXPIRED ]                       [ Update Session Activity ]
  [ Flush Session State ]                            [ Permit Operation Execution ]
  [ Invoke st.logout() ]
  [ Render Timeout Notice ]
```

1. **Inactivity Timeout:** Configured via `SESSION_TIMEOUT_MINUTES` (default: 120 minutes, minimum: 5 minutes, maximum: 1440 minutes).
2. **Explicit Sign-Out:** Users can click **"Sign Out"** in the sidebar at any time. This invokes `AuthManager.logout()`, flushes `st.session_state`, logs `AUTH_LOGOUT`, terminates the browser cookie, and re-renders the login screen.
3. **Session State Isolation:** In Streamlit, each browser tab/connection possesses an independent `session_state` thread. Query results, active inputs, and temporary state are never leaked across concurrent sessions.

---

## 6. Secrets Management & Security Boundaries

1. **Zero Secret Persistence in Source:** `.streamlit/secrets.toml` is explicitly excluded in `.gitignore` and `.dockerignore`.
2. **Safe Redaction:** `Settings.to_safe_dict()` automatically redacts `cookie_secret`, `client_secret`, and authentication tokens.
3. **Audit Log Scrubbing:** `UserAuditLogger` automatically sanitizes payload dictionaries, replacing keys matching `password`, `token`, `secret`, `api_key`, `credential`, or `authorization` with `[REDACTED]`.

---

## 7. Operational Runbook Reference
- For initial deployment configuration: see [Phase 10 Deployment Guide](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/PHASE10_DEPLOYMENT_GUIDE.md).
- For administrator bootstrap procedures: see [Phase 10 Admin Guide](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/PHASE10_ADMIN_GUIDE.md).
- For permission details: see [Phase 10 Authorization Matrix](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/PHASE10_AUTHORIZATION_MATRIX.md).
