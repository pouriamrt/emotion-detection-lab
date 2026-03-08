import time

import streamlit as st
import google_auth_oauthlib.flow
from googleapiclient.discovery import build

from src.logger import get_logger

_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
]

# Module-level dict survives across Streamlit reruns (same Python process),
# unlike st.session_state which is lost on full-page redirects.
# Keyed by OAuth state parameter → PKCE code_verifier.
_pkce_store: dict[str, str] = {}


def _build_flow(*, with_pkce: bool = True):
    """Create a Google OAuth flow from client_secret.json."""
    return google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        "client_secret.json",
        scopes=_SCOPES,
        redirect_uri=st.secrets["google_auth"]["redirect_uri"],
        autogenerate_code_verifier=with_pkce,
    )


def _exchange_code(log):
    """Handle the OAuth callback if auth code is present in query params.

    MUST run BEFORE any Streamlit component (CookieManager, etc.) is created,
    because components can trigger RerunException and interrupt the exchange.

    Returns user_info dict on success, None if no callback code present.
    """
    auth_code = st.query_params.get("code")
    oauth_state = st.query_params.get("state")
    if not auth_code:
        return None

    log.info(f"OAuth callback: code len={len(auth_code)}, state={oauth_state}")
    st.query_params.clear()

    code_verifier = _pkce_store.pop(oauth_state, None)
    log.info(f"PKCE verifier found: {code_verifier is not None}")

    flow = _build_flow(with_pkce=False)
    flow.code_verifier = code_verifier
    flow.fetch_token(code=auth_code)

    credentials = flow.credentials
    user_info_service = build(
        serviceName="oauth2", version="v2", credentials=credentials
    )
    user_info = user_info_service.userinfo().get().execute()
    log.info(f"Token exchange success: {user_info.get('email')}")
    return user_info


def check_auth():
    """Returns (is_authenticated, user_info_dict_or_None)."""
    log = get_logger(context="auth")

    try:
        has_google_auth = "google_auth" in st.secrets
    except Exception:
        has_google_auth = False

    if not has_google_auth:
        log.warning("Google OAuth not configured - running in dev mode")
        if "user" not in st.session_state:
            st.session_state["user"] = {
                "name": "Dev User",
                "email": "dev@localhost",
                "picture": "",
                "authenticated": True,
            }
            log.info("Dev user auto-logged in", user="Dev User")
        return True, st.session_state["user"]

    try:
        st.session_state.setdefault("connected", False)

        # 1. Already authenticated this session
        if st.session_state["connected"]:
            user_info = st.session_state.get("user_info", {})
            if "user" not in st.session_state or st.session_state["user"].get("email") != user_info.get("email"):
                st.session_state["user"] = {
                    "name": user_info.get("name", "Unknown"),
                    "email": user_info.get("email", "Unknown"),
                    "picture": user_info.get("picture", ""),
                    "authenticated": True,
                }
                log.info(f"User logged in: {user_info.get('email')}", user=user_info.get("name", "Unknown"))
            return True, st.session_state["user"]

        # 2. Handle OAuth callback BEFORE creating any components
        #    (CookieManager can trigger RerunException and interrupt token exchange)
        user_info = _exchange_code(log)
        if user_info:
            st.session_state["connected"] = True
            st.session_state["oauth_id"] = user_info.get("id")
            st.session_state["user_info"] = user_info
            st.session_state["user"] = {
                "name": user_info.get("name", "Unknown"),
                "email": user_info.get("email", "Unknown"),
                "picture": user_info.get("picture", ""),
                "authenticated": True,
            }
            log.info(f"User authenticated: {user_info.get('email')}", user=user_info.get("name", "Unknown"))
            # Set the auth cookie (CookieManager can safely rerun now)
            from streamlit_google_auth.cookie import CookieHandler
            cookie_handler = CookieHandler(
                cookie_name="emotion_app_auth",
                cookie_key=st.secrets["google_auth"]["cookie_secret"],
                cookie_expiry_days=30.0,
            )
            cookie_handler.set_cookie(
                user_info.get("name"),
                user_info.get("email"),
                user_info.get("picture"),
                user_info.get("id"),
            )
            st.rerun()

        # 3. Check for existing auth cookie
        from streamlit_google_auth.cookie import CookieHandler
        cookie_handler = CookieHandler(
            cookie_name="emotion_app_auth",
            cookie_key=st.secrets["google_auth"]["cookie_secret"],
            cookie_expiry_days=30.0,
        )
        time.sleep(0.3)
        token = cookie_handler.get_cookie()
        if token:
            user_info = {
                "name": token["name"],
                "email": token["email"],
                "picture": token["picture"],
                "id": token["oauth_id"],
            }
            st.query_params.clear()
            st.session_state["connected"] = True
            st.session_state["user_info"] = user_info
            st.session_state["user"] = {
                "name": user_info["name"],
                "email": user_info["email"],
                "picture": user_info["picture"],
                "authenticated": True,
            }
            log.info(f"User restored from cookie: {user_info['email']}", user=user_info["name"])
            return True, st.session_state["user"]

        # 4. Show login button — generate authorization URL with PKCE
        flow = _build_flow(with_pkce=True)
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
        )
        # Save code_verifier in module-level dict (survives the redirect)
        _pkce_store[state] = flow.code_verifier

        html_content = f"""
<div style="display: flex; justify-content: center;">
    <a href="{authorization_url}" target="_self" style="background-color: #4285f4; color: #fff; text-decoration: none; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; padding: 8px 12px; border-radius: 4px; display: flex; align-items: center;">
        <img src="https://lh3.googleusercontent.com/COxitqgJr1sJnIDe8-jiKhxDx1FrYbtRHKJ9z_hELisAlapwE9LUPh6fcXIfb5vwpbMl4xl9H9TRFPc5NOO8Sb3VSgIBrfRYvW6cUA" alt="Google logo" style="margin-right: 8px; width: 26px; height: 26px; background-color: white; border: 2px solid white; border-radius: 4px;">
        Sign in with Google
    </a>
</div>
"""
        st.markdown(html_content, unsafe_allow_html=True)
        return False, None

    except Exception as e:
        import traceback
        log.error(f"Auth error: {e}\n{traceback.format_exc()}")
        st.error(f"Authentication error: {e}. Running in dev mode.")
        st.session_state["user"] = {
            "name": "Dev User",
            "email": "dev@localhost",
            "picture": "",
            "authenticated": True,
        }
        return True, st.session_state["user"]


def require_auth():
    """Call at top of every page. Returns user dict or stops execution."""
    is_auth, user = check_auth()
    if not is_auth:
        st.stop()
    return user


def logout():
    log = get_logger(context="auth")
    user = st.session_state.get("user", {})
    log.info(f"User logged out: {user.get('email')}", user=user.get("name", "system"))
    for key in ["user", "connected", "user_info"]:
        st.session_state.pop(key, None)
    st.rerun()
