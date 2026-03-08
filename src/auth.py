import streamlit as st
from src.logger import get_logger


def check_auth():
    """Returns (is_authenticated, user_info_dict_or_None)."""
    log = get_logger(context="auth")

    if "google_auth" not in st.secrets:
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
        from streamlit_google_auth import Authenticate

        authenticator = Authenticate(
            secret_credentials_path="client_secret.json",
            cookie_name="emotion_app_auth",
            cookie_key=st.secrets["google_auth"]["cookie_secret"],
            redirect_uri=st.secrets["google_auth"]["redirect_uri"],
        )
        authenticator.check_authentification()

        if st.session_state.get("connected"):
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
        else:
            authenticator.login()
            return False, None
    except Exception as e:
        log.error(f"Auth error: {e}")
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
