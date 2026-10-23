"""Personal Assistant - Phase 1, Step 1: password-protected streaming chat (Groq)."""

import hmac
from collections.abc import Iterator
from datetime import datetime
from zoneinfo import ZoneInfo

import groq
import streamlit as st

MAX_TOKENS = 2048
MAX_HISTORY_MESSAGES = 30  # how many recent messages we send to the model
TIMEZONE = ZoneInfo("Africa/Cairo")

st.set_page_config(page_title="Personal Assistant", page_icon="🤖", layout="centered")

# Makes each paragraph align by its own language (Arabic -> right-to-left).
st.markdown(
    "<style>[data-testid='stChatMessage'] p {unicode-bidi: plaintext;}</style>",
    unsafe_allow_html=True,
)


def check_password() -> bool:
    """Show a login form until the correct password is entered."""
    if st.session_state.get("authenticated"):
        return True

    st.title("🔐 Personal Assistant")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        expected = st.secrets["APP_PASSWORD"].encode()
        # compare_digest avoids timing attacks; bytes handle non-ASCII passwords.
        if hmac.compare_digest(password.encode(), expected):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Wrong password.")
    return False


@st.cache_resource
def get_client() -> groq.Groq:
    """Create the API client once and reuse it across reruns."""
    return groq.Groq(api_key=st.secrets["GROQ_API_KEY"])


def build_system_prompt() -> str:
    now = datetime.now(TIMEZONE).strftime("%A, %d %B %Y, %H:%M")
    return (
        "You are a warm, capable personal assistant for a busy person in Egypt. "
        "Reply in the same language the user writes in (Arabic or English). "
        "Be concise and practical. Ask a clarifying question only when the "
        "answer would change materially. "
        f"Current date and time (Cairo): {now}."
    )


def stream_reply(client: groq.Groq, messages: list[dict]) -> Iterator[str]:
    """Yield the assistant's reply chunk by chunk.

    This is the only function that knows which LLM provider we use.
    """
    # Unlike Claude's API, Groq takes the system prompt as the first message.
    full_messages = [{"role": "system", "content": build_system_prompt()}, *messages]

    stream = client.chat.completions.create(
        model=st.secrets["GROQ_MODEL"],
        messages=full_messages,
        max_completion_tokens=MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        # Some chunks (e.g. the final one) have no choices or no text.
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def main() -> None:
    if not check_password():
        st.stop()  # nothing below runs until the user is signed in

    st.title("🤖 Personal Assistant")

    with st.sidebar:
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask me anything...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply(get_client(), history))
        except groq.RateLimitError:
            st.error("Free-tier rate limit reached. Wait a minute and try again.")
            st.session_state.messages.pop()
            return
        except groq.APIError as exc:
            st.error(f"Something went wrong talking to the AI service: {exc}")
            st.session_state.messages.pop()
            return

    st.session_state.messages.append({"role": "assistant", "content": reply})


main()