import time
import streamlit as st
import requests
import os
from typing import Dict, List, Optional


# Configuration - use environment variables
BASE_URL = os.getenv('BASE_URL', 'https://staging-api.alta-group.eu')
DJANGO_API_URL = os.getenv('DJANGO_API_URL', f'{BASE_URL}/api/v1/ai-assistant')
AUTH_URL = os.getenv('AUTH_URL', f'{BASE_URL}/api/token/')

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.message_count = 0
        st.session_state.auth_token = None
        st.session_state.auth_failed = False
        st.session_state.org_mapping = {
            "Generic Index": "rag-docs-index",
            "Alirec": "alirec-index",
            "Stad Koksijde": "stad-kiksijde"
        }


initialize_session_state()

# Streamlit UI Configuration
st.set_page_config(
    page_title="EPC AI Assistant",
    # page_icon="",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stChatMessage {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e6f7ff;
    }
    .citation-header {
        font-size: 0.9em;
        color: #555;
    }
    .stButton>button {
        width: 100%;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)


# Sidebar Configuration
def render_sidebar() -> Optional[str]:
    """Render the sidebar and return selected organization ID"""
    with st.sidebar:
        st.title("Settings")

        # Organization selection
        selected_org_name = st.selectbox(
            "Select Organization",
            options=list(st.session_state.org_mapping.keys()),
            index=0,
            help="Select Organization"
        )

        # Authentication
        with st.form("auth_form"):
            st.header("Authentication")
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password")

            if st.form_submit_button("Login"):
                try:
                    response = requests.post(
                        AUTH_URL,
                        data={"email": email, "password": password},
                        timeout=10
                    )
                    response.raise_for_status()
                    st.session_state.auth_token = response.json()['access']
                    st.session_state.auth_failed = False
                    st.success("Authentication successful")
                except Exception as e:
                    st.session_state.auth_failed = True
                    st.error(f"Authentication failed: {str(e)}")

        if st.session_state.auth_token:
            st.success("✅ Logged in")
        elif st.session_state.auth_failed:
            st.error("❌ Login required")

        return st.session_state.org_mapping.get(selected_org_name)


selected_org_id = render_sidebar()

# Main UI
st.title("EPC AI Assistant")
st.caption("Ask questions about EPC processes and get AI-powered answers with references")


# Chat History Rendering
def render_citation(citation: Dict, message_id: int, citation_idx: int):
    """Render a single citation with toggle functionality"""
    toggle_key = f"show_citation_{message_id}_{citation_idx}"

    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = False

    with st.container():
        col1, col2 = st.columns([0.9, 0.1])

        with col1:
            title = citation.get('title', 'Untitled Document')
            if st.button(
                    f"📄 {title}",
                    key=f"citation-btn-{message_id}-{citation_idx}",
                    help="Click to view reference content"
            ):
                st.session_state[toggle_key] = not st.session_state[toggle_key]

        with col2:
            if citation.get('url'):
                st.markdown(f"[🔗]({citation['url']})", unsafe_allow_html=True)

        if st.session_state[toggle_key]:
            with st.expander(f"Reference Content", expanded=True):
                st.markdown(citation.get('content', 'No content available'))
                if citation.get('score'):
                    st.caption(f"Relevance score: {citation['score']:.2f}")


def render_message(message: Dict):
    """Render a single chat message with citations"""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("citations"):
            st.markdown("---")
            st.markdown("**References**")
            for idx, citation in enumerate(message["citations"]):
                render_citation(citation, message["id"], idx)


# Display chat history
for message in st.session_state.messages:
    render_message(message)

# Chat Input and Processing
if prompt := st.chat_input("Ask about EPC processes..."):
    if not st.session_state.auth_token:
        st.error("Please login first")
        st.stop()

    # Add user message to history
    st.session_state.message_count += 1
    user_message = {
        "id": st.session_state.message_count,
        "role": "user",
        "content": prompt,
        "organization_id": selected_org_id
    }
    st.session_state.messages.append(user_message)

    # Display user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Prepare and display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Call Django RAG API
            with st.spinner("Researching your question..."):
                start_time = time.time()

                response = requests.post(
                    DJANGO_API_URL,
                    json={
                        "organization": selected_org_id,
                        "message": prompt
                    },
                    headers={
                        "Authorization": f"Bearer {st.session_state.auth_token}",
                        "Content-Type": "application/json"
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                # Stream the response
                for chunk in data["content"].split():
                    full_response += chunk + " "
                    response_placeholder.markdown(full_response + "▌")
                    time.sleep(0.03)

                response_placeholder.markdown(full_response)

                # Add assistant response to history
                st.session_state.message_count += 1
                assistant_message = {
                    "id": st.session_state.message_count,
                    "role": "assistant",
                    "content": full_response,
                    "citations": data.get("citations", []),
                    "organization_id": selected_org_id,
                    "response_time": time.time() - start_time
                }
                st.session_state.messages.append(assistant_message)

                # Show references if available
                if data.get("citations"):
                    st.markdown("---")
                    st.markdown("**References**")
                    for idx, citation in enumerate(data["citations"]):
                        render_citation(citation, st.session_state.message_count, idx)

        except requests.exceptions.HTTPError as e:
            st.error(f"API Error: {e.response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Network Error: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# Add clear conversation button
if st.session_state.messages and st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.message_count = 0
    st.rerun()

