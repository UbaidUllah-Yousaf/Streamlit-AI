import json
import time
import streamlit as st
import requests
import os
import uuid
from typing import Dict, List, Optional

# Configuration
DJANGO_API_URL = os.getenv('DJANGO_API_URL', 'http://localhost:8000/api/ai-assistant')
AUTH_URL = os.getenv('AUTH_URL', 'http://localhost:8000/api/token/')


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "message_count" not in st.session_state:
        st.session_state.message_count = 0

    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None

    if "auth_failed" not in st.session_state:
        st.session_state.auth_failed = False

    if "org_mapping" not in st.session_state:
        st.session_state.org_mapping = {
            "test": "9",
        }

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if "new_conversation" not in st.session_state:
        st.session_state.new_conversation = True


initialize_session_state()

# Streamlit UI Configuration
st.set_page_config(
    page_title="EPC AI Assistant",
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
                st.markdown(citation.get('content', 'No content available'), unsafe_allow_html=True)
                if citation.get('score'):
                    st.caption(f"Relevance score: {citation['score']:.2f}")


import re
import json
from typing import Dict, Tuple


def extract_json_from_markdown(markdown_text: str) -> Dict:
    """Extract JSON from markdown code block"""
    # Pattern to match ```json {...} ```
    json_pattern = r'```json\n(.*?)\n```'
    match = re.search(json_pattern, markdown_text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {"answer": markdown_text, "info": "0"}
    return {"answer": markdown_text, "info": "0"}


def format_response(response_data: (Dict, str)) -> Tuple[str, str]:
    """Format the API response and return (answer, info) tuple"""
    try:
        # Handle markdown-wrapped JSON
        if isinstance(response_data, str) and '```json' in response_data:
            response_data = extract_json_from_markdown(response_data)

        # Handle regular JSON string
        elif isinstance(response_data, str):
            try:
                response_data = json.loads(response_data)
            except json.JSONDecodeError:
                return response_data, "0"

        # Extract answer and info
        answer = response_data.get("answer", "No answer provided")
        info = response_data.get("info", "0")

        return answer, info

    except Exception as e:
        st.error(f"Error formatting response: {str(e)}")
        return "Could not process the response", "0"


def render_message(message: Dict):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            if isinstance(message["content"], (dict, str)):
                # Get both answer and info status
                answer, info = format_response(message["content"])

                # Display the answer with markdown formatting
                st.markdown(answer)

                # Show warning if info is "0"
                if info == "0" and message["content"]:
                    st.warning(
                        "Unable to find relevant information in the provided index. "
                        "This is a general answer that may not be specific to your query.",
                        icon="⚠️"
                    )
            else:
                st.markdown(str(message["content"]))

        else:
            st.markdown(message["content"])

        if message.get("retrieved_documents") and message["content"]:
            st.markdown("---")
            st.markdown("**References**")
            for idx, citation in enumerate(message["retrieved_documents"]):
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
    render_message(user_message)

    # Prepare assistant response
    full_response = ""
    try:
        # Prepare request data
        request_data = {
            "organization": selected_org_id,
            "message": prompt,
            "new_conversation": st.session_state.new_conversation
        }

        # Only include conversation_id if we have one
        if st.session_state.conversation_id:
            request_data["conversation_id"] = ""

        # Call Django RAG API
        with st.spinner("Researching your question..."):
            start_time = time.time()

            response = requests.post(
                DJANGO_API_URL,
                json=request_data,
                headers={
                    "Authorization": f"Bearer {st.session_state.auth_token}",
                    "Content-Type": "application/json"
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # After first message, it's no longer a new conversation
            st.session_state.new_conversation = False

            # Update conversation_id from response if provided
            if data.get("conversation_id"):
                st.session_state.conversation_id = data["conversation_id"]

            # Handle both simple answer format and full RAG response
            if "answer" in data:
                full_response = data
            else:
                full_response = data.get("content", {})

            # Add assistant response to history
            st.session_state.message_count += 1
            assistant_message = {
                "id": st.session_state.message_count,
                "role": "assistant",
                "content": full_response,
                "organization_id": selected_org_id,
                "response_time": time.time() - start_time,
                "retrieved_documents": data.get("retrieved_documents", [])
            }
            st.session_state.messages.append(assistant_message)

            # Display the message
            render_message(assistant_message)

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
    st.session_state.conversation_id = ""
    st.session_state.new_conversation = True
    st.rerun()
