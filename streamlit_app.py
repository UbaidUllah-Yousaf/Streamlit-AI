import time
import streamlit as st
import requests
import os

# Configuration - use environment variables
BASE_URL = os.getenv('BASE_URL', 'https://staging-api.alta-group.eu/')
DJANGO_API_URL = os.getenv('DJANGO_API_URL', f'{BASE_URL}api/ai-assistant')
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.message_count = 0

# Streamlit UI
st.title("EPC AI Assistant")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    organizations = [
        {"name": "Not Selected", "key": "rag-docs-index"},
        {"name": "Alirec", "key": "alirec-index"},
        {"name": "Stad Koksijde", "key": "stad-kiksijde"}
    ]

    # Create mapping of organization names to IDs
    org_options = {org['name']: org['key'] for org in organizations}

    # Organization selection dropdown
    selected_org_name = st.selectbox(
        "Select Organization",
        options=list(org_options.keys()),
        index=0,
        help="Select organization"
    )
    selected_org_id = org_options[selected_org_name]
    email = st.text_input("Email", placeholder="Your email address")
    password = st.text_input("Password", placeholder="Your password", type="password")


def render_citation(citation, message_id, citation_idx):
    """Render a single citation with toggle functionality"""
    toggle_key = f"show_citation_{message_id}_{citation_idx}"

    # Initialize toggle state if not exists
    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = False
    # Create a container for the citation
    with st.container():
    #     # Citation header with toggle button
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            title = citation.get('title', 'Untitled Document')
            if st.button(f"Reference {citation_idx + 1}: {title}",
                         key=f"citation-header-{message_id}-{citation_idx}",
                         help="Click to show/hide reference content"):
                st.session_state[toggle_key] = not st.session_state[toggle_key]
        with col2:
            if citation.get('url'):
                st.markdown(f"[🔗]({citation['url']})", unsafe_allow_html=True)

        # Citation content (shown if expanded)
        if st.session_state[toggle_key]:
            with st.expander(f"Reference {citation_idx + 1}: {title}", expanded=True):
                st.markdown(citation.get('content', 'No content available'))


def render_references(citations, message_id):
    """Render all citations for a message"""
    if not citations:
        return

    st.markdown("**References**")
    for idx, citation in enumerate(citations):
        render_citation(citation, message_id, idx)


# Display chat history with toggleable citations
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("citations"):
            # Use the message's unique ID if it exists, otherwise use index
            message_id = message.get("id", i)
            render_references(message["citations"], message_id)

# Chat input
if prompt := st.chat_input("Ask about EPC processes..."):
    # Add message with unique ID
    st.session_state.message_count += 1
    new_message = {
        "id": st.session_state.message_count,
        "role": "user",
        "content": prompt,
        "organization_id": selected_org_id
    }
    st.session_state.messages.append(new_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Authenticate and get token
            auth_response = requests.post(
                f'{BASE_URL}api/token/',
                data={"email": email, "password": password}
            )
            auth_response.raise_for_status()
            token = auth_response.json()['access']

            # Call Django API
            with st.spinner("Generating response..."):
                response = requests.post(
                    DJANGO_API_URL,
                    json={
                        "organization": selected_org_id,
                        "message": prompt
                    },
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                # Stream response
                for chunk in data["content"].split():
                    full_response += chunk + " "
                    response_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)

                response_placeholder.markdown(full_response)

                # Add assistant response with unique ID
                st.session_state.message_count += 1
                assistant_message = {
                    "id": st.session_state.message_count,
                    "role": "assistant",
                    "content": full_response,
                    "citations": data.get("citations", []),
                    "organization_id": selected_org_id
                }
                st.session_state.messages.append(assistant_message)

                # Render references for the new message
                if data.get("citations"):
                    render_references(data["citations"], st.session_state.message_count)

        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
        except KeyError:
            st.error("Authentication failed. Please check your credentials.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
