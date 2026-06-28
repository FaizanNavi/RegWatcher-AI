import streamlit as st
import requests

# Set page config
st.set_page_config(page_title="RegWatcher AI", page_icon="🏛️")

st.title("🏛️ RegWatcher AI")
st.markdown("Ask questions about the US Federal Register data!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about regulations..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send request to FastAPI backend
    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://127.0.0.1:8001/chat",
                json={"message": prompt},
                timeout=60
            )
            
        if response.status_code == 200:
            data = response.json()
            bot_reply = data.get("response", "No response generated.")
            
            # Display Assistant response
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
                
                # Optional: Show debug info inside an expander
                with st.expander("Agent Debug Info"):
                    st.write(f"**Route:** {data.get('route')}")
                    st.write(f"**Critic Score:** {data.get('critic_score')} / 10")
                    if data.get('validation_issues'):
                        st.write(f"**Issues Fixed:** {data.get('validation_issues')}")
                        
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            
        else:
            st.error(f"Backend returned an error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend! Is `python -m app.main` running on port 8001?")
    except Exception as e:
        st.error(f"An error occurred: {e}")
