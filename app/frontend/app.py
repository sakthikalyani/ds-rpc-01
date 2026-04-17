import streamlit as st
import requests
import time
import os
import base64
# ---- Backend URLs ----
BACKEND_URL = "http://localhost:8000/chat"
LOGIN_URL = "http://localhost:8000/login"

# ---- Session Setup ----
st.set_page_config(page_title="accessbot", layout="centered")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "img", "brainimg.png")
# image_path = "app/frontend/img/brainimg.png"
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_image}" width="40" style="margin-right: 10px;" />
        <h1 style="margin: 0;">FinsightAI</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Role-based AI assistant for smarter enterprise answers")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.role = ""
    st.session_state.messages = [] 

# ---------------------- LOGIN PAGE -------------------------
if not st.session_state.authenticated:
    st.subheader(" Login to FinsightAI")

    with st.form("login_form"):
        username = st.text_input("👤 Username")
        password = st.text_input("🔒 Password", type="password")
        login_btn = st.form_submit_button("𝗟𝗢𝗚𝗜𝗡")

        if login_btn:
            try:
                response = requests.get(LOGIN_URL, auth=(username, password))
                if response.status_code == 200:
                    server_role = response.json().get("role")
                    if server_role:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.password = password
                        st.session_state.role = server_role
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Role mismatch!")
                else:
                    st.error("❌ Invalid credentials.")
            except Exception as e:
                st.error(f"🚨 Error: {e}")

# ---------------------- CHATBOT PAGE -------------------------
else:
    col1, col2 = st.columns([7, 1])
    with col1:
        st.markdown(f"🙋 Welcome, {st.session_state.username}")
        st.markdown(f"🧑‍💼 Role: {st.session_state.role}")
    with col2:
        if st.button("Logout"):
            for k in ["authenticated", "username", "password", "role", "messages"]:
                st.session_state[k] = "" if k != "messages" else []
            st.rerun()

    

    # Chat History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    #User Input
    user_input = st.chat_input("💬 Type your question here...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("🧠 FinsightAI is thinking...")

        # Send to backend
        try:
            res = requests.post(
                BACKEND_URL,
                json={"message": user_input},
                auth=(st.session_state.username, st.session_state.password)
            )
            if res.status_code == 200:
                reply = res.json()["answer"]

                full_reply = ""
                for ch in reply:
                    full_reply += ch
                    typing_placeholder.markdown(full_reply)
                    time.sleep(0.015)

                # Save bot response
                st.session_state.messages.append({"role": "assistant", "content": full_reply})
            else:
                typing_placeholder.markdown("⚠ Error from backend.")
        except Exception as e:
            typing_placeholder.markdown(f"❌ {e}")
