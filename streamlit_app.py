import streamlit as st
import time
from datetime import datetime
import base64

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "login_time" not in st.session_state:
    st.session_state.login_time = None
if "show_logout_confirm" not in st.session_state:
    st.session_state.show_logout_confirm = False

# Default user
USERNAME = "admin"
PASSWORD = "admin"

# Function to create card-like container
def styled_container(content_function):
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            with st.container(border=True):
                content_function()

# Login page with improved UI
def login():
    # Load logo image (replace with your own logo path or use a default icon)
    logo_html = """
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="#0f52ba" stroke-width="1.5">
                <!-- Circuit board background pattern -->
                <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                <!-- Connection nodes -->
                <circle cx="8" cy="8" r="2" fill="#0f52ba"/>
                <circle cx="16" cy="8" r="2" fill="#0f52ba"/>
                <circle cx="12" cy="16" r="2" fill="#0f52ba"/>
                <circle cx="7" cy="15" r="1" fill="#0f52ba"/>
                <circle cx="17" cy="15" r="1" fill="#0f52ba"/>
                <!-- Connection lines -->
                <path d="M8 10v1.5a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V10" stroke-linecap="round"/>
                <line x1="8" y1="8" x2="16" y2="8" stroke-linecap="round"/>
                <line x1="7" y1="15" x2="10.5" y2="15" stroke-linecap="round"/>
                <line x1="13.5" y1="15" x2="17" y2="15" stroke-linecap="round"/>
                <!-- Data flow extraction arrow -->
                <path d="M12 10v4" stroke-linecap="round" stroke-width="2"/>
                <path d="M9 14l3 3 3-3" fill="#0f52ba" stroke-linecap="round"/>
            </svg>
        </div>
        <h1 style="text-align: center; color: #0f52ba; margin-bottom: 30px;">Interaction Logic Extraction System</h1>
    """
    
    st.markdown(logo_html, unsafe_allow_html=True)
    
    def login_form():
        st.markdown("<h2 style='text-align: center; color: #333;'>Sign In</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; margin-bottom: 20px;'>Please enter your credentials</p>", unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Enter username", key="username_input")
        password = st.text_input("Password", type="password", placeholder="Enter password", key="password_input")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            login_button = st.button("Login", use_container_width=True, type="primary")
        
        # Remember me checkbox and forgot password link
        cols = st.columns([1, 1])
        with cols[0]:
            st.checkbox("Remember me", key="remember")
        with cols[1]:
            st.markdown("<div style='text-align: right;'><a href='#' style='color: #0f52ba; text-decoration: none;'>Forgot password?</a></div>", unsafe_allow_html=True)
        
        if login_button:
            with st.spinner("Authenticating..."):
                if username == USERNAME and password == PASSWORD:
                    st.session_state.logged_in = True
                    st.session_state.login_time = datetime.now()
                    st.success("Login successful!")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("Incorrect username or password")
                    st.markdown("<p style='color: #666; font-size: 0.8em;'>Hint: Try 'admin' for both fields</p>", unsafe_allow_html=True)
    
    styled_container(login_form)
    
    # Footer
    st.markdown("""
        <div style="position: fixed; bottom: 0; left: 0; right: 0; text-align: center; padding: 10px; background-color: rgba(255, 255, 255, 0.7);">
            <p style="color: #666; font-size: 0.8em;">© 2025 Interaction Logic Extraction System | Privacy Policy | Terms of Service</p>
        </div>
    """, unsafe_allow_html=True)

# Logout interface
def logout():
    st.write("")  # Add some space
    
    # User profile section
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # User avatar
        st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; width: 80px; height: 80px; background-color: #0f52ba; border-radius: 50%; color: white; font-size: 2em;">
                A
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### Welcome, {USERNAME}")
        if st.session_state.login_time:
            login_time_str = st.session_state.login_time.strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"<p style='color: #666;'>Logged in since: {login_time_str}</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Account settings section
    st.markdown("### Account Settings")
    
    # Settings options
    options_col1, options_col2 = st.columns(2)
    with options_col1:
        st.button("Profile Settings", use_container_width=True)
        st.button("Notification Preferences", use_container_width=True)
    
    with options_col2:
        st.button("Security Settings", use_container_width=True)
        
        # Logout button with confirmation
        if not st.session_state.show_logout_confirm:
            if st.button("Logout", use_container_width=True, type="primary"):
                st.session_state.show_logout_confirm = True
                st.rerun()
        else:
            st.warning("Are you sure you want to logout?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Logout", use_container_width=True, type="primary"):
                    st.session_state.logged_in = False
                    st.session_state.show_logout_confirm = False
                    st.session_state.login_time = None
                    st.rerun()
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_logout_confirm = False
                    st.rerun()

# Page configuration
page1 = st.Page("app.py", title="Extraction")
page2 = st.Page("history.py", title="Result")
page3 = st.Page("Knowledge.py", title="KnowledgeBase")

# Navigation based on login state
if not st.session_state.logged_in:
    login_page = st.Page(login, title="Login")
    pg = st.navigation([login_page])
else:
    logout_page = st.Page(logout, title="Account")
    pg = st.navigation(
        {
            "Main functions": [page1, page2, page3],
            "User Settings": [logout_page]
        }
    )
pg.run()

# import streamlit as st
# import time 

# # 初始化会话状态
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# # 默认用户
# USERNAME = "admin"
# PASSWORD = "admin"

# # 登录页面
# def login():
#     st.header("Login")
#     st.divider()

#     username = st.text_input("username")
#     password = st.text_input("password", type="password")

#     if st.button("Login"):
#         if username == USERNAME and password == PASSWORD:
#             st.session_state.logged_in = True
#             st.success("login successful")
#             time.sleep(0.5)
#             st.rerun()
#         else:
#             st.error("Incorrect username or password")

# # 登出按钮
# def logout():
#     if st.button("Logout"):
#         st.session_state.logged_in = False
#         st.rerun()

# # 页面配置
# page1 = st.Page("app.py", title="Extraction")
# page2 = st.Page("history.py", title="Result")
# page3 = st.Page("Knowledge.py", title="KnowledgeBase")

# # 默认只有login页面
# if not st.session_state.logged_in:
#     login_page = st.Page(login, title="Login")
#     pg = st.navigation([login_page])
# else:
#     logout_page = st.Page(logout, title="Logout")
#     pg = st.navigation(
#         {
#             "Account Management": [logout_page],
#             "Main functions": [page1, page2, page3]
#         }
#     )
# pg.run()
   



# import streamlit as st

# pg = st.navigation([st.Page("app.py", title="Extraction"),st.Page("history.py", title="Result"),st.Page("Knowledge.py", title="KnowledgeBase")])
# pg.run()
