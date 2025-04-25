# import streamlit as st
# import streamlit_authenticator as stauth
# import yaml
# from yaml.loader import SafeLoader

# with open('test.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days']
# )

# # 登录
# try:
#     authenticator.login()
# except Exception as e:
#     st.error(e)


# if st.session_state.get('authentication_status'):
#     authenticator.logout()
#     st.write(f'Welcome *{st.session_state.get("name")}*')
#     st.title('Some content')
# elif st.session_state.get('authentication_status') is False:
#     st.error('Username/password is incorrect')
# elif st.session_state.get('authentication_status') is None:
#     st.warning('Please enter your username and password')

# if st.session_state.get('authentication_status'):
#     try:
#         if authenticator.reset_password(st.session_state.get('username')):
#             st.success('Password modified successfully')
#     except Exception as e:
#         st.error(e)


# try:
#     email_of_registered_user,\
#     username_of_registered_user,\
#     name_of_registered_user = authenticator.register_user(pre_authorized=config['pre-authorized']['emails'])
#     if email_of_registered_user:
#         st.success('User registered successfuly')
# except Exception as e:
#     st.error(e)


# try:
#     username_of_forgotten_password,\
#     email_of_forgotten_password,\
#     new_random_password = authenticator.forgot_password()
#     if username_of_forgotten_password:
#         st.success('New password to be sent securely')
#     elif username_of_forgotten_password == False:
#         st.error('Username not found')
# except Exception as e:
#     st.error(e)

# try:
#     username_of_forgotten_username,\
#     email_of_registered_username = authenticator.forget_username()
#     if username_of_forgotten_username:
#         st.success('Username to be sent securey')
#     elif username_of_forgotten_username == False:
#         st.error('Email not found')
# except Exception as e:
#     st.error(e)


# if st.session_state.get('authentication_status'):
#     try:
#         if authenticator.update_user_details(st.session_state.get('username')):
#             st.success('Entries updated successfully')
#     except Exception as e:
#         st.error(e)


# with open('test.yaml','w') as file:
#     yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
# import streamlit as st
# from streamlit_authenticator import StAuthenticator,UsernamePasswordHasher
# import streamlit_app

# # 初始化一个Streamlit应用
# st.set_page_config(
#     page_title= "Streamlit App with Authentication",
#     page_icon = "::favicon::",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # 创建一个哈希器对象，用于存储和验证用户名和密码
# hasher = UsernamePasswordHasher()

# # 假设这是用户名和密码，实际使用中应该通过更安全的方式存储和验证
# USERNAME = "admin"
# PASSWORD = hasher.hash_password("my_secure_password")

# # 创建认证器对象
# authenticator = StAuthenticator(hasher)

# # 检查用户是否已登录
# if not authenticator.is_user_authenticated():
#     # 如果用户未登录，则显示登录表单
#     authenticator.login(USERNAME,PASSWORD)

# # 如果用户已登录，则显示应用程序内容
# else:
#     # 编写应用程序逻辑
#     streamlit_app.main()

#     if st.button("Logout"):
#         authenticator.logout()
#         st.stop()

# import streamlit_authenticator as stauth
# import streamlit as st
# import streamlit_app

# # 如下代码数据，可以来自数据库
# names = ["陈晓君","管理员"]
# usernames = ["chenxj","admin"]
# passwords = ['xj1234','ad1234']


# hashed_passwords = stauth.Hasher.hash_passwords(passwords)

# authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
#                                     'some_cookie_name','some_signature_key',cookie_expiry_days = 30)

# name, authentication_status, username = authenticator.login('Login','main')

# if authentication_status:
#     with st.container():
#         cols1,cols2 = st.columns(2)
#         cols1.write('欢迎 *%s*' % (name))
#         with cols2.container():
#             authenticator.logout('Logout','main')
    
#     streamlit_app.main()

# elif authentication_status == False:
#     st.error('Username/password is incorrect')
# elif authentication_status == None:
#     st.warning('Please enter yout username and password')