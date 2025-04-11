import streamlit as st
import pandas as pd
import numpy as np
import time

df = pd.DataFrame(np.random.randn(15,3), columns=(["A","B","C"]))
my_data_element = st.line_chart(df)


for tick in range(10):
    time.sleep(.5)
    add_df = pd.DataFrame(np.random.randn(1,3), columns=(["A","B","C"]))
    my_data_element.add_rows(add_df)


st.button("Regenerate")


animal_shelter = ['cat', 'dog', 'rabbit', 'bird']

animal = st.text_input('Type an animal')

if st.button('Check avaiability'):
    have_it = animal.lower() in animal_shelter
    'We have that animal!' if  have_it else 'We don\'t have that animal.'


if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button('Click me', on_click=click_button)

if st.session_state.clicked:
    st.write('Button clicked!')
    st.slider('Select a value')