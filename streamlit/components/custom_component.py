import streamlit as st

def custom_button(label, key=None):
    """
    A simple custom button component
    """
    return st.button(label, key=key)
