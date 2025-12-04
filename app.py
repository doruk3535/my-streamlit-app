import streamlit as st

st.set_page_config(page_title="My Streamlit App", page_icon="🔥", layout="wide")

st.title("🔥 My First Streamlit App")
st.write("GitHub içinden oluşturulan Streamlit projesi başarıyla çalışıyor!")

name = st.text_input("Adın ne?")
if name:
    st.success(f"Merhaba {name}! Uygulama çalışıyor.")
