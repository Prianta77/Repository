import streamlit as st
import pandas as pd
import pickle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Buzzer Twitter Dashboard App", layout="centered")
st-sidebar.header("Dashboard")


st.title("Selamat Datang di Aplikasi Streamlit Sederhana")
st.write("Aplikasi ini dibuat untuk memenuhi tugas projek data mining")

#Load Dataset
df = pd.read_csv("Model/twitter_training.csv")

#Tampilkan Dataframe
