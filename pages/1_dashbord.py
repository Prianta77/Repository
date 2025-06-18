import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("twitter_training.csv")
    df.columns = ["ID", "Entity", "Sentiment", "Tweet"]
    df["Tweet"] = df["Tweet"].astype(str)
    df["Tweet Length"] = df["Tweet"].apply(len)
    return df

# Load data
df = load_data()

# Title
st.title("📊 Twitter Sentiment EDA Dashboard")

# Dataset overview
st.subheader("Dataset Preview")
st.write(df.head())

# Basic stats
st.subheader("Dataset Info")
st.markdown(f"**Total Rows:** {len(df)}")
st.markdown(f"**Missing Tweets:** {df['Tweet'].isnull().sum()}")
st.markdown(f"**Unique Entities:** {df['Entity'].nunique()}")
st.markdown(f"**Unique Sentiments:** {df['Sentiment'].nunique()}")

# Sentiment distribution
st.subheader("Distribusi Sentimen")
sentiment_counts = df["Sentiment"].value_counts()
st.bar_chart(sentiment_counts)

# Tweet length distribution
st.subheader("Distribusi Panjang Tweet")
fig, ax = plt.subplots()
sns.histplot(df["Tweet Length"], bins=50, kde=True, color='skyblue', ax=ax)
ax.set_xlabel("Panjang Tweet")
ax.set_ylabel("Frekuensi")
ax.set_title("Distribusi Panjang Tweet")
st.pyplot(fig)

# Show random samples
st.subheader("Contoh Tweet Acak")
st.write(df[["Entity", "Sentiment", "Tweet"]].sample(5))
