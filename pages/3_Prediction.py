import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seab as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("twitter_training.csv")
    df.columns = ["ID", "Entity", "Sentiment", "Tweet"]
    df["Tweet"] = df["Tweet"].astype(str)
    df["Tweet Length"] = df["Tweet"].apply(len)
    return df

# Train model
@st.cache_data
def train_model(df):
    df = df.dropna(subset=["Tweet"])
    X = df["Tweet"]
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    return model, vectorizer, report, accuracy

# Load data and model
df = load_data()
model, vectorizer, report, accuracy = train_model(df)

# Navigation
page = st.sidebar.radio("Navigasi", ["EDA", "Model", "Prediksi"])

if page == "EDA":
    st.title("📊 Twitter Sentiment EDA Dashboard")

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Info")
    st.markdown(f"**Total Rows:** {len(df)}")
    st.markdown(f"**Missing Tweets:** {df['Tweet'].isnull().sum()}")
    st.markdown(f"**Unique Entities:** {df['Entity'].nunique()}")
    st.markdown(f"**Unique Sentiments:** {df['Sentiment'].nunique()}")

    st.subheader("Distribusi Sentimen")
    sentiment_counts = df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("Distribusi Panjang Tweet")
    fig, ax = plt.subplots()
    sns.histplot(df["Tweet Length"], bins=50, kde=True, color='skyblue', ax=ax)
    ax.set_xlabel("Panjang Tweet")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi Panjang Tweet")
    st.pyplot(fig)

    st.subheader("Contoh Tweet Acak")
    st.write(df[["Entity", "Sentiment", "Tweet"]].sample(5))

elif page == "Model":
    st.title("🤖 Hasil Pelatihan Model Sentimen")

    st.markdown(f"### Akurasi Model: {accuracy:.2%}")

    st.subheader("Laporan Klasifikasi")
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Coba Prediksi Tweet Baru")
    user_input = st.text_area("Masukkan Tweet:", "This game is awesome!")
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"Prediksi Sentimen: {prediction}")

elif page == "Prediksi":
    st.title("🔮 Formulir Prediksi Sentimen Tweet")

    with st.form("predict_form"):
        tweet_input = st.text_area("Masukkan Tweet untuk Diprediksi:", "I love this new update!")
        submitted = st.form_submit_button("Prediksi")

        if submitted and tweet_input.strip() != "":
            input_vec = vectorizer.transform([tweet_input])
            prediction = model.predict(input_vec)[0]
            st.success(f"Prediksi Sentimen untuk Tweet ini adalah: {prediction}")
