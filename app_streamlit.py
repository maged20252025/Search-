import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# تحميل البيانات
@st.cache_data
def load_data():
    return pd.read_csv("محرك_بحث_الأحكام_القضائية.csv")

df = load_data()

# إعداد النموذج
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df["القاعدة القضائية"].astype(str))

# واجهة التطبيق
st.title("النظام الذكي لمقارنة القضايا بالأحكام القضائية")
st.write("ألصق هنا وصف القضية أو الطعن كما يكتبه المحامي:")

query = st.text_area("")

if st.button("ابحث عن أقرب الأحكام"):
    if not query.strip():
        st.warning("يرجى إدخال وصف القضية أولاً.")
    else:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, vectors)[0]
        top_idx = similarities.argmax()
        
        st.subheader("أقرب حكم:")
        st.write(f"**رقم الطعن:** {df.iloc[top_idx]['رقم الطعن']}")
        st.write(f"**القاعدة القضائية:** {df.iloc[top_idx]['القاعدة القضائية']}")
