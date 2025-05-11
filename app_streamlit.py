import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# تحميل نموذج الذكاء الاصطناعي
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# تحميل ملف البيانات
@st.cache_data
def load_data():
    df = pd.read_csv("محرك_بحث_الأحكام_القضائية.csv")
    df.dropna(inplace=True)
    return df

df = load_data()

# إنشاء التمثيلات النصية للقاعدة القضائية
embeddings = model.encode(df["القاعدة القضائية"].tolist(), convert_to_tensor=True)

# واجهة المستخدم
st.title("النظام الذكي لمقارنة القضايا بالأحكام القضائية")
query = st.text_area("الصق هنا وصف القضية أو الطعن كما يكتبه المحامي:")
if st.button("ابحث عن أقرب الأحكام") and query.strip() != "":
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_k = min(5, len(df))
    top_results = scores.topk(k=top_k)

    st.subheader("أقرب الأحكام للقضية:")
    for idx in top_results.indices:
        رقم = df.iloc[idx]["رقم الطعن"]
        قاعدة = df.iloc[idx]["القاعدة القضائية"]
        st.markdown(f"**رقم الطعن:** {رقم}")
        st.markdown(f"**القاعدة القضائية:** {قاعدة}")
        st.markdown("---")
