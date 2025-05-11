import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# تحميل النموذج مع تحديد الجهاز CPU
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

# تحميل قاعدة البيانات
@st.cache_data
def load_data():
    return pd.read_csv("محرك_بحث_الأحكام_القضائية.csv")

df = load_data()

# واجهة التطبيق
st.title("النظام الذكي لمقارنة القضايا بالأحكام القضائية")
query = st.text_area("ألصق هنا وصف القضية أو الطعن كما يكتبه المحامي:")

if st.button("ابحث عن أقرب الأحكام"):
    if query.strip() == "":
        st.warning("الرجاء إدخال وصف القضية.")
    else:
        # استخراج التمثيل العددي للنص المدخل
        query_embedding = model.encode(query, convert_to_tensor=True)

        # استخراج التمثيلات العددية للبيانات
        corpus_embeddings = model.encode(df["القاعدة القضائية"].tolist(), convert_to_tensor=True)

        # حساب التشابه
        similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]

        # ترتيب النتائج
        top_k = min(5, len(df))
        top_results = similarities.argsort(descending=True)[:top_k]

        st.subheader("أقرب الأحكام:")
        for idx in top_results:
            st.markdown(f"**رقم الطعن:** {df.iloc[idx]['رقم الطعن']}")
            st.markdown(f"**القاعدة القضائية:** {df.iloc[idx]['القاعدة القضائية']}")
            st.markdown("---")
