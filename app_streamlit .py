import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# تحميل البيانات
@st.cache_data
def load_data():
    return pd.read_csv("محرك_بحث_الأحكام_القضائية_محدث_كامل.csv")

df = load_data()

# تحميل النموذج
model = SentenceTransformer('all-MiniLM-L6-v2')

# تحويل القواعد إلى تمثيلات عددية
sentences = df['القاعدة القضائية'].astype(str).tolist()
sentence_embeddings = model.encode(sentences)

# واجهة المستخدم
st.title("النظام الذكي لمقارنة القضايا بالأحكام القضائية")
st.write(":ألصق هنا وصف القضية أو الطعن كما يكتبه المحامي")

query = st.text_area("")
if st.button("ابحث عن أقرب الأحكام"):
    if not query.strip():
        st.warning("يرجى إدخال وصف القضية أولاً.")
    else:
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
        top_idx = similarities.argmax()

        top_score = similarities[top_idx]
        threshold = 0.45

        if top_score >= threshold:
            st.subheader(":أقرب حكم")
            st.write(f"**رقم الطعن:** {df.iloc[top_idx]['رقم الطعن']}")
            st.write(f"**القاعدة القضائية:** {df.iloc[top_idx]['القاعدة القضائية']}")
        else:
            st.warning("لم يتم العثور على حكم مشابه بدرجة كافية. حاول إعادة صياغة الوصف.")