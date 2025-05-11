
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# تحميل البيانات
df = pd.read_csv("محرك_بحث_الأحكام_القضائية.csv")
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

# تحويل القواعد إلى تمثيل رقمي
embeddings = model.encode(df["القاعدة القضائية"].tolist(), convert_to_tensor=True)

st.set_page_config(page_title="مقارنة القضايا بالأحكام", layout="wide")
st.title("النظام الذكي لمقارنة القضايا بالأحكام القضائية")

query = st.text_area("ألصق هنا وصف القضية أو الطعن كما يكتبه المحامي:")

if st.button("ابحث عن أقرب الأحكام"):
    with st.spinner("يتم تحليل الوصف ومقارنة المعنى..."):
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_results = scores.argsort(descending=True)[:5]

        results = []
        for idx in top_results:
            results.append({
                "رقم الطعن": df.iloc[idx]["رقم الطعن"],
                "الدائرة": df.iloc[idx]["الدائرة"],
                "القاعدة القضائية": df.iloc[idx]["القاعدة القضائية"],
                "درجة التشابه": round(float(scores[idx]), 2)
            })

        st.success("تم العثور على أقرب الأحكام:")
        st.dataframe(pd.DataFrame(results))
