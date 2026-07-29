import streamlit as st
import joblib
from pathlib import Path
BASE_DIR=Path(__file__).parent
model=joblib.load(BASE_DIR/'sentiment_model.pkl')
vectorizer=joblib.load(BASE_DIR/'tfidf_vectorizer.pkl')
st.title('AI Text Sentiment Analyzer')
text=st.text_area('Enter a movie review')
if st.button('Analyze'):
    if text.strip():
        X=vectorizer.transform([text])
        pred=int(model.predict(X)[0])
        probs=model.predict_proba(X)[0]
        d={int(c):float(p) for c,p in zip(model.classes_,probs)}
        st.write('Prediction:', 'Positive' if pred==1 else 'Negative')
        st.write(d)
    else:
        st.warning('Please enter a review.')
