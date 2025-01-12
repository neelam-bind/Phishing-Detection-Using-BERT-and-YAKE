import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import pandas as pd
import yake
import numpy as np
from scripts.preprocess import preprocess_text

# Load pre-trained models and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('models/bert_model', num_labels=2)
tokenizer = DistilBertTokenizer.from_pretrained('models/bert_model')
yake_model = yake.KeywordExtractor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Title of the app
st.title("Phishing Email Detection")

# Input text box for users to input email or message
input_text = st.text_area("Enter Email or Message Text", "")

# Button to make prediction
if st.button("Detect Phishing"):
    if input_text:
        # Preprocess the input text
        processed_text = preprocess_text(input_text)

        # Tokenize the text
        inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True).to(device)

        # Predict using the BERT model
        with torch.no_grad():
            logits = model(**inputs).logits
            prediction = torch.argmax(logits, dim=1).item()

        # Show prediction result
        result = "Phishing Email" if prediction == 1 else "Safe Email"
        st.write(f"Prediction: {result}")
        
        # Extract keywords using YAKE
        keywords = yake_model.extract_keywords(input_text)
        st.write("Extracted Keywords:")
        for kw in keywords:
            st.write(f"- {kw[0]}")

# Footer
st.markdown("### Powered by Streamlit")
