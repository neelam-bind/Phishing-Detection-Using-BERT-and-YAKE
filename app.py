import streamlit as st
from components.custom_component import custom_button  # Custom component (if any)
from transformers import BertForSequenceClassification, BertTokenizer
from preprocess import load_and_preprocess_data
import torch
import pandas as pd

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('models/bert_model')
tokenizer = BertTokenizer.from_pretrained('models/bert_model')

# Function to evaluate the trained model
def evaluate_model(model, tokenizer, test_data):
    inputs = tokenizer(test_data['cleaned_text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    labels = torch.tensor(test_data['label'].tolist())
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return accuracy, precision, recall, f1

# Streamlit UI
st.title("Phishing Detection Using BERT")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load and preprocess the data
    df = pd.read_csv(uploaded_file)
    _, test_data = load_and_preprocess_data(df)
    
    # Display data preview
    st.write("Preview of uploaded data:")
    st.write(test_data.head())

    # Button to evaluate the model
    if st.button("Evaluate Model"):
        accuracy, precision, recall, f1 = evaluate_model(model, tokenizer, test_data)
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1-Score: {f1}")
