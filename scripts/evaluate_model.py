import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from datasets import Dataset

# Preprocessing function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    """
    Load and preprocess data (cleaning and tokenizing).
    """
    # Load dataset from CSV
    df = pd.read_csv(file_path)
    
    # Assuming the dataset has 'text' and 'label' columns
    # Perform basic text cleaning
    df['cleaned_text'] = df['text'].str.replace(r'\<.*?\>', '', regex=True)  # Remove HTML tags
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'\d+', '', regex=True)  # Remove digits
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
    
    # Convert to Dataset object from Huggingface
    dataset = Dataset.from_pandas(df)
    
    # Split into train and test datasets if needed (use the full dataset as test data for now)
    test_data = dataset
    
    return dataset, test_data

# Function to evaluate the trained model on test data
def evaluate_model(model, tokenizer, test_data):
    """
    Function to evaluate the trained model on test data.
    """
    # Tokenize test data
    inputs = tokenizer(test_data['cleaned_text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    # Get predictions from the model
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Evaluate accuracy and F1 score
    labels = torch.tensor(test_data['label'].tolist())
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return accuracy, precision, recall, f1

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('models/bert_model')
tokenizer = BertTokenizer.from_pretrained('models/bert_model')

# Load and preprocess test data
_, test_data = load_and_preprocess_data('data/test.csv')

# Evaluate the model
accuracy, precision, recall, f1 = evaluate_model(model, tokenizer, test_data)

# Print evaluation results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

