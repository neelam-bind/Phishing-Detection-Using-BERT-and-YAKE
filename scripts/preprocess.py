import re
import numpy as np
import pandas as pd

def preprocess_step_texts(text_list):
    """
    Function to preprocess a list of texts (emails).
    It applies several text-cleaning steps like removing special characters, HTML tags, digits, and more.
    """
    # Remove HTML-like tags
    text_list = [re.sub("<\w+>", '', text) for text in text_list]
    # Remove text inside curly braces
    text_list = [re.sub("{.*}", '', text) for text in text_list]
    # Remove text with special characters like =, <, >
    text_list = [re.sub("\S+[=<>]\S+", '', text) for text in text_list]
    # Remove URLs
    text_list = [re.sub("https?://\S+", '', text) for text in text_list]
    # Remove punctuation and special characters
    text_list = [re.sub("[.,!?\():\"<>#']", ' ', text) for text in text_list]
    # Replace hyphens with spaces
    text_list = [re.sub("-", ' ', text) for text in text_list]
    # Remove digits
    text_list = [re.sub("\d+", '', text) for text in text_list]
    # Remove text inside square brackets
    text_list = [re.sub("\[\w*\]", '', text) for text in text_list]
    # Remove various special characters
    text_list = [re.sub("[_$@/=\*%+|;]", '', text) for text in text_list]
    # Remove backslashes
    text_list = [re.sub("[\\\]", '', text) for text in text_list]
    # Remove carets
    text_list = [re.sub("[\^]", '', text) for text in text_list]
    # Replace multiple spaces with a single space
    text_list = [re.sub(" {2,}", ' ', text) for text in text_list]
    # Filter out words with less than 3 characters
    text_list = [" ".join(filter(lambda word: len(word) > 2, text.split())) for text in text_list]
    return text_list

def load_and_preprocess_data(file_path):
    """
    Function to load data and apply text preprocessing steps.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Apply preprocessing steps to the text data
    df['cleaned_text'] = preprocess_step_texts(df['text'].tolist())
    
    # Filter out emails with less than 5 words
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] > 5]
    
    # Return cleaned and filtered data
    return df[['cleaned_text', 'label']]

# Example usage
train_data = load_and_preprocess_data('data/train.csv')
test_data = load_and_preprocess_data('data/test.csv')

# To get the final emails and labels
final_emails = train_data['cleaned_text'].tolist()
final_labels = np.array(train_data['label'].tolist())

