# 🛡️ **Phishing Detection using Advanced NLP Techniques**  

This repository delves into the application of **state-of-the-art NLP models** like BERT and YAKE to enhance phishing detection systems. By exploring contextual understanding and keyword extraction, this project aims to tackle phishing threats in diverse forms effectively.  


## Problem Statement

Despite significant advancements in cybersecurity, phishing detection remains a challenging problem due to the dynamic and deceptive nature of these attacks. Traditional phishing detection systems, such as rule-based or heuristic approaches, often fail to keep pace with attackers’ creativity and adaptiveness. These systems struggle with subtle linguistic manipulations that exploit context and human tendencies, such as urgency in communication or official-sounding language.

The core challenge lies in developing a system capable of identifying phishing attempts in diverse, unstructured, and often noisy textual formats while maintaining a balance between accuracy, interpretability, and computational efficiency. 🧩


## 🎯 **Objective**  
To leverage **BERT (Bidirectional Encoder Representations from Transformers)** and **YAKE (Yet Another Keyword Extractor)** for:  
- 🔍 Identifying manipulative patterns in phishing emails and SMS (smishing).  
- 🛠️ Addressing challenges such as dataset imbalance, interpretability, and efficiency.  
- 🚀 Building robust phishing detection systems suitable for real-world applications.  


## 📝 **Scope**  
1. Analyze **email phishing** 📧 and **smishing (SMS phishing)** 📱.  
2. Utilize **BERT’s bidirectional contextual capabilities** to detect subtle phishing cues.  
3. Enhance feature extraction with **YAKE** for identifying malicious links and suspicious keywords.  
4. Propose scalable solutions for real-world phishing detection.  


## 🛠️ **Key Techniques**  

### 1️⃣ **BERT**  
- **Purpose**: Captures bidirectional context to detect subtle manipulations in phishing texts.  
- **Applications**: Analyzes phishing emails and SMS.  
- **Advancements**: Domain-specific fine-tuning significantly enhances detection accuracy.  

### 2️⃣ **YAKE**  
- **Purpose**: Extracts keywords and phrases critical for phishing classification.  
- **Applications**: Complements BERT by isolating phishing indicators in text.  
- **Integration**: Works alongside BERT for feature-rich detection systems.  


## 💡 **Applications**  

### 📧 **Email Phishing Detection**  
- Detects red flags such as urgency, suspicious requests, and embedded URLs.  
- Fine-tuned on phishing datasets for high precision.  

### 📱 **SMS Phishing Detection (Smishing)**  
- Analyzes brief, context-limited messages for signs of phishing.  
- Combines YAKE's keyword extraction with BERT’s contextual understanding.  


## Existing Work

Phishing detection has evolved from basic rule-based systems to sophisticated machine learning algorithms. Traditional models, such as Naive Bayes, Support Vector Machines (SVM), and Random Forests, have been widely used for email classification and phishing detection. However, these models often lack the ability to deeply understand the context or nuances of the text.

The advent of advanced NLP models, particularly transformer-based architectures like **BERT (Bidirectional Encoder Representations from Transformers)**, has introduced new possibilities. BERT excels in understanding bidirectional context and identifying subtle manipulations in language. Lightweight tools like **YAKE (Yet Another Keyword Extractor)** also play a crucial role in detecting phishing-specific keywords, enhancing the overall detection accuracy. 📈


## 🚩 **Research Gaps and Challenges**  

### 1. Dataset Imbalance  
- **Problem**: Limited and skewed phishing datasets hinder model performance.  
- **Solution**: Employ data augmentation, oversampling, or synthetic data generation.  

### 2. Interpretability  
- **Problem**: BERT’s complex architecture lacks transparency in predictions.  
- **Solution**: Integrate tools like **SHAP (SHapley Additive exPlanations)** for explainability.  


## Methodology

### BERT: Contextual Language Understanding 🧠

BERT is a transformer-based model that understands the bidirectional context of input text. Unlike traditional models that process text in one direction, BERT analyzes the entire sequence, capturing the relationship between words more effectively. This bidirectional analysis is especially useful in detecting phishing cues, as it helps understand deceptive phrases and context.

#### Key Components of BERT:
- **Self-Attention Mechanism**: Determines the importance of each word in relation to others.
- **Positional Encoding**: Ensures that the order of words is retained.
- **Fine-Tuning**: For phishing detection, BERT is fine-tuned using labeled data to classify phishing attempts.

### YAKE: Keyword Extraction 🔑

YAKE is an unsupervised keyword extraction tool that identifies important terms within a text. It evaluates keywords based on several features:
- **Term Frequency (TF)**: Frequency of a term in the document.
- **Document Frequency (DF)**: Number of documents containing the term.
- **Position of the Term**: Keywords appearing early in the document get more weight.

YAKE uses these features to calculate the relevance score of each keyword.


### Data Preprocessing 🧹

Data preprocessing steps ensure that the input data is clean and ready for analysis:
- **Text Cleaning**: Removal of noise like special characters, URLs, and digits.
- **Tokenization**: Breaking the text into smaller units (tokens) using **WordPiece** tokenization for handling rare words.
- **Text Normalization**: Standardization by converting to lowercase, expanding contractions, and removing stopwords.


### Training & Evaluation 🏋️

The model is trained using **supervised learning** with labeled data. The loss function used is **Cross-Entropy Loss**, and the optimizer is **Adam**. Evaluation metrics like **precision, recall, F1 score**, and **accuracy** are used to assess the model’s performance.


## Code Implementation

The code implementation can be found in the project [Google Colab notebook](https://colab.research.google.com/drive/1TTkSrigsT6PT5f6DdJMUvtntyq9onvjx?usp=sharing). Below is a brief snippet:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the dataset
file_path = '/content/phishingemails/Phishing_Email.csv'
email_dataset = pd.read_csv(file_path)

# Mapping labels
label_to_int = {'Safe Email': 0, 'Phishing Email': 1}
email_dataset['Email Type'] = email_dataset['Email Type'].map(label_to_int)

# Splitting dataset into train and test
train_emails, test_emails, train_labels, test_labels = train_test_split(
    email_dataset['Email Text'], email_dataset['Email Type'], test_size=0.2, random_state=42)


## Results

The system achieved significant improvements in detecting phishing emails and SMS, using a combination of **BERT** and **YAKE**. This hybrid approach effectively handled dataset imbalance, capturing subtle phishing cues while maintaining interpretability.

## Future Directions 🔮
While the current system shows promise, there are areas for future improvement:

- **Handling Multilingual Datasets**: Extending the system to support multiple languages.
- **Real-time Detection**: Optimizing the model for real-time phishing detection in emails and SMS.
- **Explainability**: Integrating explainability tools like SHAP to improve model transparency.

## Conclusion
This project demonstrates the powerful synergy between **BERT** and **YAKE** for phishing detection. By combining contextual language models and keyword extraction techniques, we have created a more accurate and interpretable solution for phishing detection, adaptable to various forms of phishing attacks. 🚀

## References 📚
- Gupta, et al. (2024) - Phishing detection using BERT.
- Chakkarwar, et al. (2023) - Transformer-based models in NLP.
- Salloum, et al. (2021) - Keyword extraction for phishing emails.
- Koroteev (2021) - Handling dataset imbalance in NLP models.


