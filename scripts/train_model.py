from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load dataset (replace with your dataset file paths)
dataset = load_dataset('csv', data_files={'train': 'data/train.csv', 'test': 'data/test.csv'})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory for model predictions and checkpoints
    evaluation_strategy="epoch",     # Evaluation strategy to use
    learning_rate=2e-5,              # Learning rate for optimization
    per_device_train_batch_size=8,   # Batch size per device during training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    num_train_epochs=3,              # Number of training epochs
    weight_decay=0.01,               # Weight decay for optimization
)

# Initialize Trainer
trainer = Trainer(
    model=model,                     # The model to be trained
    args=training_args,              # Training arguments
    train_dataset=tokenized_datasets['train'],  # Training dataset
    eval_dataset=tokenized_datasets['test'],    # Evaluation dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('models/bert_model')
tokenizer.save_pretrained('models/bert_model')

