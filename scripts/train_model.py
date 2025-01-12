import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from tqdm import tqdm
from datasets import load_dataset  # Assuming dataset is from Huggingface datasets

# Load the dataset (using Huggingface Datasets for this example)
train_dataset = load_dataset("csv", data_files="data/train.csv")["train"]
val_dataset = load_dataset("csv", data_files="data/test.csv")["train"]

# Tokenizer initialization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Define DataLoader for training and validation
batch_size = 10
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Load the model with DistilBERT for binary classification (2 labels)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Optimizer setup (Adam optimizer)
optimizer = optim.Adam(model.parameters(), lr=1e-7)

# Move the model to GPU if available
if torch.cuda.is_available():
    model = model.to('cuda')

# Training settings
num_epochs = 10
train_loss, val_loss = [], []

# Model training loop
model.train()
for epoch in range(num_epochs):
    epoch_train_loss = 0
    epoch_val_loss = 0
    
    # Training phase
    for dictionary, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training"):
        input_ids = dictionary['input_ids'].to('cuda')
        attention_mask = dictionary['attention_mask'].to('cuda')
        labels = labels.to('cuda')

        # Forward pass
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        
        optimizer.zero_grad()  # Clear previous gradients
        epoch_train_loss += loss.item()  # Accumulate training loss
        loss.backward()  # Backpropagate gradients
        optimizer.step()  # Update model weights
    
    train_loss.append(epoch_train_loss)

    # Validation phase
    model.eval()
    with torch.no_grad():
        for dictionary, labels in tqdm(val_dataloader, desc=f"Epoch {epoch + 1} Validation"):
            input_ids = dictionary['input_ids'].to('cuda')
            attention_mask = dictionary['attention_mask'].to('cuda')
            labels = labels.to('cuda')

            # Forward pass on validation data
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            epoch_val_loss += loss.item()  # Accumulate validation loss
    
    val_loss.append(epoch_val_loss)
    
    # Print training and validation losses
    print(f"Epoch {epoch + 1}: Train loss = {epoch_train_loss:.5f}, Val loss = {epoch_val_loss:.5f}")

    model.train()  # Set model back to training mode after evaluation

# Save the trained model after training
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_model")

# Optionally, save the training and validation loss plots or logs for future analysis
