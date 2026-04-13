#  Nueral Networks for Sentiment Analysis classification

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

# Mapping review vocab to integer indices and encoding reviews as padded sequences
def build_vocab(texts, max_vocab=20_000):
    
    # Vocabulary -> Integer mapping based on word frequency
    counter = Counter(word for text in texts for word in text.split())
    vocab   = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab

# Encode review text into integer indices based on vocab, with padding/truncation to max_len
def encode(text, vocab, max_len=300):
    
    # Encoding string into integer indices based on vocab, with padding/truncation to max_len
    tokens = text.split()[:max_len]
    indices = [vocab.get(t, 1) for t in tokens]
    padded  = indices + [0] * (max_len - len(indices))
    return padded


# Dataset class for loading reviews and labels, applying encoding to convert text to integer sequences
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=300):
        self.data   = [encode(t, vocab, max_len) for t in texts]
        self.labels = list(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx],   dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


# CNN architecture with embedding layer, multiple convolutional filters, max pooling, dropout, and final linear layer for binary classification
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100,
                 filter_sizes=(2, 3, 4), dropout=0.5):
        super().__init__()

        # vocab mapping to vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    # Forward pass through embedding, convolutional layers with ReLU and max pooling, dropout, and final linear layer
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        # Pooling outputs from multiple convolutional filters
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = c.max(dim=2).values
            pooled.append(c)
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)


# Training loop for the CNN model, including data loading, optimization, and validation accuracy reporting
def train_cnn(X_train, y_train, X_test, y_test,
              epochs=5, batch_size=64, lr=1e-3):

    vocab       = build_vocab(X_train)
    vocab_size  = len(vocab)

    train_ds    = ReviewDataset(X_train, y_train, vocab)
    test_ds     = ReviewDataset(X_test,  y_test,  vocab)
    train_dl    = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl     = DataLoader(test_ds,  batch_size=batch_size)

    model       = TextCNN(vocab_size)
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    criterion   = nn.BCEWithLogitsLoss()

    print("Neural Network (CNN): Training")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_dl:
                preds   = torch.sigmoid(model(X_batch)) > 0.5
                correct += (preds == y_batch.bool()).sum().item()
                total   += len(y_batch)

        print(f"Epoch {epoch+1}/{epochs}  "
              f"Loss: {total_loss/len(train_dl):.4f}  "
              f"Accuracy: {correct/total:.4f}")

    print()
    return model, vocab