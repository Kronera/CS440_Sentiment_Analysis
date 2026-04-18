#  Neural Networks for Sentiment Analysis classification

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
    tokens  = text.split()[:max_len]
    indices = [vocab.get(t, 1) for t in tokens]
    padded  = indices + [0] * (max_len - len(indices))
    return padded


def load_glove(glove_path: str, vocab: dict, embed_dim: int) -> torch.Tensor:
    """
    Load a GloVe .txt file and build an embedding matrix aligned to vocab.

    - <PAD> (index 0) stays all zeros so padded positions contribute nothing.
    - <UNK> (index 1) is set to the mean of all loaded GloVe vectors so
      unknown words get a reasonable representation rather than random noise.
    - Every other word is looked up; if GloVe doesn't have it we fall back
      to the mean vector as well.

    Args:
        glove_path: path to a GloVe file, e.g. "glove.6B.100d.txt"
        vocab:      word -> index dict produced by build_vocab()
        embed_dim:  must match the dimensionality of the GloVe file (50/100/200/300)

    Returns:
        FloatTensor of shape (vocab_size, embed_dim) ready for nn.Embedding.from_pretrained()
    """
    print(f"Loading GloVe embeddings from {glove_path} ...")

    # Read every vector from the file into a plain dict
    glove_vectors = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            glove_vectors[word] = vec

    # Mean vector used for <UNK> and any out-of-vocabulary words
    all_vecs   = np.stack(list(glove_vectors.values()))
    mean_vec   = all_vecs.mean(axis=0)

    vocab_size    = len(vocab)
    weight_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    hits, misses = 0, 0
    for word, idx in vocab.items():
        if word in ("<PAD>",):
            # Keep zero vector for padding
            continue
        if word in glove_vectors:
            weight_matrix[idx] = glove_vectors[word]
            hits += 1
        else:
            # Unknown word — use mean vector
            weight_matrix[idx] = mean_vec
            misses += 1

    print(f"  GloVe coverage: {hits} hits, {misses} misses out of {vocab_size} vocab words")
    return torch.tensor(weight_matrix)


# Dataset class for loading reviews and labels
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


# CNN architecture: embedding → parallel conv filters → max pool → dropout → linear
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=100,
                 filter_sizes=(2, 3, 4), dropout=0.5,
                 pretrained_embeddings: torch.Tensor = None):
        super().__init__()

        if pretrained_embeddings is not None:
            # Load GloVe weights; frozen=True initially — unfreeze after warm-up epochs
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=True,          # frozen at first — see train_cnn for unfreeze logic
                padding_idx=0,
            )
            embed_dim = pretrained_embeddings.shape[1]  # honour actual GloVe dim
        else:
            # Fallback: random embeddings (original behaviour)
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(filter_sizes), 1)

    def freeze_embeddings(self):
        self.embedding.weight.requires_grad_(False)

    def unfreeze_embeddings(self):
        self.embedding.weight.requires_grad_(True)

    # Forward pass: embed → conv+ReLU+maxpool → concat → dropout → linear
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = c.max(dim=2).values
            pooled.append(c)

        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)


# Training loop with freeze/unfreeze schedule for pretrained embeddings
def train_cnn(X_train, y_train, X_test, y_test,
              epochs=10, batch_size=64, lr=1e-3,
              glove_path: str = None, glove_dim: int = 100,
              freeze_epochs: int = 3):
    """
    Train the TextCNN.

    Args:
        glove_path:    path to a GloVe .txt file. If None, random embeddings are used.
        glove_dim:     dimensionality of the GloVe file (must match filename, e.g. 100).
        freeze_epochs: number of epochs to keep embeddings frozen before unfreezing.
                       Only relevant when glove_path is provided.
        epochs:        total training epochs (increased default from 5 → 10 for GloVe).
    """
    vocab      = build_vocab(X_train)
    vocab_size = len(vocab)

    train_ds = ReviewDataset(X_train, y_train, vocab)
    test_ds  = ReviewDataset(X_test,  y_test,  vocab)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size)

    # Load GloVe if a path was supplied
    pretrained = None
    if glove_path:
        pretrained = load_glove(glove_path, vocab, glove_dim)

    model     = TextCNN(vocab_size, pretrained_embeddings=pretrained)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print("Neural Network (CNN): Training")
    if glove_path:
        print(f"  Embeddings frozen for first {freeze_epochs} epoch(s), then unfrozen.\n")

    for epoch in range(epochs):

        # Unfreeze embeddings after the warm-up period
        if glove_path and epoch == freeze_epochs:
            model.unfreeze_embeddings()
            print(f"  Epoch {epoch+1}: embeddings unfrozen — fine-tuning begins.\n")

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

        frozen_tag = " [frozen]" if (glove_path and epoch < freeze_epochs) else ""
        print(f"Epoch {epoch+1}/{epochs}  "
              f"Loss: {total_loss/len(train_dl):.4f}  "
              f"Accuracy: {correct/total:.4f}{frozen_tag}")

    print()
    return model, vocab