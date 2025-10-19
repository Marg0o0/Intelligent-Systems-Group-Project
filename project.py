import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score,classification_report
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
import kagglehub
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
print("imports done")

#%% dataset

# download dataset
path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")

# load CSV into a pandas dataframe
dataset = pd.read_csv(f"{path}/dataset.csv")

# see the first rows
#print(dataset.head())
#print(dataset.shape)

# see the data available
#print(list(dataset.columns)) 
#print(dataset.iloc[5])

#%% pre processing

#remove duplicates, speech-like tracks and rows with missing entries
dataset = dataset.drop_duplicates(subset=["track_name", "artists"], keep="first")
dataset = dataset[dataset['speechiness'] <= 0.66]
dataset = dataset.dropna() 

#number of unique genres after pre-processing
#num_genres = dataset['track_genre'].nunique()

#%% feature selection

#select relevant features
selected_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'acousticness', 'instrumentalness', 'valence', 'tempo'
]

#subset the dataset
X = dataset[selected_features].values
y = dataset['track_genre'].values

#create labels for the genres
le = LabelEncoder()
y_id = le.fit_transform(y)
num_genres = len(le.classes_)

#%% train test split

#split proportions
test_size = 0.15
val_size = 0.15

#split into train (.7) and temp (.3)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_id, test_size=(test_size + val_size), random_state=42,
)

#split temp into validation (.15) and test (.15)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(test_size / (test_size + val_size)), random_state=42,
)

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


#%% standardizing variables (Key, loudness, tempo)

scaler=StandardScaler()
X_train[:, [2, 3, 8]] = scaler.fit_transform(X_train[:, [2, 3, 8]])
X_val[:, [2, 3, 8]]   = scaler.transform(X_val[:, [2, 3, 8]])
X_test[:, [2, 3, 8]]  = scaler.transform(X_test[:, [2, 3, 8]])

#print(X_train)

#%% data preparation and defining hyperparameters

#hyperparameters
embedding_dim = 30
num_epochs = 200
learning_rate = 0.001
dropout = 0.2
batch_size = 64

#data conversion to PyTorch tensors
X_train = torch.tensor(X_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.long)
X_val = torch.tensor(X_val, dtype = torch.float32)
y_val = torch.tensor(y_val, dtype = torch.long)

#create dataset and loader
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#%% encoders

#genre encoder (y -> z)
class GenreEncoder(nn.Module):
    def __init__(self, num_genres, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_genres, embedding_dim)

    def forward(self, genre_id):
        if genre_id.dim() == 2 and genre_id.size(1) == 1:
            genre_id = genre_id.squeeze(1)
        z = self.embedding(genre_id)  # [batch, embedding_dim]
        return z

#feature encoder (x -> ẑ)
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout_prob = 0.3):
        super().__init__()
        self.input = nn.Linear (input_dim, 64)
        self.hidden1 = nn.Linear (64, 64)
        self.hidden2 = nn.Linear (64, 64)
        self.out = nn.Linear (64, embedding_dim)
        self.dropout = nn.Dropout (p = dropout_prob)


    def forward(self, x):
        x = F.relu (self.input(x)); x = self.dropout(x)

        x = F.relu (self.hidden1(x)); x = self.dropout(x)

        x = F.relu (self.hidden2(x)); x = self.dropout(x)

        z_hat = self.out(x)
        return z_hat  # ẑ ∈ R^(batch_size × embedding_dim)

#%% creating the instances to use in the training loop

#
genre_embedding = GenreEncoder(num_genres, embedding_dim)

#
model = FeatureEncoder(
    input_dim = X_train.shape[1], 
    embedding_dim = embedding_dim, dropout_prob = dropout
)

#loss
criterion = nn.CosineSimilarity(dim=1)

#optimizer
optimizer = optim.Adam(
    list(model.parameters()) + list(genre_embedding.parameters()),
    lr = learning_rate
)

#%% training Loop

for epoch in range(num_epochs):
    #ativa modo de treino nos dois modelos
    model.train()
    genre_embedding.train()

    total_loss = 0.0

    for x_feat, y_genre in train_dataloader:
        optimizer.zero_grad()
        
        z = genre_embedding (y_genre) # [batch, embedding_dim]
        
        z_hat = model (x_feat)  # [batch, embedding_dim]

        cos_sim = criterion( z_hat, z)
        loss = 1 - cos_sim.mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

