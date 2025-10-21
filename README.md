# Music Genre Clustering with Neural Embeddings and Fuzzy Logic

This project builds a hybrid system to analyze and cluster songs based on their musical characteristics. It first trains a neural network to learn feature embeddings that capture the underlying relationships between audio attributes (e.g., energy, danceability, valence, tempo). These embeddings are then grouped using Fuzzy C-Means, producing clusters that allow genres to belong to multiple groups with varying degrees of membership, reflecting the inherent overlap in musical styles.
The system is trained and evaluated on a curated subset of the Spotify Tracks Dataset, with 10 main genre classes, providing a balance between model performance and interpretability.
