import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
import seaborn as sns

# Load MNIST data
train_images = idx2numpy.convert_from_file("mnist/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("mnist/train-labels.idx1-ubyte")
test_images = idx2numpy.convert_from_file("mnist/t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("mnist/t10k-labels.idx1-ubyte")

# Combine and preprocess
X = np.concatenate((train_images, test_images)).reshape(-1, 28*28) / 255.0
y = np.concatenate((train_labels, test_labels))

# K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X)

# Map each cluster to the most frequent true label
def map_clusters(clusters, true_labels):
    label_map = np.zeros(10, dtype=int)
    for i in range(10):
        mask = clusters == i
        if np.any(mask):
            label_map[i] = mode(true_labels[mask], keepdims=True).mode[0]
    return label_map

label_map = map_clusters(clusters, y)
predicted_labels = np.array([label_map[c] for c in clusters])

# Accuracy
print(f"Accuracy: {accuracy_score(y, predicted_labels):.2f}")

# Confusion Matrix
sns.heatmap(confusion_matrix(y, predicted_labels), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# PCA visualization
X_pca = PCA(n_components=2).fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', s=1)
plt.title("K-Means Clustering (PCA 2D)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.colorbar(label="Cluster")
plt.show()

