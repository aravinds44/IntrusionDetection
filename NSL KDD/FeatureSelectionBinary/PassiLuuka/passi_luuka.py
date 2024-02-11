import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def passi_luuka_feat_sel(data, measure='luca', p=1):
    num_classes = int(np.max(data[:, -1]) + 1)  # Number of classes
    num_samples = data.shape[0]             # Number of samples
    num_features = data.shape[1] - 1        # Number of features

    data_old = data.copy()

    # Extract features and labels
    data_v = data[:, :num_features]
    data_c = data[:, num_features]

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the feature data and transform it
    data_v = scaler.fit_transform(data_v)

    # Concatenate scaled features with labels
    data = np.hstack((data_v, data_c.reshape(-1, 1)))

    # Forming idealvec using arithmetic mean
    idealvec_s = np.zeros((num_classes, num_features))
    for k in range(num_classes):
        idealvec_s[k, :] = np.mean(data[data[:, -1] == k][:, :num_features], axis=0)

    # Sample data
    datalearn_s = data[:, :num_features]

    # Initialize similarity matrix
    sim = np.zeros((num_classes, num_samples, num_features))

    # Compute similarities
    for i in range(num_classes):
        mask = (data[:, -1] == i)
        sim[i][mask] = (1 - np.abs(idealvec_s[i] ** p - datalearn_s[mask] ** p)) ** (1 / p)

    delta = 1E-10
    sim[sim == 0] = delta
    sim[sim == 1] = 1 - delta

    A = np.sum(sim[0], axis=0)
    B = np.sum(sim[1], axis=0)

    Mu = A / (A + B)

    Mu_2 = B / (A+B)

    H = -(Mu * np.log(Mu) + (Mu_2) * np.log(Mu_2))


    # H = -(Mu * np.log(Mu) + (1 - Mu) * np.log(1 - Mu))

    return H

# Load sample data
# data = pd.read_csv("../data/BinaryClassify/train_nsl_kdd_binary_encoded.csv")
# data = np.array(data)
data = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 0],
    [7, 8, 9, 1],
    [10, 11, 12, 1],
])



entropy = passi_luuka_feat_sel(data)

print(entropy)