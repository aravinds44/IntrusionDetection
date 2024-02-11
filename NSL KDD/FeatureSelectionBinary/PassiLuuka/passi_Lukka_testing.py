

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def feat_sel_sim(data, measure, p=1):
    """
    Feature selection method using similarity measure and fuzzy entropy measures.

    Args:
    data (numpy.ndarray): Data matrix containing class values.
    measure (str): Fuzzy entropy measure, either 'luca' or 'park'.
    p (float): Parameter of Lukasiewicz similarity measure. Default is 1.

    Returns:
    data_mod (numpy.ndarray): Data without removed feature.
    index_rem (int): Index of removed feature in original data.
    """

    l = np.max(data[:, -1])  # #-classes
    m = data.shape[0]        # #-samples
    t = data.shape[1] - 1    # #-features

    dataold = data.copy()



    # Extract features and labels
    data_v = data[:, :t]
    data_c = data[:, t]
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    # Fit the scaler to the feature data and transform it
    data_v = scaler.fit_transform(data_v)
    # Concatenate scaled features with labels
    data = np.hstack((data_v, data_c.reshape(-1, 1)))

    # Forming idealvec using arithmetic mean
    idealvec_s = np.zeros((l, t))
    for k in range(1, l + 1):
        idealvec_s[k - 1, :] = np.mean(data[np.where(data[:, -1] == k)][:, :t], axis=0)

    # Sample data
    datalearn_s = data[:, :t]

    # Initialize similarity matrix
    sim = np.zeros((l, m, t))

    #Compute similarities
    for i in range(l):

        mask = (data[:,-1] == i+1)
        sim[i][mask] = (1 - np.abs(idealvec_s[i]**p - datalearn_s[mask] **p))**(1/p)

    A = np.sum(sim[0],axis=0)
    B = np.sum(sim[1],axis=0)

    Mu = A/(A+B)

    H = -(Mu * np.log(Mu) +(1-Mu)* np.log(1 - Mu))


    # # Compute similarities
    # for j in range(m):
    #     for i in range(t):
    #         for k in range(l):
    #             sim[i, j, k] = (1 - np.abs(idealvec_s[k, i] ** p - datalearn_s[j, i]) ** p) ** (1 / p)

    # Reduce dimensions in sim
    # sim = np.reshape(sim, (t, m * l)).T
    # H=None
    # # Possibility for two different entropy measures
    # if measure == 'luca':
    #     # Modifying zero and one values of the similarity values to work with De Luca's entropy measure
    #     delta = 1E-10
    #     sim[sim == 0] = delta
    #     sim[sim == 1] = 1 - delta
    #     H = np.sum(-sim * np.log(sim) - (1 - sim) * np.log(1 - sim), axis=0)
    # elif measure == 'park':
    #     H = np.sum(np.sin(np.pi / 2 * sim) + np.sin(np.pi / 2 * (1 - sim)) - 1, axis=0)


    # Find maximum feature
    index_rem = np.argmax(H)

    # Removing feature from the data
    data_mod = np.hstack((dataold[:, :index_rem], dataold[:, index_rem + 1:]))

    return data_mod, index_rem


# # Sample data
# data = np.array([
#     [1, 2, 3, 1],
#     [4, 5, 6, 2],
#     [7, 8, 9, 1],
#     [10, 11, 12, 2],
# ])
data = pd.read_csv("../data/BinaryClassify/train_nsl_kdd_binary_encoded.csv")
data = data.values.astype(int)
data[:,-1] = data[:,-1]+1
# Run the function with 'luca' measure
data_mod, index_rem = feat_sel_sim(data, measure='luca')

print("Modified data:", data_mod)
print("Removed feature index:", index_rem)
