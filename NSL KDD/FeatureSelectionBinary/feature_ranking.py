import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif,r_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets for training and testing
train_data = pd.read_csv("../data/BinaryClassify/train_nsl_kdd_binary_encoded.csv")
test_data = pd.read_csv("../data/BinaryClassify/test_nsl_kdd_binary_encoded.csv")

# Reduce sample size for testing
test_data = test_data.sample(frac=0.1, random_state=42)

# Features and labels for training
X_train = train_data.drop("binaryoutcome", axis=1)
y_train = train_data["binaryoutcome"]

# Features and labels for testing
X_test = test_data.drop("binaryoutcome", axis=1)
y_test = test_data["binaryoutcome"]


# Define a function to apply a filter method, evaluate accuracy, and store results
def evaluate_filter(filter_method, name, X_train, y_train, X_test, y_test):
    filter_model = filter_method
    accuracy_scores = []
    num_features = []

    # Fit the filter model
    filter_model.fit(X_train, y_train)

    # Sort features or use as is based on the method
    if hasattr(filter_model, 'scores_'):
        feature_info = list(zip(X_train.columns, filter_model.scores_))
        feature_sorted = X_train.columns[np.argsort(filter_model.scores_)[::-1]]
    elif hasattr(filter_model, 'get_support'):
        retained_features = X_train.columns[filter_model.get_support()]
        feature_info = list(zip(X_train.columns, filter_model.get_support()))
        feature_sorted = retained_features

    print(f"Feature Information: {feature_info}")

    for i in range(1, X_train.shape[1] + 1):
        # Extract top i features based on your sorted feature list
        x_train_filtered = X_train[feature_sorted[:i]]
        x_test_filtered = X_test[feature_sorted[:i]]

        # Ensure at least one feature is selected
        if x_train_filtered.shape[1] > 0:
            # Train the model on the filtered training data
            clf = RandomForestClassifier()
            clf.fit(x_train_filtered, y_train)

            # Test the model on the filtered test data
            accuracy = clf.score(x_test_filtered, y_test)
            accuracy_scores.append(accuracy)
            num_features.append(i)

            print(
                f"Iteration {i}: {name} - Num Features: {i}, Test Accuracy: {accuracy:.4f}")

    plt.plot(num_features, accuracy_scores, label=name)


# Apply different filter methods
variance_threshold = VarianceThreshold(threshold=0.0)
evaluate_filter(variance_threshold, "Variance Threshold", X_train, y_train, X_test, y_test)

select_k_best_anova = SelectKBest(f_classif, k=X_train.shape[1])
evaluate_filter(select_k_best_anova, "ANOVA", X_train, y_train, X_test, y_test)

select_k_best_mutual_info = SelectKBest(mutual_info_classif, k=X_train.shape[1])
evaluate_filter(select_k_best_mutual_info, "Mutual Information", X_train, y_train, X_test, y_test)

select_k_best_chi2 = SelectKBest(chi2, k=X_train.shape[1])
evaluate_filter(select_k_best_chi2, "Chi-Squared", X_train, y_train, X_test, y_test)

plt.xlabel("Number of Features")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Comparison of Filter Methods")
plt.legend()  # Add legend for better visualization
plt.show()
