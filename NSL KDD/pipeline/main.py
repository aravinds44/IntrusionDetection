from data_loader import load_nsl_kdd_data
from feature_selection import feature_selection_pipeline

# Load the dataset
data_dir = "../data/BinaryClassify"
train_data, test_data = load_nsl_kdd_data(data_dir)

# Separate features and target variable
X_train = train_data.drop(columns=["binaryoutcome"])  # Assuming 'label' is the target column
y_train = train_data["binaryoutcome"]

# Test feature selection methods
methods = ["chi2", "mutual_info", "random_forest"]
k = 10  # Number of top features to select

for method in methods:
    X_selected, selected_features = feature_selection_pipeline(X_train, y_train, method=method, k=k)
    print(f"Top {k} features selected using {method}: {list(selected_features)}")
