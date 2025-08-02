import pandas as pd

# Load the Excel file
file_path = "/Users/aravind/Projects/IntrusionDetection/IntrusionDetection/NSL KDD/FeatureSelectionBinary/feature_ranking/feature_ranking_plot/feature_ranking.xlsx"
excel_data = pd.read_excel(file_path, sheet_name=None, header=None)

# Feature name → feature number
feature_number_map = {
    'duration': 'f1', 'protocol_type': 'f2', 'service': 'f3', 'flag': 'f4',
    'src_bytes': 'f5', 'dst_bytes': 'f6', 'land': 'f7', 'wrong_fragment': 'f8',
    'urgent': 'f9', 'hot': 'f10', 'num_failed_logins': 'f11', 'logged_in': 'f12',
    'num_compromised': 'f13', 'root_shell': 'f14', 'su_attempted': 'f15',
    'num_root': 'f16', 'num_file_creations': 'f17', 'num_shells': 'f18',
    'num_access_files': 'f19', 'num_outbound_cmds': 'f20', 'is_host_login': 'f21',
    'is_guest_login': 'f22', 'count': 'f23', 'srv_count': 'f24', 'serror_rate': 'f25',
    'srv_serror_rate': 'f26', 'rerror_rate': 'f27', 'srv_rerror_rate': 'f28',
    'same_srv_rate': 'f29', 'diff_srv_rate': 'f30', 'srv_diff_host_rate': 'f31',
    'dst_host_count': 'f32', 'dst_host_srv_count': 'f33',
    'dst_host_same_srv_rate': 'f34', 'dst_host_diff_srv_rate': 'f35',
    'dst_host_same_src_port_rate': 'f36', 'dst_host_srv_diff_host_rate': 'f37',
    'dst_host_serror_rate': 'f38', 'dst_host_srv_serror_rate': 'f39',
    'dst_host_rerror_rate': 'f40', 'dst_host_srv_rerror_rate': 'f41'
}

# Feature selection sets
sets = {
    "Entropy-Based": ["info_gain", "gain_ratio", "su", "gini_index"],
    "Statistical": ["chi_2", "pearson"],
    "Similarity-Based": ["reliefF", "fischer_score", "passi_luuka"]
}

# Print set definitions
print("📌 Feature Selection Sets:")
print("==========================")
for set_name, sheets in sets.items():
    print(f"🔹 {set_name}: {', '.join(sheets)}")
print("\n")

# Read top 10 features from each sheet
top_features_dict = {}
for sheet_name, df in excel_data.items():
    df = df.iloc[1:, :2]  # Skip header row
    df.columns = ['Rank', 'Feature']
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df = df.dropna(subset=['Rank'])
    top_features = df.nsmallest(10, 'Rank')['Feature'].tolist()
    top_features_dict[sheet_name] = set(top_features)

# Helper function to format feature with number
def format_feature(f):
    return f"{f} ({feature_number_map.get(f, 'N/A')})"

# Process each set
for set_name, sheet_list in sets.items():
    feature_sets = [top_features_dict[sheet] for sheet in sheet_list if sheet in top_features_dict]

    union_features = set().union(*feature_sets)
    intersection_features = set.intersection(*feature_sets) if feature_sets else set()

    print(f"\n📂 {set_name} Methods")
    print("=" * (len(set_name) + 10))

    print("\n🔗 Union of Top Features:")
    for i, feature in enumerate(sorted(union_features), 1):
        print(f"   {i}. {format_feature(feature)}")

    print("\n🤝 Intersection of Top Features:")
    if intersection_features:
        for i, feature in enumerate(sorted(intersection_features), 1):
            print(f"   {i}. {format_feature(feature)}")
    else:
        print("   (No common features found within this group)")
