import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io
import contextlib

# Read the dataset
creditcard_df = pd.read_csv('Marketing_data.csv')

# Open an output file to capture printed outputs
with open("output.txt", "w") as f_out:
    with contextlib.redirect_stdout(f_out):
        print("=== Data Summary ===")
        # .info() writes directly to stdout
        creditcard_df.info()
        
        print("\n=== Statistical Description ===")
        print(creditcard_df.describe())
        
        # Plot missing data heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(creditcard_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
        plt.title("Missing Data Heatmap")
        plt.savefig("missing_data_heatmap.png")
        plt.close()
        
        # Fill missing values for 'MINIMUM_PAYMENTS' with the mean
        creditcard_df.loc[creditcard_df['MINIMUM_PAYMENTS'].isnull(), 'MINIMUM_PAYMENTS'] = creditcard_df['MINIMUM_PAYMENTS'].mean()
        print("\n=== Missing Data After Imputation ===")
        print(creditcard_df.isnull().sum())
        
        # Check duplicate entries
        duplicates = creditcard_df.duplicated().sum()
        print("\nNumber of duplicate entries:", duplicates)

# Drop the 'CUST_ID' column as it's not needed
creditcard_df.drop('CUST_ID', axis=1, inplace=True)
creditcard_df.fillna(creditcard_df.mean(), inplace=True)
# Plot distributions for each column
plt.figure(figsize=(10, 50))
for i, col in enumerate(creditcard_df.columns):
    plt.subplot(len(creditcard_df.columns), 1, i + 1)
    # Using histplot with KDE (displot creates its own figure by default)
    sns.histplot(creditcard_df[col], kde=True, color="g", edgecolor="black")
    plt.title(col)
plt.tight_layout()
plt.savefig("distributions.png")
plt.close()

# Plot correlations heatmap
correlations = creditcard_df.corr()
f, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(correlations, annot=True)
plt.title("Correlations Heatmap")
plt.savefig("correlations_heatmap.png")
plt.close()

# Scale the data
scaler = StandardScaler()
creditcard_df_scaled = scaler.fit_transform(creditcard_df)

# Append scaled data shape to output file
with open("output.txt", "a") as f_out:
    with contextlib.redirect_stdout(f_out):
        print("\nScaled data shape:", creditcard_df_scaled.shape)

# Use the elbow method to decide on a number of clusters
scores_1 = []
range_values = range(1, 20)
for i in range_values:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(creditcard_df_scaled[:, :7])
    scores_1.append(kmeans.inertia_)

plt.figure()
plt.plot(range_values, scores_1, 'bx-')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.savefig("elbow_curve.png")
plt.close()

# Choose 5 clusters (as in your original code) and perform clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(creditcard_df_scaled)
labels = kmeans.labels_

# Retrieve cluster centers (scaled) and write them to output
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=creditcard_df.columns)
with open("output.txt", "a") as f_out:
    with contextlib.redirect_stdout(f_out):
        print("\n=== Cluster Centers (Scaled) ===")
        print(cluster_centers)

# Inverse transform the cluster centers for interpretation
cluster_centers_inv = scaler.inverse_transform(cluster_centers)
cluster_centers_inv = pd.DataFrame(cluster_centers_inv, columns=creditcard_df.columns)
with open("output.txt", "a") as f_out:
    with contextlib.redirect_stdout(f_out):
        print("\n=== Cluster Centers (Original Scale) ===")
        print(cluster_centers_inv)

# Concatenate the cluster labels with the original dataframe
creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster': labels})], axis=1)
with open("output.txt", "a") as f_out:
    with contextlib.redirect_stdout(f_out):
        print("\n=== First 5 Rows of Clustered Data ===")
        print(creditcard_df_cluster.head())

# Create a folder for histogram images if it doesn't exist
hist_folder = "histograms"
if not os.path.exists(hist_folder):
    os.makedirs(hist_folder)

# Define a list of colors to use for the different clusters
cluster_colors = ['red', 'green', 'blue', 'orange', 'purple']

# Plot histograms for each variable by cluster and save them in the separate folder
for col in creditcard_df.columns:
    plt.figure(figsize=(35, 5))
    for j in range(5):
        plt.subplot(1, 7, j + 1)
        cluster_data = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
        cluster_data[col].hist(bins=20, color=cluster_colors[j % len(cluster_colors)], edgecolor='black')
        plt.title(f"{col}\nCluster {j}")
    plt.tight_layout()
    plt.savefig(os.path.join(hist_folder, f"histogram_{col}.png"))
    plt.close()

# Perform PCA to reduce dimensions to 2 components for visualization
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(creditcard_df_scaled)
pca_df = pd.DataFrame(principal_comp, columns=['pca1', 'pca2'])
pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)
with open("output.txt", "a") as f_out:
    with contextlib.redirect_stdout(f_out):
        print("\n=== First 5 Rows of PCA Data ===")
        print(pca_df.head())

# Create and save the PCA scatter plot with cluster coloring
plt.figure(figsize=(10, 10))
sns.scatterplot(x="pca1", y="pca2", hue="cluster", data=pca_df,
                palette=['red', 'green', 'blue', 'pink', 'yellow'])
plt.title("PCA Scatter Plot of Clusters")
plt.savefig("pca_scatter.png")
plt.close()
