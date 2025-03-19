# Unsupervised Machine Learning for Customer Segmentation

## Overview
This repository demonstrates how unsupervised machine learning can be applied to segment bank customers based on their transaction behaviors. By identifying distinct customer groups, businesses can target their marketing campaigns more effectively. The analysis is based on the dataset available from [Kaggle](https://www.kaggle.com/arjunbhasin2013/ccdata).

## Project Objectives
- **Enhanced Marketing Strategies:** Use machine learning to transform marketing initiatives through precise customer segmentation.
- **Data Processing & Visualization:** Import, clean, and visualize customer transaction data using Python libraries.
- **K-Means Clustering:** Apply the k-means clustering algorithm to group customers.
- **Optimal Cluster Determination:** Utilize the elbow method to determine the optimal number of clusters.
- **Dimensionality Reduction:** Employ Principal Component Analysis (PCA) for visualizing clusters in reduced dimensions.
- **Actionable Insights:** Translate clustering results into meaningful insights for targeted marketing.

## Repository Structure
```
.
├── Marketing_data.csv                  # Dataset file
├── main.py                             # Python Script 
├── Output/                             # Folder where all generated outputs are saved
│   ├── missing_data_heatmap.png        # Heatmap showing missing values
│   ├── distributions.png               # Combined distribution plots for all features
│   ├── correlations_heatmap.png        # Heatmap of feature correlations
│   ├── elbow_curve.png                 # Elbow curve for optimal cluster estimation
│   ├── histograms/                     # Folder containing individual histograms per feature & cluster
│   │   ├── histogram_<feature1>.png    
│   │   ├── histogram_<feature2>.png    
│   │   └── ...                        
│   ├── pca_scatter.png                 # PCA scatter plot with cluster coloring
│   └── output.txt                      # Log file with all printed outputs
└── README.md                           # This file
```

## How to Run
1. **Install Dependencies:**  
   Ensure you have Python installed (preferably 3.7 or later) and install the required packages using:
   ```
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```
2. **Dataset:**  
   Download the dataset from [Kaggle](https://www.kaggle.com/arjunbhasin2013/ccdata) and place the CSV file in the repository root, naming it `Marketing_data.csv`.

3. **Jupyter Notebook:**  
   Open `Unsupervised_ML_K_Means_Customer_Segmentation.ipynb` in Jupyter Notebook to view the complete analysis or run it interactively.

4. **Python Script:**  
   You can also run the Python script version of the analysis (if available) to generate all outputs automatically. All figures and log outputs will be saved to the **Output** folder.

## Outputs
All generated graphs, logs, and results are saved in the **Output** folder:
- **Heatmaps:** Missing data and correlation heatmaps.
- **Distributions & Histograms:** Overall distributions and separate histograms for each feature by cluster (stored under `Output/histograms/`).
- **Elbow Curve & PCA Scatter Plot:** Visualizations to help determine the number of clusters and display clusters in two dimensions.
- **Logs:** Detailed outputs and summaries are recorded in `Output/output.txt`.

## Conclusion
This project showcases the effective application of unsupervised machine learning techniques (k-means clustering and PCA) to segment customers. The insights derived from this analysis can help tailor marketing strategies to specific customer groups, thereby improving overall engagement and campaign performance.
