
###  `README.md`


# Task 8: Clustering with K-Means

##  Objective
The aim of this task is to perform **Unsupervised Learning** using **K-Means Clustering** on a customer segmentation dataset. The goal is to group customers into distinct clusters based on their purchasing behavior and demographics.


##  Dataset
**Name:** Mall Customer Segmentation Dataset  
**Columns:**
- `CustomerID` – Unique customer identifier
- `Gender` – Categorical (Male/Female)
- `Age` – Age of the customer
- `Annual Income (k$)` – Annual income in thousands
- `Spending Score (1-100)` – Score assigned based on purchasing behavior

---

##  Tools & Libraries Used
- Python 3.x
- `pandas` for data manipulation
- `numpy` for numerical operations
- `matplotlib` and `seaborn` for visualization
- `scikit-learn` for machine learning and clustering
- `PCA` for dimensionality reduction

---

##  Preprocessing Steps
1. Loaded the dataset and removed `CustomerID` (irrelevant for clustering).
2. Encoded the categorical column `Gender` using `LabelEncoder`.
3. Scaled the data using `StandardScaler` for better performance of K-Means.
4. Used `PCA` to project the data into 2D for visualization purposes.

---

##  K-Means Clustering Workflow

### 1. **Elbow Method**
Used to determine the optimal number of clusters (K) by plotting the **inertia** (within-cluster sum of squares) for K = 1 to 10.

### 2. **K-Means Clustering**
Applied K-Means with the optimal K (chosen = 5), using:
```python
KMeans(n_clusters=5, init='k-means++', random_state=42)
````

### 3. **Cluster Visualization**

Used PCA to reduce features to 2 dimensions and visualized the clusters using `Seaborn`.

### 4. **Evaluation: Silhouette Score**

Calculated **Silhouette Score** to evaluate how well the clusters are separated.
**Score Obtained:** *e.g., 0.553*

---

##  Results

* **Optimal Clusters (K):** 5
* **Silhouette Score:** (Varies based on scaling but usually > 0.5)
* **Cluster Insight:**

  * Groups formed based on age, income, and spending patterns.
  * Clear segmentation between low spenders, high spenders, and moderate consumers.

---

##  Files Included

| File                         | Description                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------ |
| `task8_kmeans_clustering.py` | Full Python code with preprocessing, clustering, evaluation, and visualization |
| `Mall_Customers.csv`         | Dataset used (to be added by user)                                             |
| `README.md`                  | This documentation file                                                        |

---

## 📚 Learnings

* Applied **unsupervised learning** using K-Means clustering.
* Learned to evaluate clusters using **Elbow Method** and **Silhouette Score**.
* Understood the impact of preprocessing on clustering performance.
* Practiced dimensionality reduction with **PCA** for 2D visualization.

---

##  How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python task8_kmeans_clustering.py
```

---

##  Reference Links

* [Mall Customer Dataset on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
* [Scikit-learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

---

Author: Sonai18-tech
