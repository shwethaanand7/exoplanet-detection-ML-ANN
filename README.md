# Exoplanet Detection using ML and ANN

This project uses both **Machine Learning (ML)** algorithms and an **Artificial Neural Network (ANN)** to classify whether a star system contains an exoplanet (`LABEL` 1) or not (`LABEL` 0). The dataset includes flux values collected from stellar light curves, which are analyzed and modeled to make accurate predictions.

---

## Libraries Used

- **Pandas**: Data manipulation and analysis  
- **NumPy**: Numerical computing  
- **Matplotlib**: Data visualization  
- **Seaborn**: Statistical data visualization  
- **Scikit-learn**: Machine learning algorithms and evaluation metrics  
- **Imbalanced-learn**: Handling class imbalance with SMOTE  
- **TensorFlow** & **Keras**: Building and training the ANN  
- **SciPy**: Gaussian filtering for data preprocessing  

---

## Data

- **`exoTrain.csv`**: Training dataset containing flux values and labels (`LABEL` column).  
- **`exoTest.csv`**: Testing dataset used to evaluate model performance.

---

## Workflow

### 1. Data Preprocessing
- Convert the `LABEL` column to binary values: `0` (Not Exoplanet) and `1` (Exoplanet).  
- Visualize class distribution and handle missing values.  
- Normalize flux values and apply Gaussian filtering for noise reduction.  
- Remove outliers using statistical methods.  

### 2. Handling Class Imbalance
- Apply **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset, ensuring the models learn effectively from both classes.

---

## Machine Learning Models

### Algorithms Implemented
1. **K-Nearest Neighbors (K-NN)**: Classifies based on proximity of data points.  
2. **Logistic Regression**: Linear model for binary classification.  
3. **Decision Tree**: Tree-based classification algorithm.

### Evaluation Metrics
- **Accuracy**, **Precision**, **Recall**, **F1-score**  
- **Confusion Matrix** and **Receiver Operating Characteristic (ROC)** curves  
- **Area Under Curve (AUC)** for evaluating model performance  

---

## Artificial Neural Network (ANN)

### ANN Model Overview
- The ANN is built using **Keras** with:  
  - Two hidden layers, each with 4 neurons and **ReLU** activation.  
  - An output layer using **Sigmoid** activation for binary classification.  

### ANN Workflow
1. **Data Preprocessing**: Same steps as ML models with added **dimensionality reduction** using PCA (90% variance retained).  
2. **Resampling**: Use SMOTE to balance the dataset.  
3. **Training**: Train the ANN using the Adam optimizer and binary cross-entropy loss.  
4. **Cross-Validation**: Evaluate using 5-fold cross-validation to compute mean accuracy and variance.

---

## Results

- Multiple ML models and the ANN were evaluated using the metrics above.  
- The **ANN outperformed ML models** in classifying exoplanets, showcasing the effectiveness of deep learning for this task.  

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/exoplanet-classification.git
   cd exoplanet-classification
