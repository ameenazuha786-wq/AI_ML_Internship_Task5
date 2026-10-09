# AI/ML Internship Task 5: Decision Trees and Random Forests

## Objective
To learn tree-based machine learning models for classification and compare the performance of Decision Tree and Random Forest classifiers.

## Dataset
The Heart Disease dataset (`heart.csv`) is used to predict the target class based on health-related features.

## Tools and Technologies
- Python
- Pandas
- Scikit-learn
- Matplotlib
- VS Code

## Tasks Performed
1. Loaded and explored the dataset.
2. Separated the input features and target variable.
3. Split the dataset into training and testing sets (80:20).
4. Trained a Decision Tree Classifier.
5. Controlled overfitting by limiting tree depth.
6. Visualized the Decision Tree.
7. Trained a Random Forest Classifier.
8. Compared model accuracy.
9. Analyzed feature importance.
10. Evaluated both models using 5-fold cross-validation.

## Results

### Test Accuracy
- Decision Tree: 98.54%
- Random Forest: 100.00%

### Cross-Validation Accuracy
- Decision Tree: 100.00%
- Random Forest: 99.71%

### Important Features
The Random Forest feature-importance analysis identified features including `oldpeak`, `thal`, `age`, `chol`, and `exang`.

## Project Files
- `task5.py` – Python source code.
- `heart.csv` – Dataset.
- `decision_tree.png` – Decision Tree visualization.
- `feature_importance.png` – Feature importance chart.

## Conclusion
The Decision Tree and Random Forest classifiers were trained and evaluated on the Heart Disease dataset. Random Forest achieved 100.00% test accuracy, while cross-validation was used to compare model performance across multiple data splits. These results apply to this dataset and evaluation setup and do not guarantee real-world prediction performance.
