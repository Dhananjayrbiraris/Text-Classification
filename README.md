
# Text Classification Model using Naive Bayes

This project demonstrates a basic text classification pipeline using a Naive Bayes classifier. The pipeline performs data loading, preprocessing, vectorization, model training, and evaluation.

## Features:
- Load data from different file formats (`CSV`, `Excel`, `JSON`, `PDF`).
- Preprocess and prepare data for classification.
- Vectorize text data using `CountVectorizer`.
- Train a Naive Bayes classification model (`MultinomialNB`).
- Evaluate the model with accuracy and detailed classification report.

## Dependencies:
This project uses the following libraries:
- `pandas`
- `numpy`
- `sklearn`
- `pdfplumber` (for PDF file processing)

To install the required dependencies, run:
```bash
pip install pandas numpy scikit-learn pdfplumber
```

## Code Structure:
### 1. **Load and Explore Data**:
   - The `load_data(filepath)` function loads data from various formats, including CSV, Excel, JSON, and PDF.
   - For PDF files, the text is extracted using the `pdfplumber` library.

### 2. **Preprocess Data**:
   - The `preprocess_data(data)` function allows the user to specify which columns contain the text and label data.
   - It handles the separation of features (`X`) and labels (`y`) for the classification task.

### 3. **Vectorize Text Data**:
   - The `vectorize_text(X_train, X_test)` function uses the `CountVectorizer` from `sklearn` to convert text data into numerical vectors (Bag of Words model).
   - This transformation is applied to both training and testing data.

### 4. **Train Model**:
   - The `train_model(X_train_vec, y_train)` function trains a Naive Bayes classifier (`MultinomialNB`) on the vectorized training data.

### 5. **Evaluate Model**:
   - The `evaluate_model(model, X_test_vec, y_test)` function predicts the labels for the test data and computes the accuracy score along with a classification report (precision, recall, F1-score).

## How to Use:
1. Run the Python script.
2. When prompted, provide the file path for the dataset.
   - Supported formats: `.csv`, `.xlsx`, `.xls`, `.json`, `.pdf`.
3. The script will guide you through selecting the text and label columns in the dataset.
4. The model will train on the data and output the accuracy score and a detailed classification report.

### Example Output:
```bash
Please enter the file path for the dataset: data.csv
Data loaded successfully.
Available columns in the dataset: ['text', 'label']
Enter the column name for text data: text
Enter the column name for labels: label
Accuracy: 0.85
Classification Report:
               precision    recall  f1-score   support

        class_1       0.87      0.90      0.88       200
        class_2       0.83      0.79      0.81       150

    accuracy                           0.85       350
   macro avg       0.85      0.84      0.84       350
weighted avg       0.85      0.85      0.85       350
```

## Troubleshooting:
- **Unsupported file format**: If the script cannot recognize your file format, make sure it is one of the supported formats: `.csv`, `.xlsx`, `.xls`, `.json`, or `.pdf`.
- **Column errors**: Ensure that the column names you provide match those in your dataset. If the data format changes, you might need to modify the column extraction logic accordingly.

## Notes:
- This is a basic text classification pipeline. You can expand it to include additional preprocessing steps (e.g., stopword removal, stemming, etc.) and different models.
- The model currently uses the `MultinomialNB` classifier, which is particularly suited for text classification tasks. You can try experimenting with other classifiers from the `sklearn` library for different results.

---
