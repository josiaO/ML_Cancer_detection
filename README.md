# ML_Cancer_detection. Breast Cancer Detection Project

This project uses machine learning to classify breast cancer tumors as malignant or benign based on 30 features. It includes a Jupyter notebook for training the model and a Streamlit app for user interaction.

---

## Project Overview

Breast cancer is one of the most common cancers worldwide. Early detection and accurate classification of tumors can significantly improve treatment outcomes. This project leverages machine learning to classify tumors based on their characteristics, using a dataset of 30 features.

---

## Project Structure

'''
Cancer_detection/
│
├── dataset/                # Folder for your dataset
│   └── data.csv            # Input dataset
│
├── src/                    # Source code folder
│   ├── cancer_classifier.ipynb  # Jupyter notebook for training the model
│   ├── cancer_app.py        # Streamlit app for user interaction
│   └── model.pkl            # Saved trained model
│
├── .gitignore              # File to exclude unnecessary files from Git
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
'''

---

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical data, and scales features.
- **Model Training**: Trains multiple machine learning models (Logistic Regression, Decision Tree, Random Forest) and evaluates their performance.
- **Streamlit App**: Provides an interactive interface for users to input tumor features and get predictions.

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/josiaO/Cancer_detection.git
cd Cancer_detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run cancer_app.py
```

---

## Dataset

The dataset used in this project is the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

### How to Add the Dataset

1. Download the dataset from the link above.
2. Place the `data.csv` file in the `dataset/` directory:

The dataset contains 30 features extracted from digitized images of breast mass. These features describe the characteristics of the cell nuclei present in the image. The target variable is the diagnosis (`M` for malignant, `B` for benign).

---

## Dependencies

The project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `streamlit`

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Model Details

The following models were trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

The Random Forest model achieved the highest accuracy and was saved as `model.pkl` for deployment.

---

## Results

- **Accuracy**: The Random Forest model achieved an accuracy of over 95% on the test set.
- **Classification Report**: Detailed precision, recall, and F1-score metrics are available in the Jupyter notebook.

---

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request.

## Acknowledgments

- The dataset used in this project is publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).
- Special thanks to the open-source community for providing the tools and libraries used in this project.
