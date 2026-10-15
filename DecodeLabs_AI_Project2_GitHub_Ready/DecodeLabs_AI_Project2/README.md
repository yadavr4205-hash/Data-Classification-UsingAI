# DecodeLabs AI Project 2: Data Classification Using AI

A beginner-friendly supervised machine-learning project that loads a dataset, explores its structure, splits it into training and testing sets, trains a classification algorithm, and evaluates its predictions.

## Project objective

Build a basic classification model using a small dataset, as required by the DecodeLabs Project 2 brief.

## Dataset

This project uses the **Iris dataset** bundled with scikit-learn. It contains 150 flower samples, four numeric measurements, and three flower classes:

- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)

The dataset is loaded directly by the code, so no separate download is required.

## Algorithm

A **Decision Tree Classifier** is trained on 80% of the dataset. The remaining 20% is held out for testing. The split is stratified so each flower class is represented proportionally in both sets.

## Project structure

```text
DecodeLabs_AI_Project2/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup and run

Python 3.9 or newer is recommended.

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. (Recommended) Create and activate a virtual environment:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install the dependency:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the project:
   ```bash
   python main.py
   ```

## What the program does

1. Loads and prints basic information about the Iris dataset.
2. Displays the feature names, class counts, and a few sample rows.
3. Splits the data into training and test sets.
4. Trains a Decision Tree classifier.
5. Prints test accuracy, a classification report, and a confusion matrix.
6. Predicts the class of an example flower.

The exact evaluation output can vary with library versions. The program prints the actual metrics when run; no result is hard-coded.

## Concepts demonstrated

- Dataset loading and basic exploration
- Features (`X`) and labels (`y`)
- Supervised learning
- Train/test split
- Classification model training
- Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix
- Prediction on a new example

## Limitations

The Iris dataset is small and clean, so its performance may not represent results on messy real-world data. A test score is an estimate, not a guarantee of future accuracy.

## GitHub upload

Create a new empty repository on GitHub, then run these commands from this project folder (replace the URL with your own repository URL):

```bash
git init
git add .
git commit -m "Complete DecodeLabs AI Project 2"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

If Git asks you to sign in, complete GitHub's authentication prompt. Never put passwords or access tokens in this repository.
