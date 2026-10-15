"""
DecodeLabs AI Project 2
Data Classification Using AI

This project trains and evaluates a supervised machine-learning classifier
using the Iris dataset available in scikit-learn.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # 1. Load and understand the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("=" * 60)
    print("DecodeLabs | Project 2: Data Classification Using AI")
    print("=" * 60)
    print(f"Dataset: Iris")
    print(f"Number of rows: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {', '.join(iris.feature_names)}")
    print(f"Classes: {', '.join(iris.target_names)}")
    print("\nFirst 5 feature rows:")
    print(X[:5])
    print("\nClass counts:")
    for class_id, class_name in enumerate(iris.target_names):
        print(f"  {class_name}: {(y == class_id).sum()}")

    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 3. Train a simple classification algorithm
    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X_train, y_train)

    # 4. Evaluate the model on unseen test data
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel: Decision Tree Classifier")
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, target_names=iris.target_names, zero_division=0
    ))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions))

    # 5. Try a new example (measurements in cm)
    new_flower = [[5.1, 3.5, 1.4, 0.2]]
    predicted_class = model.predict(new_flower)[0]
    print("\nExample prediction:")
    print(f"Measurements: {new_flower[0]}")
    print(f"Predicted flower class: {iris.target_names[predicted_class]}")


if __name__ == "__main__":
    main()
