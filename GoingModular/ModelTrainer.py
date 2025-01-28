import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# ModelTrainer Class
class ModelTrainer:
  def __init__(self,
           df: pd.DataFrame):
    
    self.df = df

  def plot_confusion_matrix(self,
                            y_true,
                            y_pred,
                            labels,
                            model_name):
    """
    Plot confusion matrix for a given model.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.show()

  def train_and_evaluate(self,
                         model,
                         X_train,
                         X_test,
                         y_train,
                         y_test,
                         model_name,
                         labels):
    """
    Train the model and evaluate its performance.
    """
    print(f"Training {model_name}...\n")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    self.plot_confusion_matrix(y_test, y_pred, labels, model_name)

  def ModelPreparing(self):
    """
    Train and evaluate machine learning models.
    """
    print("Training models...")
    # Prepare the dataset
    X = np.array(self.df["Word2Vec_Sentence_wv"].tolist())
    y = self.df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

    # Labels for confusion matrix
    labels = ['neutral', 'hate', 'offensive']

    # Train and evaluate XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    self.train_and_evaluate(xgb_model, X_train, X_test, y_train, y_test, "XGBoost", labels)

    # Train and evaluate SVM
    svm_model = SVC(kernel="linear", random_state=42)
    self.train_and_evaluate(svm_model, X_train, X_test, y_train, y_test, "SVM", labels)
