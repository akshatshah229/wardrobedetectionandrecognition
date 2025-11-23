import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.image as mpimg

warnings.filterwarnings('ignore')

def main():
    print("Loading Fashion MNIST data...")
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Use a smaller subset for testing purposes to speed up execution
    print("Subsetting data for quick testing (first 2000 samples)...")
    X_train = X_train[:2000]
    y_train = y_train[:2000]
    X_test = X_test[:500]
    y_test = y_test[:500]

    print("Preprocessing...")
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)

    # PCA
    print("Running PCA...")
    pca = PCA(svd_solver='randomized', n_components=150, whiten=True, random_state=0)
    
    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(random_state=0)
    model_lr = make_pipeline(pca, lr)
    model_lr.fit(X_train, y_train)
    y_test_hat_lr = model_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_test_hat_lr, normalize=True) * 100
    print(f"Accuracy: {acc_lr}")

    # Naive Bayes
    print("\n--- Naive Bayes ---")
    gnb = GaussianNB()
    model_nb = make_pipeline(pca, gnb)
    model_nb.fit(X_train, y_train)
    y_test_hat_nb = model_nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_test_hat_nb, normalize=True) * 100
    print(f"Accuracy: {acc_nb}")

    # SVM
    print("\n--- SVM ---")
    svc = SVC(kernel='rbf', class_weight='balanced')
    model_svc = make_pipeline(pca, svc)
    param_grid = {'svc__C': [1, 5, 10], 'svc__gamma': [0.001, 0.005]}
    cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True) # Reduced splits for speed
    grid = GridSearchCV(model_svc, param_grid, cv=cv, scoring='accuracy')
    grid.fit(X_train, y_train)
    bestModel_svc = grid.best_estimator_
    y_test_hat_svm = bestModel_svc.predict(X_test)
    acc_svm = accuracy_score(y_test, y_test_hat_svm, normalize=True) * 100
    print(f"Accuracy: {acc_svm}")
    
    # Decision Tree
    print("\n--- Decision Tree ---")
    dt = DecisionTreeClassifier()
    # PCA transform manually for GridSearch as in notebook (though pipeline is better usually)
    pca_dt = PCA(svd_solver='randomized', n_components=150, whiten=True, random_state=0)
    X_train_pca_dt = pca_dt.fit_transform(X_train)
    X_test_pca_dt = pca_dt.transform(X_test)
    
    param_grid_dt = {'max_depth': [10, 11, 12]}
    cv_dt = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
    grid_dt = GridSearchCV(dt, param_grid_dt, cv=cv_dt, scoring='accuracy', return_train_score=True)
    grid_dt.fit(X_train_pca_dt, y_train)
    bestModel_dt = grid_dt.best_estimator_
    y_test_hat_dt = bestModel_dt.predict(X_test_pca_dt)
    acc_dt = accuracy_score(y_test, y_test_hat_dt, normalize=True) * 100
    print(f"Accuracy: {acc_dt}")

    # Neural Network
    print("\n--- Neural Network ---")
    model_nn = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[50, 50])
    # Re-using PCA from DT section or fitting new one? Notebook fits new one.
    # Let's just use the fitted pca_dt for consistency with notebook logic which seemed to refit/transform
    model_nn.fit(X_train_pca_dt, y_train)
    y_test_hat_nn = model_nn.predict(X_test_pca_dt)
    acc_nn = accuracy_score(y_test, y_test_hat_nn, normalize=True) * 100
    print(f"Accuracy: {acc_nn}")

    # Validation on Custom Images
    print("\n--- Validation on Custom Images ---")
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    data_files = ['test_bag.png', 'test_coat.png', 'test_dress.png']
    
    X_data = []
    rgb_weights = [0.2989, 0.5870, 0.1140]
    
    for file in data_files:
        try:
            img = mpimg.imread(file)
            # Resize
            abc = cv2.resize(img, (28, 28))
            # Rotate (Notebook does this, so I will too)
            abc = cv2.rotate(abc, cv2.ROTATE_90_CLOCKWISE)
            # Grayscale
            # Check if image has 4 channels (RGBA) or 3 (RGB)
            if img.shape[2] == 4:
                grayscale_image = np.dot(abc[...,:3], rgb_weights)
            else:
                grayscale_image = np.dot(abc[...,:3], rgb_weights)
            X_data.append(grayscale_image)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return

    X_data = np.array(X_data)
    X_validate = X_data.reshape(3, 28*28)
    
    # New labels: Bag (8), Coat (4), Dress (3)
    y_validate = np.array([8, 4, 3]) 

    print("Predicting on custom images using best SVM model...")
    # SVM model in notebook was trained on PCA data. 
    # We need to transform X_validate using the same PCA.
    # In the notebook, for SVM: model_svc = make_pipeline(pca, svc). 
    # So we can just use bestModel_svc.predict(X_validate) IF bestModel_svc includes the PCA step.
    # In my code: grid = GridSearchCV(model_svc, ...). bestModel_svc is the pipeline.
    # So it should handle PCA automatically.
    
    y_validate_hat = bestModel_svc.predict(X_validate)
    
    for i in range(len(y_validate_hat)):
        print(f"{i+1}. Predicted: {class_names[y_validate_hat[i]]} (True: {class_names[y_validate[i]]})")
        
    acc_val = accuracy_score(y_validate, y_validate_hat, normalize=True) * 100
    print(f"Validation Accuracy: {acc_val}%")

if __name__ == "__main__":
    main()
