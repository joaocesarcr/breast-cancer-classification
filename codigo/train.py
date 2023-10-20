from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from confusionmatrix import confusionMatrix
#input: dados de treino/teste 
#return: retorna uma confusionMatrix (classe definida anteriormente) para cada um dos modelos utilizados (knn, svm, logistic regression e random forest).
def train(X_train, y_train, X_test, y_test):
  scaler = StandardScaler()
  X_test_scaled = scaler.fit_transform(X_test)
  X_train_scaled = scaler.fit_transform(X_train)

  # KNN
  knn_classifier = KNeighborsClassifier(n_neighbors=int(sqrt(len(X_train))))
  knn_classifier.fit(X_train_scaled, y_train)
  knn_predictions = knn_classifier.predict(X_test_scaled)
  knn = confusionMatrix(y_test, knn_predictions)

  # Logistic Regression
  logreg_classifier = LogisticRegression()
  logreg_classifier.fit(X_train_scaled, y_train)
  logreg_predictions = logreg_classifier.predict(X_test_scaled)
  logreg = confusionMatrix(y_test, logreg_predictions)

  # SVM
  svm_classifier = SVC(kernel='linear')
  svm_classifier.fit(X_train_scaled, y_train)
  svm_predictions = svm_classifier.predict(X_test_scaled)
  svm = confusionMatrix(y_test, svm_predictions)

  # Random Forest
  rf_classifier = RandomForestClassifier(n_estimators=100)
  rf_classifier.fit(X_train_scaled, y_train)
  rf_predictions = rf_classifier.predict(X_test_scaled)
  rf = confusionMatrix(y_test, rf_predictions)

  return [knn, logreg, svm, rf]
