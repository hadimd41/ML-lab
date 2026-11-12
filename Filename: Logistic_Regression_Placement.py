
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = np.array([0, 1, 2, 3, 4, 5, 1, 2, 3, 4]).reshape(-1, 1)
Y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, Y)

Y_pred = model.predict(X)

print("Predicted Values:", Y_pred)
print("Actual Values:", Y)
print("Accuracy:", accuracy_score(Y, Y_pred))
print("Confusion Matrix:")
print(confusion_matrix(Y, Y_pred))
print("Classification Report:")
print(classification_report(Y, Y_pred))

plt.scatter(X, Y, color='red')
plt.plot(X, model.predict_proba(X)[:, 1], color='blue')
plt.xlabel("Practice Tests Completed")
plt.ylabel("Placement Eligibility")
plt.title("Logistic Regression")
plt.show()
