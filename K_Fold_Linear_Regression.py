
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

X = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).reshape(-1, 1)
Y = np.array([35, 40, 43, 48, 52, 57, 61, 66, 70, 74])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = LinearRegression()
errors = []

for train, test in kf.split(X):
    X_train, X_test = X[train], X[test]
    Y_train, Y_test = Y[train], Y[test]

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    mse = mean_squared_error(Y_test, Y_pred)
    errors.append(mse)

    print("Actual Values:", Y_test)
    print("Predicted Values:", Y_pred)
    print("Mean Squared Error:", mse)
    print()

print("Average MSE:", np.mean(errors))
