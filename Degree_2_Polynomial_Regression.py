import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([-2, -1, 0, 1, 2]).reshape(-1, 1)
Y = np.array([9, 2, 1, 6, 17])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, Y)

Y_pred = model.predict(X_poly)

print("Predicted values:", Y_pred)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(X, Y, color='red')
plt.plot(X, Y_pred, color='blue')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Degree 2 Polynomial Regression")
plt.show()
