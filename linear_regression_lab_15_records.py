
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).reshape(-1, 1)
Y = np.array([35, 38, 42, 45, 49, 52, 56, 60, 63, 67, 70, 74, 78, 81, 85])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Predicted Marks:", Y_pred)
print("Actual Marks:", Y_test)
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
print("R2 Score:", r2_score(Y_test, Y_pred))

plt.scatter(X, Y, color='red')
plt.plot(X, model.predict(X), color='blue')
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression - 15 Records")
plt.show()
