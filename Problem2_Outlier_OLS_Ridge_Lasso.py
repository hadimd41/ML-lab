
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Problem2_Outlier_OLS_Ridge_Lasso.csv')

X = data[['X']]
Y = data['Y']

ols = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)

models = {'OLS': ols, 'Ridge': ridge, 'Lasso': lasso}

for name, model in models.items():
    model.fit(X, Y)
    Y_pred = model.predict(X)

    print(name)
    print("Coefficient:", model.coef_)
    print("Intercept:", model.intercept_)
    print("MSE:", mean_squared_error(Y, Y_pred))
    print("R2 Score:", r2_score(Y, Y_pred))
    print()

    plt.plot(X, Y_pred, label=name)

plt.scatter(X, Y, color='red', label='Actual Data')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("OLS, Ridge and Lasso with Outlier")
plt.legend()
plt.show()
