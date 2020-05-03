import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, RidgeClassifier, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

boston = pd.read_csv("data/housing.csv", sep=",", encoding="utf-8")
boston['MEDV'] = boston['MEDV']/1000

# correlation_matrix = boston.corr().round(2)
# sns.heatmap(data=correlation_matrix, vmax=.3, center=0,
#                     square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
# plt.show()
#
# for column in ['LSTAT' ,'RM', 'PTRATIO']:
#     x = boston[column]
#     y = boston['MEDV']
#     plt.scatter(x, y, marker='o', edgecolors='black', color='red')
#     plt.title('Correlation between PTRATIO and MEDV (Ratio of pupil-teacher in the area/median value of owner-occupied homes)')
#     plt.xlabel('Pupil-Teacher ratio in area')
#     plt.ylabel('Housevalue per 1000$')
#     plt.show()

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM'], boston['PTRATIO']], columns = ['LSTAT', 'RM', 'PTRATIO'])
Y = boston['MEDV']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

algorithms = [
    LinearRegression(),
    Lasso(alpha=0.1)
]
for algorithm in algorithms:
    regression_model = algorithm
    regression_model.fit(X_train, Y_train)

    # model evaluation for training set
    # y_train_predict = regression_model.predict(X_train)
    # mse = np.sqrt(mean_squared_error(Y_train, y_train_predict))
    # mae = mean_absolute_error(Y_train, y_train_predict)
    # print("The model performance for training set")
    # print("--------------------------------------")
    # print('MSE is {}'.format(mse))
    # print('MAE score is {}'.format(mae))
    # print("\n")

    # model evaluation for testing set
    y_test_predict = regression_model.predict(X_test)
    mse = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    mae = mean_absolute_error(Y_test, y_test_predict)

    print("The model performance for testing set")
    print("--------------------------------------")
    print('MSE is {}'.format(mse))
    print('MAE score is {}'.format(mae))
