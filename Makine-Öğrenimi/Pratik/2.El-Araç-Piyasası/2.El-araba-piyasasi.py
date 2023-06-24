# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

# Veri hazırlama
dataFrame = pd.read_csv("ford.csv")
print(dataFrame.head())
print("*********************************")
dataFrame.drop(["model", "transmission", "fuelType"], axis=1, inplace=True) #model kolonunu attık
print(dataFrame)

print(dataFrame.describe())
print(dataFrame.corr())
print(dataFrame.isnull().sum())
#%%
#Veri analizi
sbn.displot(dataFrame["price"])

dataFrame.sort_values("price", ascending=False).head(10)
print(len(dataFrame) * 0.01) #veri setinin %1'ini bulduk ve bunu çıkartıcaz (179)

newDataFrame = dataFrame.sort_values("price", ascending=False).iloc[179:]
dataFrame = newDataFrame
dataFrame.sort_values("price", ascending=False).head(10)

sbn.displot(dataFrame["price"])

print(dataFrame.groupby("year").mean()["price"])
dataFrame = dataFrame[dataFrame.year != 2060]

#%%
#Veri işleme
# y = a*x + b

x = dataFrame.drop("price", axis=1).values
y = dataFrame.price.values
print(y)
print(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
print(x_train.shape) #12450 veri
print(x_test.shape) #5336 veri

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

print(model.intercept_)
print("year, mileage, tax, mpg, engineSize: ", model.coef_)

#%%
#Model Performansı (Eğitim) %73
y_head = model.predict(x_train)

from sklearn.metrics import mean_squared_error, r2_score
print("\nEğitim:")
print("MSE: ", mean_squared_error(y_train, y_head))
print("Başarı Skoru: ", r2_score(y_train, y_head)) 

#%%
#Model Performansı (Test) %72

y_head = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
print("\nTest: ")
print("MSE: ", mean_squared_error(y_test, y_head))
print("Başarı Skoru: ", r2_score(y_test, y_head))

plt.scatter(y_test, y_head)

#%%
#Polinom Regresyonu (Eğitim) %80

from sklearn.preprocessing import PolynomialFeatures
print("\nPolinom Regresyonu:")

polinomReg = PolynomialFeatures(degree=3)
x_polinom = polinomReg.fit_transform(x_train)

polinomModel = LinearRegression()
polinomModel.fit(x_polinom, y_train)
y_head = polinomModel.predict(x_polinom)

print("\nPolinom Regresyonu Eğitim:")
print("MSE: ", mean_squared_error(y_train, y_head))
print("Eğitim Skoru: ", r2_score(y_train, y_head))


#%%
#Polinom Regresyonu (Test) %55
x_polinom = polinomReg.fit_transform(x_test)
y_head = polinomModel.predict(x_polinom)

print("Polinom Regresyonu Test:")
print("MSE: ", mean_squared_error(y_test, y_head))
print("Test Skoru: ", r2_score(y_test, y_head))

