# -*- coding:utf-8 -*-
#Veri ön izlenimi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataframe = pd.read_excel("Ev-satis.xlsx")
print(dataframe)

print(dataframe.isnull().sum())
print(dataframe.describe())

#%%
#Veri Analizi

#Korelasyon
plt.figure(figsize=(12,10))
cor = dataframe.corr() #sütunların ikili korelasyonlarını hesaplar
sbn.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#dağılım grafiği
sbn.pairplot(dataframe)
sbn.displot(dataframe["Fiyat"])

#%%
# Veri setlerini oluşturma

x = dataframe[["Oda_Sayısı", "Net_m2", "Katı", "Yaşı"]].values
y = dataframe["Fiyat"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

#y = a*x + a*x2 + ... + b
print("\nModel Bias Değeri: ", model.intercept_) #bias (b)

#özellik katsayısı (a)
print("Oda Sayısı, Net m2, Kat, Yaş Katsayıları: ", model.coef_, "\n")

#%%
# Verileri tahmine etme ve doğruluk skoru  (Eğitim)

y_head = model.predict(x_train)

print("\n*****EĞİTİM TAHMİNLERİ*****")
for i, prediction in enumerate(y_head):
    
    print("Tahmin Edilen Fiyat: %.2f, Gerçek Fiyat: %s" % (prediction, y[i]))


from sklearn.metrics import r2_score
r2_score(y_train, y_head)
#0.73 [%73]

#gerçek ve tahmin fiyatlarının dağılım grafiği
plt.scatter(y_train, y_head)
plt.xlabel("Gerçek fiyatlar")
plt.ylabel("Tahmini Fiyatlar")

#%%
# Modeli test etme

y_head_test = model.predict(x_test)

print("\n*****TEST TAHMİNLERİ*****")
for i, prediction in enumerate(y_head_test):
    
    print("Tahmin Edilen Fiyat: %.2f, Gerçek Fiyat: %s" % (prediction, y[i]))
    
r2_score(y_test, y_head_test)
#0.69 (%69)

plt.scatter(y_test, y_head_test)
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmini Fiyat")

oda_sayisi = 3
net_m2 = 105
kat = 4
yas = 8

print("\nYeni Evin Fiyatı: ₺", model.predict([[oda_sayisi, net_m2, kat, yas]]))
#770.25 
