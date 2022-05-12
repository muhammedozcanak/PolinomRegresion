# PolinomRegresion


import matplotlib.pyplot as plt
import pandas as pd

polinom = pd.read_excel('polinom.xlsx')

x=polinom.iloc[:,1:2]
y=polinom.iloc[:,2:]
X=x.values
Y=y.values


from sklearn.linear_model import LinearRegression


from sklearn.preprocessing import PolynomialFeatures
derece = PolynomialFeatures(degree = 3)
x_derece = derece.fit_transform(X)
linear = LinearRegression() 
linear.fit(x_derece,y)
plt.scatter(X,Y)
plt.plot(X,linear.predict(derece.fit_transform(X)), color='green')
plt.show()
pred=(linear.predict(derece.fit_transform(X)))

print(linear.predict(derece.fit_transform([[5.5]]))) 
print(linear.predict(derece.fit_transform([[11]]))) 
print(linear.predict(derece.fit_transform([[13]]))) 

from sklearn.metrics import r2_score

print("Başarı Oranı:")
print(r2_score(Y,pred))

![Ekran Alıntısı](https://user-images.githubusercontent.com/99281922/168009712-db1524ac-0895-4e9f-aaf0-3493ee377b20.PNG)
