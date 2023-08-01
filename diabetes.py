import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('C:\\Users\\furka\\Downloads\\diabetes.csv')
data.head()
y=data.Outcome.values
x_ham_veri=data.drop(["Outcome"],axis=1)
x= (x_ham_veri-np.min(x_ham_veri))/(np.max(x_ham_veri)-np.min(x_ham_veri)) #normalizasyon işlemi


print("Normalizasyon öncesi ham veriler")
print(x_ham_veri.head())
print("Normalizasyon sonrası ham veriler")
print(x.head())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=1) #train ile test datası ayrılıyor
#knn modeli oluşturuluyor
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction= knn.predict(x_test)
print("k=3 için test verilerimizin doğrulama testi sonucu",knn.score(x_test,y_test)*100)
sayac=1
for k in range(1,11):
    knn_yeni= KNeighborsClassifier(n_neighbors=k)
    knn_yeni.fit(x_train,y_train)
    print(sayac," ", "Doğruluk oranı %",knn_yeni.score(x_test,y_test)*100)
    sayac+=1
# Yeni bir hasta tahmini için:
from sklearn.preprocessing import MinMaxScaler

# normalization yapıyoruz - daha hızlı normalization yapabilmek için MinMax  scaler kullandık...
sc = MinMaxScaler()
sc.fit_transform(x_ham_veri)

new_prediction = knn.predict(sc.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
new_prediction[0]