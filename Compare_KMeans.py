from Linear_kmeans import KMeans
import Normal_kmeans as nk
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('wine.csv',header = None)
header=['class','Alcohol',' Malic acid', 'Ash', 'Alcalinity', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines','Proline']
df.columns = header
X = df[df.columns[df.columns!='class']].values
y = df['class'].values
scaler = StandardScaler()
X_new = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_new,y,test_size = 0.2, random_state = 12)

print("Training Linear Kmeans")
start_time = time.time()
kmeans = KMeans(k=3)
kmeans.fit(X_train)
print("Time Taken for Linear Kmeans:", time.time() - start_time)

print("Training Normal Kmeans")
start_time = time.time()
n_kmeans = nk.KMeans(k=3) 
n_kmeans.fit(X_train)
print("Time Taken by Normal Kmeans:", time.time() - start_time)
