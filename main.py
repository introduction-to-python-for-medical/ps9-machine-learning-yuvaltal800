%load_ext autoreload
%autoreload 2

# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv

import pandas as pd
df = pd.read_csv('parkinsons.csv') 
df.head()

x=df[['MDVP:Fo(Hz)','MDVP:Flo(Hz)']]
y=df['status']
x.head()
y.head()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)

y_perd=model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_perd)



joblib.dump(model, 'my_model.joblib')
