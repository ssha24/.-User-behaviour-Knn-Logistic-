import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lb=LabelEncoder()

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=11)
df=pd.read_csv("user_behavior_dataset.csv")
df["Device Model"]=lb.fit_transform(df["Device Model"])
df["Operating System"]=lb.fit_transform(df["Operating System"])
df["Gender"]=lb.fit_transform(df["Gender"])
x=df[["Device Model","Operating System","App Usage Time (min/day)","Screen On Time (hours/day)","Battery Drain (mAh/day)","Number of Apps Installed","Data Usage (MB/day)","Age","Gender"]]
y=df["User Behavior Class"]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df)
#train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
def y_pred():
    lr.fit (x_train,y_train)
    y_pred=lr.predict(x_test)
    r2=r2_score(y_test,y_pred)
    print(r2)
y_pred()
def k_n ():
    kn.fit(x_train,y_train)
    u=kn.predict(x_test)
    r2=r2_score(y_test,u)
    print(r2)
k_n()