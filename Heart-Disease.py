import numpy as np                   
import pandas as pd                  
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('heart.csv')

df.head()
df.info()
df.describe()
df.shape
df.isnull().sum()
df.duplicated().sum()
df['HeartDisease'].value_counts().plot(kind='bar')
df.isnull().sum()
col = ["Age","RestingBP","Cholesterol","MaxHR"]
for c,i in enumerate(col,1):
    plt.subplot(2,2,c)
    sns.histplot(x = df[i])
plt.tight_layout()
plt.show()
df['Cholesterol'].value_counts()
ch_mean = df.loc[df['Cholesterol'] != 0,"Cholesterol"].mean()
ch_mean
df['Cholesterol'] = df['Cholesterol'].replace(0,ch_mean)
df['Cholesterol'] = df['Cholesterol'].round(2)
ch_mean = df.loc[df['RestingBP'] != 0,"RestingBP"].mean()
df['RestingBP'] = df['RestingBP'].replace(0,ch_mean)
df['RestingBP'] = df['RestingBP'].round(2)
col = ["Age","RestingBP","Cholesterol","MaxHR"]
for c,i in enumerate(col,1):
    plt.subplot(2,2,c)
    sns.histplot(x = df[i])
plt.tight_layout()
plt.show()
sns.countplot(x=df["Sex"],hue=df["HeartDisease"])
sns.countplot(x=df["ChestPainType"],hue=df["HeartDisease"])
sns.countplot(x=df["FastingBS"],hue=df["HeartDisease"])
sns.boxplot(x="HeartDisease",y="Cholesterol",data=df)
sns.violinplot(x="HeartDisease",y="Age",data=df)
sns.heatmap(df.corr(numeric_only=True),annot=True)
df_encode = pd.get_dummies(df,drop_first=True)
df_encode
df_encode = df_encode.astype(int)
df_encode
df_encode.columns
l_encode = df.copy()
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
col = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
for i in col:
    l_encode[i] =encode.fit_transform(l_encode[i]) 
l_encode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
X = df_encode.drop('HeartDisease',axis =1)
y = df_encode['HeartDisease']
X_train,X_test,y_train,y_test = 
train_test_split(X,y,test_size=0.20,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test,y_pred)
acc
f1 = f1_score(y_test,y_pred)
f1
model = KNeighborsClassifier()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test,y_pred)
acc
f1 = f1_score(y_test,y_pred)
f1
model = GaussianNB()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test,y_pred)
acc
f1 = f1_score(y_test,y_pred)
f1
model = DecisionTreeClassifier()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test,y_pred)
acc
f1 = f1_score(y_test,y_pred)
f1
model = SVC()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test,y_pred)
acc
f1 = f1_score(y_test,y_pred)
f1
models = {
    "Logistic Regression" : LogisticRegression(),
    "KNN" : KNeighborsClassifier(),
    "Naive Byeas" : GaussianNB(),
    "Decision tree" :DecisionTreeClassifier(),
    "SVM" : SVC()
}
result = []
for name,model in models.items():
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    result.append({
        'model' : name,
        'Accuracy': round(acc,4),
        'f1_score':round(f1,4)
    })
result
output:
[{'model': 'Logistic Regression', 'Accuracy': 0.8696, 'f1_score': 0.8857},
 {'model': 'KNN', 'Accuracy': 0.8641, 'f1_score': 0.8815},
 {'model': 'Naive Byeas', 'Accuracy': 0.8533, 'f1_score': 0.8683},
 {'model': 'Decision tree', 'Accuracy': 0.7935, 'f1_score': 0.8155},
 {'model': 'SVM', 'Accuracy': 0.8478, 'f1_score': 0.8679}]
