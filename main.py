import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
mail=pd.read_csv("spam.csv")
mail.drop_duplicates(inplace=True,subset='Message')
mail_total=mail["Category"].value_counts()

#plt.pie(mail_total, labels=["notspam","spam"],autopct="%1.1f%%")
#plt.show()
lb=LabelEncoder()
mail["Category"]=lb.fit_transform(mail["Category"])
print(mail.dtypes)

x=mail["Message"]
y=mail["Category"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

tf=TfidfVectorizer(lowercase=True,stop_words="english")

x_train_tf=tf.fit_transform(x_train)
x_test_tf=tf.transform(x_test)

lr=LinearSVC()

lr.fit(x_train_tf,y_train)

y_pre=lr.predict(x_test_tf)

print("report",classification_report(y_test,y_pre))