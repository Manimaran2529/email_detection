import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
mail=pd.read_csv("email_spam.csv")
mail.drop_duplicates(inplace=True)
#mail_total=mail["type"].value_counts()
#plt.pie(mail_total,autopct="%1.1f%%")
#plt.show()

mail["type"]=mail["type"].map({"spam":0,"not spam":1})
#print(mail.dtypes)
mail["combined_text"] = mail["title"] + " " + mail["text"]



x=mail["combined_text"]
y=mail["type"]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

tf_x=TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1, 2))


X_train_tfidf=tf_x.fit_transform(X_train)
X_test_tfidf=tf_x.transform(X_test)

lr=LogisticRegression(max_iter=1000,
                       class_weight="balanced")
lr.fit(X_train_tfidf,y_train)

y_predict=lr.predict(X_test_tfidf)

print("score",classification_report(y_test,y_predict))

a = input("Enter your mail: ")

b = tf_x.transform([a])
c_predict = lr.predict(b)

if c_predict[0] == 0:
    print("🚨 This mail is SPAM")
else:
    print("✅ This mail is NOT SPAM")
