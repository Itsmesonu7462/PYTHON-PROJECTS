import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv('Womens_Clothing_Reviews.csv')


data = data.dropna(subset=['Review Text', 'Rating'])


data['Sentiment'] = data['Rating'].apply(lambda x: 1 if x >= 4 else 0)


vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['Review Text'])
y = data['Sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
