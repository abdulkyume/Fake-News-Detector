import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import itertools
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
from sklearn import metrics
from nltk import tokenize


nltk.download('stopwords')
print('Stop Words which will be used are : ', stopwords.words('english'))
print('')

dataset = pd.read_csv('Dataset/train.csv')
print('Dataset: ')
print(dataset)
print('')

dataset = dataset.drop(['id'], axis=1)

print('Dataset\'s Rows and Columns : ', dataset.shape)
print('')

print('Total Count of Empty cells: ')
print(dataset.isnull().sum())
print('')

dataset.fillna('')

print('Showing Fake and Real News Count Plot')
dataset.groupby(['label'])['title'].count().plot(kind="bar")
plt.title(label='Number of Fake and Real News', fontweight=10, pad='2.0')
plt.xticks([0, 1], ['True', 'False'])
plt.xlabel('News')
plt.ylabel('Total Count')
plt.show()

dataset['content'] = dataset['author'] + ' ' + dataset['title']
print('Merged Dataset: ')
print(dataset['content'])
print('')

print('The Stemming Process Have Started...')
stemm = PorterStemmer()


def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', str(content))
    content = content.lower()
    content = content.split()
    content = [stemm.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content


dataset['content'] = dataset['content'].apply(stemming)
print('Stemming Completed!!!')
print('Dataset After Stemming: ')
print(dataset['content'])

token_space = tokenize.WhitespaceTokenizer()


def counter(text, column_text, quantity):
    allWords = ' '.join([str(text) for text in text[column_text]])
    token_phrase = token_space.tokenize(allWords)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()), "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns="Frequency", n=quantity)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_frequency, x="Word", y="Frequency", color='blue')
    ax.set(xlabel="Words")
    ax.set(ylabel="Count")
    plt.xticks(rotation='vertical')
    plt.show()


print("Showing False News Words\nWait A Moment...")
counter(dataset[dataset["label"] == 1], "text", 30)

print("Showing Real News Words\nWait A Moment...")
counter(dataset[dataset["label"] == 0], "text", 30)

print('Showing Fake News Words \nWait For a Moment...')
fake_data = dataset[dataset["label"] == 1]
allWords = ' '.join([str(text) for text in fake_data.text])
wordcloud = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(allWords)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
print('closed\n')

print('Showing Real News Words\nWait For a Moment...')
fake_data = dataset[dataset["label"] == 0]
allWords = ' '.join([str(text) for text in fake_data.text])
wordcloud = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(allWords)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
print('closed\n')

X = dataset['content'].values
print('Content Values: ')
print(X)
print('')

Y = dataset['label'].values
print('Output Values: ')
print(Y)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# CountVectorizer & Logistic Regression
print('CountVectorizer & Logistic Regression: \n')
pipe = Pipeline([('cvec', CountVectorizer()), ('lr', LogisticRegression(solver='liblinear'))])
pipe_params = {'cvec__ngram_range': [(1, 1), (2, 2), (1, 3)], 'lr__C': [0.01, 1]}
gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
gs.fit(X_train, Y_train)
print("Best score:", gs.best_score_)
print("Train score", gs.score(X_train, Y_train))
print("Test score", gs.score(X_test, Y_test))
prediction = gs.predict(X_test)
cm = metrics.confusion_matrix(Y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
print('')

# TfidfVectorize & Logistic Regression
print('TfidfVectorize & Logistic Regression: \n')
pipe = Pipeline([('tvect', TfidfVectorizer()), ('lr', LogisticRegression(solver='liblinear'))])
pipe_params = {'tvect__max_df': [.75, .98, 1.0], 'tvect__min_df': [2, 3, 5], 'tvect__ngram_range': [(1, 1), (1, 2), (1, 3)], 'lr__C': [1]}
gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
gs.fit(X_train, Y_train)
print("Best score:", gs.best_score_)
print("Train score", gs.score(X_train, Y_train))
print("Test score", gs.score(X_test, Y_test))
prediction = gs.predict(X_test)
cm = metrics.confusion_matrix(Y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
print('')

# CountVectorizer & MultinomialNB
print('CountVectorizer & MultinomialNB: \n')
pipe = Pipeline([('cvec', CountVectorizer()), ('nb', MultinomialNB())])
pipe_params = {'cvec__ngram_range': [(1, 1), (1, 3)], 'nb__alpha': [.36, .6]}
gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
gs.fit(X_train, Y_train)
print("Best score:", gs.best_score_)
print("Train score", gs.score(X_train, Y_train))
print("Test score", gs.score(X_test, Y_test))
prediction = gs.predict(X_test)
cm = metrics.confusion_matrix(Y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
print('')

# TfidfVectorizer & MultinomialNB
print('TfidfVectorizer & MultinomialNB: \n')
pipe = Pipeline([('tvect', TfidfVectorizer()), ('nb', MultinomialNB())])
pipe_params = {'tvect__max_df': [.75, .98], 'tvect__min_df': [4, 5], 'tvect__ngram_range': [(1, 2), (1, 3)], 'nb__alpha': [0.1, 1]}
gs = GridSearchCV(pipe, param_grid=pipe_params, cv=5)
gs.fit(X_train, Y_train)
print("Best score:", gs.best_score_)
print("Train score", gs.score(X_train, Y_train))
print("Test score", gs.score(X_test, Y_test))
prediction = gs.predict(X_test)
cm = metrics.confusion_matrix(Y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
print('')

# Decision Tree Classifier
print('Decision Tree Classifier: \n')
pipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression())])
model = pipe.fit(X_train, Y_train)
prediction = model.predict(X_test)
print("Score: ", accuracy_score(Y_test, prediction))
cm = metrics.confusion_matrix(Y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
print('')

# Random Forest Classifier
pipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
model = pipe.fit(X_train, Y_train)
prediction = model.predict(X_test)
print('Random Forest Classifier :\n')
print("accuracy: ", accuracy_score(Y_test, prediction))
cm = metrics.confusion_matrix(Y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
