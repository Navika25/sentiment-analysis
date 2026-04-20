from sklearn.feature_extraction.text import CountVectorizer

def vectorize_text(X_train):
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    return vectorizer, X_train_vec