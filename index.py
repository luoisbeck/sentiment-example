import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

# file_list = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
hist_data_amazon = np.genfromtxt('amazon_cells_labelled.txt', delimiter='\t', dtype=None)
hist_data_yelp = np.genfromtxt('yelp_labelled.txt', delimiter='\t', dtype=None)

hist_data = np.concatenate((hist_data_amazon, hist_data_yelp))

# TODO: Parse imdb and yelp files

X_vals, Y_vals = zip(*hist_data)

my_prediction = ['its a sunny day', 'ive seen better days']

vectorizer = TfidfVectorizer(min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(X_vals)
my_prediction = vectorizer.transform(my_prediction)

X_train, X_test, Y_train, Y_test = train_test_split(train_vectors, Y_vals, test_size=0.7, random_state=42)

C_iter = [0.1, 1, 10, 100]

for c_value in C_iter:
    # for degree_value in range(0,10):
        # print ("degree value: " + str(degree_value))
    clf = svm.SVC(kernel='linear', C=c_value)  # , degree=int(degree_value))
    clf.fit(X_train, Y_train)
    print("State: " + str(42) + " - C: " + str(c_value) + " - Score: " + str(clf.score(X_test, Y_test)))
    print("Prediction: " + str(clf.predict(my_prediction)))