import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.tokenize import word_tokenize

df_amazon = pd.read_csv (r"D:\Manoj\1ExcelR\Data\amazon_alexa.tsv",  sep="\t")
df_amazon= df_amazon.dropna()
df_amazon.info()
import string
punctuations = string.punctuation

#Stop words
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
mystopword = set(["...","also","would","still","dot","n't"])

def stopfun(wt1):
    filtered1 =[]
    for w in wt1:
        if w not in stop_words:
            if w not in string.punctuation:
                if w not in mystopword:
                    if len(w)>2:
                        filtered1.append(w)
    return filtered1

str10="it will help in cleaning the text?"
mytokens10=word_tokenize(str10)
mytokens10
st=stopfun(mytokens10)
st

#Lemma
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
def lemfun(wt1):
    lem_words=[]
    for w in wt1:
        lem_words.append(lem.lemmatize(w,'v'))          #v - verb
    return lem_words

def pre_process(str1):
    # Tokenization
    str1 = str1.lower()
    mytokens=word_tokenize(str1)
    # Removing stop words & punctution
    mytokens = stopfun (mytokens)
    # Lemmatizing
    mytokens= lemfun(mytokens)
    return mytokens

str10 = "It will help in cleaning cleaned the text?"
pp=pre_process(str10)
pp

# the features we want to analyze
X = df_amazon['verified_reviews']
# the labels, or answers, we want to test against
ylabels = df_amazon['feedback'] 

#CountVectorizer
#cv_vector = CountVectorizer(tokenizer = pre_process)
cv_vector = CountVectorizer(tokenizer = pre_process,min_df= .05,max_df=.90)
    # min_df is used for removing terms that appear too infrequently
    # max_df is used for removing terms that appear too frequently
x_train_cv = cv_vector.fit_transform(X)
x_train_cv_df = pd.DataFrame(x_train_cv.toarray(),columns=list(cv_vector.get_feature_names_out()))
x_train_cv_df.shape

tot1 =x_train_cv_df.sum(axis=0)
tot1.sort_values(inplace= True,ascending=False)

#TfidfVectorizer
tf_vector = TfidfVectorizer(tokenizer = pre_process,min_df= .05,max_df=.90)
x_train_tf = tf_vector.fit_transform(X)
x_train_tf_df = pd.DataFrame(x_train_tf.toarray(),columns=list(tf_vector.get_feature_names_out()))
x_train_tf_df.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train_cv_df , ylabels, test_size=0.25,
                                                    random_state=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB as MB

classifier = RandomForestClassifier(n_estimators=100,max_depth=3,min_samples_leaf=5,
                                    random_state=10)
#classifier = MB()

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
# Model Accuracy
print('Training Accuracy : {:.3f}'.format(classifier.score(X_train, y_train)))
print('Testing Accuracy : {:.3f}'.format(classifier.score(X_test, y_test)))

##############################################3


