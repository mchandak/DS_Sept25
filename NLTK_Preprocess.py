import nltk

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.tokenize import word_tokenize
str1= "India is exporting $100 billion software services to USA"
str1 = str1.lower()
str1

#Word Tokenization
wt = word_tokenize(str1)
print(wt)

#Sentence Tokenization
from nltk.tokenize import sent_tokenize
str2 = "India is exporting $100 billion Software to USA. Software is one of the most growing sector"
ws=sent_tokenize(str2)
print(ws)
ws[0]

# Frequency distribution
from nltk.probability import FreqDist
wt1 = word_tokenize(str2)
fdist = FreqDist(wt1)
fdist
fdist.most_common(2)
fdist.plot(10)              #10 words

#Stop words
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)

filtered1 =[]
for w in wt:
    if w not in stop_words:
        filtered1.append(w)
print("Tokenized :",wt)
print("Filterd :",filtered1)


#Lemmatization  (it needs context)
str3 = "I am a runner running in the race as i love to run since I ran past years"
#str3 = "connection connectivity connected connecting"
#str3 = "studying studies studied"
wt1 = word_tokenize(str3)
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
lem_words=[]
for w in wt1:
    lem_words.append(lem.lemmatize(w,'v'))
lem_words

#Stemming
str4 = "connection connectivity connected  connecting"
str4 = "studying studies studied"
str4 = "likes liked likely"
str4 = "I am a runner running in the race as i love to run since I ran past years"

from nltk.stem import PorterStemmer
wt = word_tokenize(str4)

ps = PorterStemmer()
stemmed_words=[]
for w in wt:
    stemmed_words.append(ps.stem(w))
stemmed_words

#CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

import pandas as pd
#Count Vectorizer
cv1 = CountVectorizer()

lstVect = ["Hi How are you How are you doing",
         "I a doing very very good",
         "Wow that's awesome really awesome"]

x_traincv = cv1.fit_transform(lstVect)

x_traincv_df = pd.DataFrame(x_traincv.toarray(),columns=list(cv1.get_feature_names_out()))
x_traincv_df

#TF-IDF Vectorizer
# lstVect = ["It is useful product","It is good product", "It is useful product",
#            "It is excellent product", "it is Great product", "good product", "very good product"]                  

tf1 = TfidfVectorizer()
x_traintv = tf1.fit_transform(lstVect)

x_traintv_df = pd.DataFrame(x_traintv.toarray(),columns=list(tf1.get_feature_names_out()))
x_traintv_df



#######################################

#Part of speech (POS)
#nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('tagsets')
#nltk.download('tagsets_json')

pos = nltk.pos_tag(wt)
pos
#nltk.help.upenn_tagset() #list of all tag








############
import nltk
nltk.data.path
nltk.data.path.append("E:\Manoj\BI\\nltk_data")
nltk.data.path.remove('E:\\Manoj\\BI\\1 Python ML BI 19\\ML Python\\env\\nltk_data' )
##########
#Caring’ -> Lemmatization -> ‘Care’
#‘Caring’ -> Stemming -> ‘Car’
