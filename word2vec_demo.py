import gensim.downloader as api

print("Downloading word2vec-google-news-300 model via gensim ...")
model = api.load('word2vec-google-news-300')

print("Similarity between 'king' and 'queen':", model.similarity('king', 'queen'))
print("Most similar to 'computer':", model.most_similar('computer', topn=5))

temp1=model['cricket']

model.most_similar('cricket')

model.similarity('man','woman')

model.similarity('man','PHP')

model.doesnt_match(['PHP','java','monkey'])
vec = model['king'] - model ['man'] + model['woman']
vec = model['INR'] - model ['India'] + model['England']
model.most_similar([vec])
