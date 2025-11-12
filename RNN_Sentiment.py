import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import random

# Set seeds for reproducibility
seed_value = 42
tf.random.set_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Text data
texts = [
    "I love this product", "This is amazing", "Great quality",
    "Excellent service", "Best purchase ever", "Very satisfied",
     "Very Bad","Terrible quality", "Worst experience",
    "Very disappointed", "Poor service", "Bad product" 
]
labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

# Step 1: Tokenization - Convert text to sequences
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

#tokenizer.word_index
#sentence_to_sequence = dict(zip(texts, sequences))

# Step 2: Padding - Make all sequences equal length
max_length = 5
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = labels

# Step 3: Build RNN model
model = Sequential([
    Embedding(input_dim=100, output_dim=8, input_length=max_length, name='embedding_layer'),
    SimpleRNN(units=8, activation='relu', name='rnn_layer'),
    Dense(1, activation='sigmoid', name='output_layer')
])

# Step 4: Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train model
model.fit(X, y, epochs=50, batch_size=4, verbose=1,shuffle=False)

# Step 6: Test predictions
test_texts = ["I love this", "Very bad"]
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_X = pad_sequences(test_sequences, maxlen=max_length, padding='post')

predictions = model.predict(test_X)

for text, pred in zip(test_texts, predictions):
    sentiment = "Positive" if pred > 0.5 else "Negative"
    print(f"{text} -> {sentiment} ({pred[0]:.4f})")

#########################################


embedding_layer = Embedding(input_dim=100, output_dim=4, input_length=max_length)
model = Sequential([embedding_layer])
embedding_output = model.predict(X)
embedding_output

# input_dim=100	Vocabulary size — max number of words that
# can be indexed (word IDs go from 1 to 99)
# output_dim=4	Size of the embedding vector for each word 
#(how many numbers represent each word)
# input_length=max_length	The length of each input sentence
# (number of tokens per text, after padding)