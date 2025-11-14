import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer

text = """
the weather is nice today the weather is rainy tomorrow 
the sun is shining bright the morning is beautiful the night is dark
the sky is clear and blue the clouds are white and fluffy
I love the beautiful weather on sunny days
the rain brings fresh green plants and flowers to the garden
the wind blows gently through the trees on the hillside
the stars shine brightly in the night sky above us
the moon rises high in the dark sky every night
the sunset paints the sky with orange and pink colors
the sunrise brings a new day full of hope and energy
the temperature is perfect for outdoor activities today
I enjoy walking in the park during the early morning hours
the weather forecast predicts rain for tomorrow afternoon
the humid air makes it uncomfortable to go outside
the clear blue sky indicates good weather for the weekend
"""

# Step 1: Tokenization - Convert words to numbers
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1

print(f"Vocabulary size: {vocab_size}")
print(f"Total tokens: {len(sequences)}\n")

# Step 2: Create training sequences (4 words -> predict next word)
seq_length = 4
X, y = [], []

for i in range(len(sequences) - seq_length):
    X.append(sequences[i:i+seq_length])
    y.append(sequences[i+seq_length])

X = np.array(X)
y = np.array(y)

print(f"Training samples: {len(X)}\n")

# Step 3: Build LSTM model with stacked layers for better learning
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=seq_length),
    LSTM(units=32, activation='tanh', return_sequences=True),
    LSTM(units=16, activation='tanh'),
    Dense(64, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

# Step 4: Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training model...")
model.fit(X, y, epochs=150, batch_size=8, verbose=1)

# Step 5: Create word-to-index reverse mapping
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

# Prediction function
def predict_next_word(input_text):
    # Convert input text to token sequence
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    # Keep only last seq_length (4) tokens
    # Example: if input has 6 words but seq_length=4, keep last 4 words
    token_list = token_list[-seq_length:]
    
    # Pad with zeros if sequence is shorter than seq_length
    if len(token_list) < seq_length:
        # Pad with zeros (0) at the beginning to match seq_length
        token_list = [0] * (seq_length - len(token_list)) + token_list
    
    # Make prediction using trained model
    # pred is probability distribution over all vocab_size words
    pred = model.predict(np.array([token_list]), verbose=0)
    # Find index of word with highest probability (most likely next word)
    # np.argmax() returns index of maximum value
    predicted_idx = np.argmax(pred[0])
    predicted_word = reverse_word_index.get(predicted_idx, "unknown")
    confidence = pred[0][predicted_idx]
    
    return predicted_word, confidence

# Test predictions
print("\nNext Word Predictions:")
test_inputs = [
    "the weather is nice",
    "the sun is shining",
    "the night sky is",
    "the beautiful weather"
]

for inp in test_inputs:
    next_word, conf = predict_next_word(inp)
    print(f"'{inp}' -> '{next_word}' (confidence: {conf:.4f})")




