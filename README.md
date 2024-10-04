
# Next Word Prediction Using LSTM

This project demonstrates how to build a deep learning model using Long Short-Term Memory (LSTM) networks for next word prediction. We use the text of Shakespeare's *Hamlet* as the dataset and develop a Streamlit application to provide real-time next-word predictions.

## Project Overview

The model is trained to predict the next word in a sequence of words, using the following steps:

1. **Data Collection**: The text of Shakespeare's *Hamlet* is loaded and saved for further processing.
2. **Data Preprocessing**: The text is tokenized, sequences are padded, and the data is split into training and testing sets.
3. **Model Building**: An LSTM model is built, consisting of an embedding layer, two LSTM layers, and a dense output layer with a softmax activation function to predict the next word.
4. **Model Training**: The model is trained with early stopping to avoid overfitting.
5. **Model Evaluation**: The model is tested using example sentences.
6. **Deployment**: A Streamlit web application is deployed to allow users to input a sequence of words and receive the predicted next word in real time.

## Features

- **LSTM Model**: A deep learning model using LSTM layers to handle sequence prediction.
- **Early Stopping**: Prevents overfitting by monitoring validation loss and stopping the training process when the loss does not improve.
- **GRU Model**: An alternate implementation using GRU (Gated Recurrent Unit) layers is also provided.
- **Streamlit Application**: A simple web interface where users can input a sequence of words and get the predicted next word.
- **Word Tokenization**: The text is tokenized using Keras, and sequences are padded to ensure uniform input.

## Data Collection

We use Shakespeare's *Hamlet* text from the NLTK library. The data is saved locally for preprocessing and model training.

```python
import nltk
from nltk.corpus import gutenberg
nltk.download('gutenberg')
data = gutenberg.raw('shakespeare-hamlet.txt')

with open('hamlet.txt', 'w') as file:
    file.write(data)
```

## Model Architecture

The model is built with Keras and consists of:

- **Embedding Layer**: Converts the input tokens into dense vectors of fixed size.
- **LSTM Layers**: Two LSTM layers to learn the sequential patterns in the text.
- **Dense Layer**: The final layer predicts the next word with a softmax activation function.

```python
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation="softmax"))
```

## Model Training

The model is trained using categorical cross-entropy loss and the Adam optimizer. We also implement early stopping to monitor validation loss and stop training when the loss stops improving.

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])
```

## Streamlit Application

The model and tokenizer are saved and used in a Streamlit web app to predict the next word based on user input.

```python
st.title("Next Word Prediction with LSTM")
input_text = st.text_input("Enter the sequence of words", "To be or not to")
if st.button("Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')
```

## Installation

To run the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/next-word-prediction.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Dependencies

- `tensorflow`
- `numpy`
- `nltk`
- `streamlit`
- `pickle`
  
Install the dependencies using:

```bash
pip install tensorflow numpy nltk streamlit
```

## Usage

- Run the `app.py` file to start the Streamlit application.
- Input a sequence of words in the text box, and click "Predict Next Word" to see the prediction.
  
## Model Saving

The trained LSTM model and the tokenizer are saved for future use:

```python
model.save('next_word_lstm.h5')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

## Results

The model is capable of predicting the next word in a sequence with reasonable accuracy, depending on the complexity of the input. For example:

```bash
Input: "To be or not to be"
Predicted: "considered"
```

## Future Improvements

- Experiment with different datasets to generalize the model.
- Fine-tune the model by adjusting hyperparameters like the number of LSTM units or the learning rate.
- Extend the Streamlit app to provide more advanced options, such as choosing the number of next words to predict.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
