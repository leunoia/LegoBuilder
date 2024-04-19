# Replace model.add(LSTM(100)) with model.add(GRU(100, return_sequences=True)) or model.add(Bidirectional(GRU(64, return_sequences=True)))
# This code is the Transformer model. The line above is for the LSTM model.
from numpy import transformers 
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import AdamW

model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
model.compile(optimizer=AdamW(learning_rate=3e-5))
model.fit(tokenized_data, labels)
model.summary()

startText = "lego"
for i in range(3):
    tokenList = tokenizer.texts_to_sequences([startText])[0]
    tokenList = pad_sequences([tokenList], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(tokenList, verbose=0)

    newText = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            newText = word
            break
    startText += " " + newText
print(startText.title())