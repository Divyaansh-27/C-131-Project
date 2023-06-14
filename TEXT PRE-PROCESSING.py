import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
dataframe=pd.read_excel('C:/Users/HP/Desktop/Python/Project/Project-131/product_dataset/updated_product_dataset.xlsx')
dataframe.head()
dataframe["Emotion"].unique()
encode_product={"Postive":0,"Neutral":1,"Negative":2}
dataframe.replace(encode_product,inplace=True)
dataframe.head()
training_sentences=[]
training_labels=[]
for i in range(len(dataframe)):
    sentence=dataframe.loc[i,"Text"]
    training_sentences.append(sentence)
    label=dataframe.loc[i,"Emotion"]
    training_labels.append(label)
training_sentences[50]
training_labels[50]
vocab_size=10000
embedding_dim=16
oov_taok="<OOV>"
training_size=20000
tokenize=Tokenizer(num_words=vocab_size,oov_token=oov_taok)
tokenize.fit_on_texts(training_sentences)
word_index=tokenize.word_index
word_index["wrong"]
training_sequences=tokenize.texts_to_sequences(training_sentences)
print(training_sequences[0])
print(training_sequences[1])
print(training_sequences[2])
padding_type='post'
max_length=100
trunc_type='post'
training_padded=pad_sequences(training_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)
training_padded
print(training_padded[0:3])