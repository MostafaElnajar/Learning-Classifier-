


import pandas as pd
import numpy as np
import re
import nltk
import string 
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from keras.layers import Dense, LSTM, Embedding, Bidirectional
from keras.utils import to_categorical
from keras.models import Sequential
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer



class TextClassification:
    def __init__(self):  
        self.le = LabelEncoder()
        self.lemm=WordNetLemmatizer()
        self.tokenizer = Tokenizer()
        self.stop_words = stopwords.words("english")  
        self.vocabSize=0
        self.embedding_matrix=0
        self.model_train()


    def remove_stopwords(self,text):
        no_stop = []
        for word in text.split(' '):
            if word not in self.stop_words:
                no_stop.append(word)
        return " ".join(no_stop)

    def remove_punctuation_func(self,text):
        return re.sub(r'[^a-zA-Z0-9]', ' ', text)

    def clean(self,text):
        text = re.sub(r'[^a-zA-Z ]', '', text)
        text = text.lower()      
        return text

   
    #################


   



    def model_getdata(self):
        self.df = pd.read_csv('dataset.csv')
        self.df = self.df[['Sentence','Type']]



    def model_processing(self):
        self.df.drop_duplicates(inplace=True,keep=False)
        self.df['new_Sentence'] = self.df.Sentence.apply(lambda x:x.lower())
        self.df['new_Sentence'] = self.df.Sentence.apply(lambda x:self.lemm.lemmatize(x))
        self.df['New_Sentence'] = self.df.Sentence.apply(self.clean)
        self.df['New_Sentence'] = self.df.Sentence.apply(self.remove_stopwords)
        self.df['New_Sentence'] = self.df.Sentence.apply(self.remove_punctuation_func)

        X = self.df['new_Sentence']
        y = self.df['Type']

        y = self.le.fit_transform(y)
        y = to_categorical(y)
    #     x = cv.fit_transform(df.Text)
        self.text_train, self.text_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

        self.tokenizer.fit_on_texts(self.text_train)
        sequences_train = self.tokenizer.texts_to_sequences(self.text_train)
        sequences_test = self.tokenizer.texts_to_sequences(self.text_test)
        self.X_train = pad_sequences(sequences_train, maxlen=48, truncating='pre')
        self.X_test = pad_sequences(sequences_test, maxlen=48, truncating='pre')

        self.vocabSize = len(self.tokenizer.index_word) + 1

        # Read GloVE embeddings
        path_to_glove_file = 'glove.6B.200d.txt'
        num_tokens = self.vocabSize
        embedding_dim = 200
        hits = 0
        misses = 0
        embeddings_index = {}
        # Read word vectors
        with open(path_to_glove_file,encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

        # Assign word vectors to our dictionary/vocabulary
        self.embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                self.embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        return self.vocabSize,self.embedding_matrix 


    def build_model(self): 
        model = Sequential()
        model.add(Embedding(self.vocabSize, 200, input_length=len(self.X_train[1]), weights=[self.embedding_matrix], trainable=False))
        model.add(LSTM(256, dropout=0.2,recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(128, dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
        return model

    def model_train(self):
        
        self.model_getdata() 
        self.model_processing()
      
        self.model=self.build_model()
        self.model.fit(self.X_train,
                        self.y_train,
                        validation_data=(self.X_test, self.y_test),
                        verbose=1,
                        batch_size=64,
                        epochs=10)  



    def model_evaluate(self):
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=1) 
        return loss, acc


    def model_predict(self, text):
        sentence = self.clean(text)
        sentence = self.remove_stopwords(sentence)
        sentence = self.remove_punctuation_func(sentence)
        sentence = self.tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=48, truncating='pre')

        result = self.le.inverse_transform(np.argmax(self.model.predict(sentence), axis=-1))[0]

        return result

