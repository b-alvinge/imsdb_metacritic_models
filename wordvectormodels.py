import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from gensim.models import Word2Vec
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

import string

import re

import contractions


#import gensim.downloader as api

from gensim.models.fasttext import load_facebook_vectors

from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="facebook/fasttext-en-vectors", filename="model.bin")




from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from names_dataset import NameDataset
nd = NameDataset()
top_US_first_names = nd.get_top_names(n=10000, use_first_names=True, country_alpha2='US')['US']
top_US_last_names = nd.get_top_names(n=10000, use_first_names=False, country_alpha2='US')['US']


top_US_first_names_list = top_US_first_names['M']+top_US_first_names['F']
top_US_last_names_list = top_US_last_names

# Example: Load your dataset
df = pd.read_excel('title_script_scores_output2.xlsx')

print("data rows before cleaning scores: ",len(df))

df = df[pd.to_numeric(df['meta_score'], errors='coerce').notnull()]

print("data rows after cleaning scores: ",len(df))

df['script'] = df['script'].replace(r'_x000D_','', regex=True)

#df['script'] = df['script'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

#df['script'] = df['script'].str.lower()

#df['script'] = df['script'].replace(r"""[^a-zA-Z\s.,-:'"]""", '', regex=True)

#df['script'] = df['script'].replace(r"""[^a-zA-Z\s]""", '', regex=True)



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['script'], df['meta_score'], test_size=0.2, random_state=1)

#X_train, X_hold, y_train, y_hold = train_test_split(df['script'], df['meta_score'], train_size=.6)
#X_valid, X_test, y_valid, y_test = train_test_split(X_hold, y_hold, train_size=.5)

#with open('movies_text.txt') as f_input:
#    movies_text_sentences = [list(filter(('@').__ne__, word_tokenize(sentence))) for line in f_input for sentence in sent_tokenize(line)]




#X_train_sentences = [word_tokenize(sentence) for script in X_train for sentence in sent_tokenize(script)]


# Train a Word2Vec model
#w2v_model = Word2Vec(sentences=movies_text_sentences, vector_size=10, window=10, min_count=1, workers=4)

#w2v_model = api.load('word2vec-google-news-300')

#w2v_model = api.load('fasttext-wiki-news-subwords-300')
w2v_model = load_facebook_vectors(model_path)

#print(w2v_model.similar_by_word("not", topn=10))

one_hot_model = {}
for script in df['script']:
    one_hot_vector = one_hot(script, n=500)
    tokens = script.split(' ')
    one_hot_dict = dict(zip(tokens, one_hot_vector))
    one_hot_model.update(one_hot_dict)



# Function to vectorize scripts based on the model
def vectorize_scripts(scripts, model=None, oov_handling=False, join_method='mean', padding_size=2000000):
    if join_method == 'mean' or join_method == 'sum':
        size = model.vector_size
        vectorized_scripts = np.zeros((len(scripts), size))
    elif join_method == 'padding' or join_method == 'concatenate':
        size = padding_size
        vectorized_scripts = []
    for i, script in enumerate(scripts):
        vectors = []
        for sentence in sent_tokenize(script):
            within_vocab_sentence = []
            sentence = contractions.fix(sentence)
            for token in word_tokenize(sentence):
                try:
                    vectors.append(model[token])
                except KeyError:
                    print("could not find vector for token: ", token)
                if not(isinstance(model, dict)) and token not in model.key_to_index and oov_handling == True:
                    print(token, "is not a 'word' in the vocabulary.")
                    if re.search('[a-zA-Z]', token) is not None:
                        print(token, "might be a word or name.")
                        if len(re.findall(r'\W+', token)) < 1:
                            if token not in stopwords.words('english'):
                                if token[0]+token[1:].lower() in top_US_first_names_list+top_US_last_names_list:
                                    print(token[0]+token[1:].lower(), "is a name.")

                                elif token.lower() in model.key_to_index:
                                    print(token.lower(), "is a word in the vocabulary.")
                                    word_vector = model[token.lower()]
                                    vectors.append(word_vector)
                                    within_vocab_sentence.append(token)

                            else:
                                print(token, "is a stop word.")
                        else:
                            print(token, "is not a singular word or name.")
                            re_token_search = re.search('\W*(\w+-?\w+-?\w+-?\w+-?\w+-?\w+)\W*', token)
                            if re_token_search is not None:
                                print(re_token_search.group(1),"might be a word.")
                                if re_token_search.group(1).lower() in model.key_to_index:
                                    print(re_token_search.group(1).lower(), "is a word in the vocabulary.")
                                    word_vector = model[re_token_search.group(1).lower()]
                                    vectors.append(word_vector)
                                    within_vocab_sentence.append(token)


                    else:
                        print(token, "is a non-word token.")


        if vectors:
            if join_method == 'mean':
                vectorized_scripts[i] = np.mean(vectors, axis=0)
            elif join_method == 'sum':
                vectorized_scripts[i] = np.sum(vectors, axis=0)
            elif join_method == 'concatenate':
                concatenated_vector = np.concatenate(vectors)
                #print(concatenated_vector.shape)
                vectorized_scripts.append(concatenated_vector)
            elif join_method == 'padding':
                vectorized_scripts.append(vectors)
        else:
            vectorized_scripts[i] = np.zeros(size)

    if join_method == 'padding' or join_method == 'concatenate':
        vectorized_scripts = pad_sequences(vectorized_scripts, maxlen=size, padding='post')
    #print(vectorized_scripts.shape)
    return vectorized_scripts




# Vectorize the training and testing scripts
X_train_vect = vectorize_scripts(X_train, w2v_model, join_method='concatenate')
#X_val_vect = vectorize_scripts(X_valid, w2v_model, join_method='concatenate')
X_test_vect = vectorize_scripts(X_test, w2v_model,join_method='concatenate')

# Train the model
#regressor = GradientBoostingRegressor(verbose=True)

#regressor = MLPRegressor(random_state=1, max_iter=1000000, verbose=True, early_stopping=False, hidden_layer_sizes=(500,), learning_rate='adaptive')

regressor = PLSRegression(n_components=100)
"""
batch_size, train_loss_, valid_loss_rmse, valid_loss_r2 = 50, [], [], []
for _ in range(150):
    for b in range(batch_size, len(y_train), batch_size):
        X_batch, y_batch = X_train_vect[b-batch_size:b], y_train[b-batch_size:b]
        regressor.partial_fit(X_batch, y_batch)
        train_loss_.append(regressor.loss_)
        valid_loss_rmse.append(mean_squared_error(y_valid, regressor.predict(X_val_vect) / 2))
        valid_loss_r2.append(r2_score(y_valid, regressor.predict(X_val_vect)))
        print("train_loss: ", train_loss_[-1],"valid_loss_rmse: ",valid_loss_rmse[-1], "valid_loss_r2: ",valid_loss_r2[-1])

plt.plot(range(len(train_loss_)), train_loss_, label="train loss")
plt.plot(range(len(train_loss_)), valid_loss_rmse, label="validation loss (rmse)")
plt.legend()
plt.savefig('results.png')
"""
regressor.fit(X_train_vect, y_train)

# Make predictions
predictions = regressor.predict(X_test_vect)

# Evaluate the model
rmse = sqrt(mean_squared_error(y_test, predictions))

r2_score = r2_score(y_test, predictions)
print(f'test_Root Mean Squared Error: {rmse}')
print(f'test_R2_score: {r2_score}')

# Example of making a prediction with new text
#new_text = []
#new_predictions = pipeline.predict(new_text)
#print(f'Predictions: {new_predictions}')
