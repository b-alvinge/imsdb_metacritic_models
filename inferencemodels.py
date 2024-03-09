import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt


from causalnlp import Autocoder

from tqdm import tqdm

import torch

# Example: Load your dataset
df = pd.read_hdf('./data/in/title_script_summary_genres_score.h5')

print("data rows before cleaning scores: ",len(df))

df = df[pd.to_numeric(df['meta_score'], errors='coerce').notnull()]

print("data rows after cleaning scores: ",len(df))

df['script'] = df['script'].replace(r'_x000D_','', regex=True)

ac = Autocoder()
torch.cuda.empty_cache()

# Function to infer genres for a given text.
def infer_genres(df, input_column, multilabel):
    genre_nli_template = "The genre of this text is {}"
    genres = [
        "Horror",
        "Action",
        "Myth",
        "Memoir",
        "Coming-of-Age",
        "Science Fiction",
        "Crime",
        "Comedy",
        "Western",
        "Gangster",
        "Fantasy",
        "Thriller",
        "Detective",
        "Love"
    ]
    df = ac.code_custom_topics(docs=df[input_column].values, df=df, labels=genres,
                               nli_template = genre_nli_template, max_length=512, multilabel=multilabel,
                               batch_size=12)
    return df

# Function to infer aesthetic qualities for a given text.
def infer_aesthetic_qualities(df, input_column, multilabel):
    aesthetic_nli_template = "This text has {}"
    aesthetic_qualities = [
        'Engaging Characters',
        'A Compelling Plot',
        'Universal Themes',
        'Immersive World-building',
        'Emotional Resonance'
    ]
    s = (df.pop(input_column)
         .str.split(r'\s*\r\n\s*\r\n(\r\n)*', expand=True, regex=True)
         .stack()
         .rename(input_column)
         .reset_index(level=1, drop=True))

    df = df.join(s).reset_index(drop=True)

    df[input_column].replace(r'^\s*$', np.nan, inplace=True, regex=True)
    df.dropna(subset=[input_column], inplace=True)


    df = ac.code_custom_topics(docs=df[input_column].values, df=df[['title', 'script']], labels=aesthetic_qualities,
                               nli_template = aesthetic_nli_template, max_length=512, multilabel=multilabel,
                               batch_size=32)
    return df


#df_out = infer_genres(df, 'meta_summary', multilabel=True)


df_out = infer_aesthetic_qualities(df.iloc[0:2], 'script', multilabel=True)



df_out.to_hdf('./data/out/title_script_aesthetic_qualities.h5', key='df')


# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(df['script'], df['meta_score'], test_size=0.2, random_state=1)

"""

# Train the model
#regressor = GradientBoostingRegressor(verbose=True)

regressor = MLPRegressor(random_state=1, max_iter=1000, verbose=True, learning_rate='adaptive')

#regressor = PLSRegression(n_components=100)

regressor.fit(X_train_genres, y_train)

# Make predictions
predictions = regressor.predict(X_test_genres)

# Evaluate the model
rmse = sqrt(mean_squared_error(y_test, predictions))

r2_score = r2_score(y_test, predictions)
print(f'test_Root Mean Squared Error: {rmse}')
print(f'test_R2_score: {r2_score}')

# Example of making a prediction with new text
#new_text = []
#new_predictions = pipeline.predict(new_text)
#print(f'Predictions: {new_predictions}')
"""