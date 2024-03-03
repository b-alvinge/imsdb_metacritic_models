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


# Example: Load your dataset
df = pd.read_excel('./data/in/title_script_summary_genres_score.xlsx')

print("data rows before cleaning scores: ",len(df))

df = df[pd.to_numeric(df['meta_score'], errors='coerce').notnull()]

print("data rows after cleaning scores: ",len(df))

df['script'] = df['script'].replace(r'_x000D_','', regex=True)

ac = Autocoder()


# Function to infer genres for a given movie script.
def infer_genres(df):
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
    df = ac.code_custom_topics(docs=df['meta_summary'].values, df=df, labels=genres,
                               nli_template = "The genre of this text is {}", max_length=1024, multilabel=True,
                               batch_size=32)

    return df

# Vectorize the training and testing scripts
df_genres = infer_genres(df)

df_genres.to_excel('./data/out/title_script_summary_genres_score_truby_genres.xlsx')


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