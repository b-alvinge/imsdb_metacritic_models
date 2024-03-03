import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

import string

from nltk.corpus import stopwords
stop = stopwords.words('english') + list(string.punctuation)

# Example: Load your dataset
df = pd.read_excel('title_script_scores_output.xlsx')

df['script'] = df['script'].replace(r'_x000D_','', regex=True)

df['script'] = df['script'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop])).str.lower()

df['script'] = df['script'].replace('[^a-zA-Z0-9 ]', '', regex=True)

print("data rows before cleaning scores: ",len(df))

df = df[pd.to_numeric(df['meta_score'], errors='coerce').notnull()]

print("data rows after cleaning scores: ",len(df))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['script'], df['meta_score'], test_size=0.2, random_state=42)


# Create a machine learning pipeline
pipeline = make_pipeline(TfidfVectorizer(),
              GradientBoostingRegressor())

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

# Evaluate the model
rmse = sqrt(mean_squared_error(y_test, predictions))
r2_score = r2_score(y_test, predictions)
print(f'Root Mean Squared Error: {rmse}')
print(f'R2_score: {r2_score}')

# Example of making a prediction with new text
#new_text = []
#new_predictions = pipeline.predict(new_text)
#print(f'Predictions: {new_predictions}')