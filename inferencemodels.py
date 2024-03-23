import pandas as pd

import numpy as np


from causalnlp import Autocoder

import torch

# Example: Load your dataset
df = pd.read_hdf('./data/in/title_script_summary_genres_score.h5')

print("data rows before cleaning scores: ",len(df))

df = df[pd.to_numeric(df['meta_score'], errors='coerce').notnull()]

print("data rows after cleaning scores: ",len(df))

df['script'] = df['script'].replace(r'_x000D_','', regex=True)

ac = Autocoder()
torch.cuda.empty_cache()

# Function to infer genres for a given set of texts.
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
                               batch_size=64)
    return df

# Function to infer genre philosophies for a given set of texts.
def infer_genre_philosophy(df, input_column, multilabel):
    theme_nli_template = "this text is about {}"
    themes = ["confronting death and facing one's ghosts from the past",
     'realising that most of success comes about through taking action',
     "seeking immortality through finding one's destiny in life",
     "examining one's life in order to create one's true self",
     "making the right choices now so as to ensure a better future for all",
    "protecting the weak and bringing the guilty to justice",
    "how success comes when one strips away all facades and show others who one really is",
    "when one helps others make a home, one creates a civilisation where everyone is free to live their best life",
    "Not being enslaved by absolute power and money or one will pay the ultimate price",
    "discovering the magic in oneself that makes life itself an art form",
    "looking for the truth and assigning guilt in spite of the danger",
    "learning how to love is the key to happiness"]

    s = (df.pop(input_column)
         .str.split(r'\s*\r\n\s*\r\n(\r\n)*', expand=True, regex=True)
         .stack()
         .rename(input_column)
         .reset_index(level=1, drop=True))

    df = df.join(s).reset_index(drop=True)

    df[input_column].replace(r'^\s*$', np.nan, inplace=True, regex=True)
    df.dropna(subset=[input_column], inplace=True)



    df = ac.code_custom_topics(docs=df[input_column].values, df=df[['title', 'script', 'meta_genres', 'meta_score']], labels=themes,
                               nli_template = theme_nli_template, max_length=512, multilabel=multilabel,
                               batch_size=32)
    return df

# Function to infer aesthetic qualities for a given set of texts.
def infer_aesthetic_qualities(df, input_column, multilabel):
    aesthetic_nli_template = "this text is {}"
    aesthetic_qualities = ['Terrifying',
     'Exhilarating',
     'Epic',
     'Heartwarming',
     'Enlightening',
     'Authentic',
     'Gritty',
     'Inspirational',
     'Rhythmic',
     'Show-stopping',
     'Competitive',
     'Imaginative',
     'Moody',
     'Futuristic',
     'Suspenseful',
     'Hilarious',
     'Rugged',
     'Magical',
     'Gripping',
     'Intriguing',
     'Poignant',
     'Passionate']

    s = (df.pop(input_column)
         .str.split(r'\s*\r\n\s*\r\n(\r\n)*', expand=True, regex=True)
         .stack()
         .rename(input_column)
         .reset_index(level=1, drop=True))

    df = df.join(s).reset_index(drop=True)

    df[input_column].replace(r'^\s*$', np.nan, inplace=True, regex=True)
    df.dropna(subset=[input_column], inplace=True)



    df = ac.code_custom_topics(docs=df[input_column].values, df=df[['title', 'script', 'meta_genres', 'meta_score']], labels=aesthetic_qualities,
                               nli_template = aesthetic_nli_template, max_length=512, multilabel=multilabel,
                               batch_size=64)
    return df                                        


#df_out = infer_genres(df, 'meta_summary', multilabel=True)

df_out = infer_genre_philosophy(df, 'script', multilabel=True)

#df_out = infer_aesthetic_qualities(df, 'script', multilabel=True)



df_out.to_hdf('./data/out/title_script_metagenres_metascore_genre_philosophy_multilabel.h5', key='df')

