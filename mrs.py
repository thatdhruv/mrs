#
# mrs.py
#
# Music Recommendation System using Machine  Learning
# Machine Learning Internship Project (IEP-2023)
#
# Uses a dataset of Spotify songs with different genres and audio features
# https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
# Dataset last updated October 2022
# Size on disk: 20.1MB
#
# Created by Dhruv Trivedi on 19-07-2023
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')

f = pd.read_csv('dataset.csv')

"""
plt.subplots(figsize = (15, 5))

floats = []
for col in f.columns:
    if f[col].dtype == 'float':
        floats.append(col)
print(len(floats))

for i, col, in enumerate(floats):
    plt.subplot(2, 5, i + 1)
    sb.distplot(f[col])
plt.tight_layout()
plt.show()
"""

f = f.sort_values(by=['popularity'], ascending=False).head(10000)
f.drop_duplicates(subset=['track_name'], keep = 'first', inplace = True)

gen = CountVectorizer()
gen.fit(f['track_genre'])

def get_similarities(song_name, data):
    text1 = gen.transform(data[data['track_name'] == song_name]['track_genre']).toarray()
    num1 = data[data['track_name'] == song_name].select_dtypes(include = np.number).to_numpy()

    sim = []
    for i, row in  data.iterrows():
        track_name = row['track_name']
        text2 = gen.transform(data[data['track_name'] == track_name]['track_genre']).toarray()
        num2 = data[data['track_name'] == track_name].select_dtypes(include = np.number).to_numpy()
        text_sim = cosine_similarity(text1, text2)[0][0]
        num_sim = cosine_similarity(num1, num2)[0][0]
        sim.append(text_sim + num_sim)
    return sim

def recommend(song_name, data = f):
    if f[f['track_name'] == song_name].shape[0] == 0:
        print('No results found for \'{}\'\nShowing some songs you may like:\n'.format(song_name))

        for song in data.sample(n = 5)['track_name'].values:
            print(song)
        return

    data['similarity_factor'] = get_similarities(song_name, data)
    data.sort_values(by=['similarity_factor', 'popularity'], ascending = [False, False], inplace = True)
    print('\nHere are some songs similar to \'{}\':'.format(song_name))
    print(data[['track_name', 'artists']][2:7])

song = str(input("Enter a song name: "))
recommend(song)

