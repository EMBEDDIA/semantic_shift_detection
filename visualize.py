import pandas as pd

import pickle
from sklearn.metrics.pairwise import cosine_similarity
import plotly
import plotly.graph_objs as go
import numpy as np
from scipy.stats.stats import pearsonr
import argparse
import os

def get_shifts(input_path):
    shifts_dict = {}
    df_shifts = pd.read_csv(input_path, sep=',', encoding='utf8')
    for idx, row in df_shifts.iterrows():
        shifts_dict[row['word']] = row['shift_index']
    return shifts_dict


def get_cos_dist(words, shifts_dict, path, years):

    cds = []
    shifts = []
    word_list = []

    with open(path, 'rb') as f:
        vocab_vectors = pickle.load(f)

    for w in words:

        words_emb = []

        for year in years:
            year_word = w + '_' + year
            if year_word in vocab_vectors:
                words_emb.append(vocab_vectors[year_word])

        print(w)

        cs = cosine_similarity(words_emb[0], words_emb[1])[0][0]
        cds.append(1 - cs)
        shifts.append(shifts_dict[w])
        word_list.append(w)


    return cds, shifts, word_list


def visualize(x,y, words):

    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)


    trace0 = go.Scatter(
        x=x,
        y=y,
        name='Words',
        mode='markers+text',

        marker=dict(

            size=12,
            line=dict(
                width=0.5,
            ),
            opacity=0.75,
        ),
        textfont=dict(color="black", size=19),
        text=words,
        textposition='bottom center'
    )

    trace1 = go.Scatter(
        x=x,
        y=poly1d_fn(x),
        mode='lines',
        name='logistic regression',

    )

    layout = dict(title='Correlation between gs semantic shifts and calculated shifts',
                  yaxis=dict(zeroline=False, title= 'Semantic shift index', title_font = {"size": 20},),
                  xaxis=dict(zeroline=False, title= 'Cosine distance', title_font = {"size": 20},),
                  hovermode='closest',

                  )

    data = [trace0, trace1]
    fig = go.Figure(data=data, layout=layout)
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    plotly.offline.plot(fig, filename='visualizations/liverpool.html')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', type=str,
                        help='Path to output time embeddings',
                        default='embeddings/liverpool.pickle')
    parser.add_argument('--shifts_path', type=str,
                        help='Path to gold standard semantic shifts path',
                        default='data/liverpool/liverpool_shift.csv')
    args = parser.parse_args()


    years = ['2013', '2017']

    shifts_dict = get_shifts(args.shifts_path)
    words = list(shifts_dict.keys())
    cds, shifts, words = get_cos_dist(words, shifts_dict, args.embeddings_path, years)

    #don't add text to the graph for these words, makes graph less messy
    dont_draw_list = ['millionaires', 'schedules', 'tourists', 'moaned', 'semifinals', 'desert', 'talents', 'scorpion',
                      'seeded', 'vomit', 'naked', 'strings', 'alternatives', 'leaks', 'bait', 'erect', 'graduate',
                      'travel', 'determine', 'explaining', 'soak', 'mouthpuiece', 'congestion', 'revisionism', 'slave',
                      'revisonist', 'emotion', 'behaviour', 'listen', 'sentence', 'voice', 'relieved', 'mouthpiece', 'astonishing',
                      'participate', 'implied', 'astonishing', 'revisionist', 'patient', 'preventing', 'accomplish', 'narrative',
                      'listened', 'egyptian', 'clenched', 'croatian', 'leans', 'snake']

    filtered_words = []

    for w in words:
        if w in dont_draw_list:
            filtered_words.append('')
        else:
            filtered_words.append(w)

    words = filtered_words

    print("Pearson coefficient: ", pearsonr(cds, shifts))
    visualize(cds, shifts, words)




