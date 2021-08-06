# Processing Hate Data - Part 2
# Separating into two parts for more readability & less processing
import pandas as pd
import nltk
import re


# Setting display width for panda outputs
pd.set_option('display.max_colwidth', 200)
# PorterStemmer
ps = nltk.PorterStemmer()


def stem_words(text):
    return_text = [ps.stem(word) for word in text]
    return return_text


def save_to_pickle():
    data_Hate.to_pickle('../Data/Data-Hate-Stemmed-DF.pkl')
    print('Saved clean data to Pickle')
    return


# Importing Pickle file created in other file
data_Hate = pd.read_pickle('../Data/Data-Hate-DF.pkl')
# Finding non words
# words = set(nltk.corpus.words.words())
# test = " ".join(w for w in data_Hate.text_no_stop_words[0] if w in words or not w.isalpha())

# Stemming words
data_Hate['cleaned_text_tokens'] = data_Hate['text_no_stop_words_tokens'].apply(lambda x: stem_words(x))
save_to_pickle()

















