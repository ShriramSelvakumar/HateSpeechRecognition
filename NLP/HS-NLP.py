import pandas as pd
import os
import nltk
import re


# Setting display width for panda outputs
# pd.set_option('display.max_colwidth', 200)


class HateSpeechNLP:
    def __init__(self, data):
        self.data = data
        # Getting stopwords from NLTK corpus
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.path = '../Data/'
        self.emojis_array = []

    def fit_transform(self, stem=True, save=True):
        # Removing @usernames
        self.data['text_nousername'] = self.data['text'].apply(lambda x: self.remove_username(x))
        # Removing RT word
        self.data['text_RT'] = self.data['text_nousername'].apply(lambda x: self.remove_RT(x))
        # Removing links
        self.data['text_links'] = self.data['text_RT'].apply(lambda x: self.remove_links(x))
        # Removing Emojis
        # data_Hate['text_emoji'] = data_Hate['text_links'].apply(lambda x: remove_emoji(x))
        # Removing Punctuations
        self.data['text_nopunct'] = self.data['text_links'].apply(lambda x: self.remove_punctuation(x))
        # Get array of emojis
        self.emojis_array = self.finding_emojis()
        # Separating Emojis - to make each emoji as a single token
        self.data['text_emoji'] = self.data['text_nopunct'].apply(lambda x: self.separate_emojis(x))
        # Tokenizing
        self.data['text_tokens'] = self.data['text_emoji'].apply(lambda x: self.tokenize(x))
        # Remove stop words and make text lower case
        self.data['text_no_stop_words_tokens'] = self.data['text_tokens'].apply(lambda x: self.remove_stop_words(x))
        # Saving to pickle
        if save:
            self.save_to_pickle()

    # Function to remove @usernames
    @staticmethod
    def remove_username(text):
        return " ".join(re.split('[@][a-zA-Z0-9_]+', text))

    # Function to remove 'RT' word
    @staticmethod
    def remove_RT(text):
        return_text = " ".join(re.split('[R][T]', text))
        return return_text

    # Removing Punctuations
    @staticmethod
    def remove_punctuation(text):
        # Removing "&amp"
        return_text = " ".join(re.split('[&][a][m][p]', text))
        # Punctuations without '
        punctuations = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
        # Removing regular punctuations
        return_text = " ".join(re.split('[' + punctuations + ']+', return_text))
        # sometimes ' connect words - "I'm" - below line outputs - "Im"
        return_text = "".join(re.split('[\'’]', return_text))
        #  Removing other punctuations
        return_text = " ".join(re.split('[…“”—⁣️‼„‘´, ･─•⠀❓¿˽।‍ ‍‍]+', return_text))
        return_text = " ".join(re.split(r'\\', return_text))
        return return_text

    # Function to remove links
    @staticmethod
    def remove_links(text):
        # return_text = " ".join(re.split(r'http\S+', text))
        return_text = " ".join(re.split(r'http[a-zA-Z0-9.\/:]+|https[a-zA-Z0-9.\/:&]+', text))
        return return_text

    # Function to remove emoji strings - in format "&#12378"
    @staticmethod
    def remove_emoji(text):
        return_text = " ".join(re.split('[&#][0-9]+', text))
        return return_text

    # Function to Tokenize
    @staticmethod
    def tokenize(text):
        # tokens = " ".join(re.findall('[\w]+', text))
        tokens = text.split()
        tokens = [word.lower() for word in tokens]
        return tokens

    # Finding emojis
    def finding_emojis(self):
        emojis = pd.DataFrame()
        for line in self.data.text_nopunct:
            non_characters = [char for char in re.findall('\W', line) if char not in [' ', '\n']]
            if non_characters:
                emojis = emojis.append(non_characters)
        return emojis[0].unique()

    # Function to make separate emojis as individual tokens
    def separate_emojis(self, text):
        return_text = ''
        for char in text:
            if char in self.emojis_array:
                return_text = return_text + ' ' + char + ' '
            else:
                return_text = return_text + char
        return return_text

    # Function to make words lower case and remove stop words
    def remove_stop_words(self, text):
        return_text = [word for word in text if word not in self.stopwords]
        return return_text

    # Save DataFrame to pickle
    def save_to_pickle(self):
        self.data.to_pickle(self.path + 'Data-Hate-DF.pkl')
        print('Saved DataFrame to Pickle')
        return


