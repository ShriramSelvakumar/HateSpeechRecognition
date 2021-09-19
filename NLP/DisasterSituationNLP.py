import pandas as pd
import os
import nltk
import datetime
import re

nltk.download('stopwords')


class DisasterSituationNLP:
    def __init__(self, data, stem=True, save=False, default_name=False):
        self.data = data
        self.stem = stem
        self.save = save
        self.default_name = default_name
        # Getting stopwords from NLTK corpus
        self.stopwords = nltk.corpus.stopwords.words('english')
        # Adding 'RT' to stopwords
        self.stopwords.append('rt')
        # Base path to store data
        self.path = '../Data/'
        if self.stem:
            self.ps = nltk.PorterStemmer()  # PorterStemmer
        self.emojis_array = []

    def fit_transform(self):
        # Removing @usernames
        self.data['text_nousername'] = self.data['text'].apply(lambda x: self.remove_username(x))
        # Removing links
        self.data['text_nolinks'] = self.data['text_nousername'].apply(lambda x: self.remove_links(x))
        # Removing Punctuations
        self.data['text_nopunct'] = self.data['text_nolinks'].apply(lambda x: self.remove_punctuation(x))
        # Get array of emojis
        self.emojis_array = self.finding_emojis()
        # Separating Emojis - to make each emoji as a single token
        self.data['text_emoji'] = self.data['text_nopunct'].apply(lambda x: self.separate_emojis(x))
        # Tokenizing
        self.data['text_tokens'] = self.data['text_emoji'].apply(lambda x: self.tokenize(x))
        # Create clean text attribute/feature
        self.data['cleaned_text'] = self.data['text_tokens'].apply(lambda x: self.join_tokens(x))
        # Remove stop words and make text lower case
        self.data['text_no_stop_words_tokens'] = self.data['text_tokens'].apply(lambda x: self.remove_stop_words(x))
        # Create clean text attribute/feature
        self.data['cleaned_text_no_stop_words'] = self.data['text_no_stop_words_tokens'].apply(lambda x:
                                                                                               self.join_tokens(x))
        if self.stem:
            # Stemming words
            self.data['stemmed_text_tokens'] = self.data['text_no_stop_words_tokens'].apply(lambda x:
                                                                                            self.stem_words(x))
            # Create clean text attribute/feature
            self.data['cleaned_stemmed_text'] = self.data['stemmed_text_tokens'].apply(lambda x: self.join_tokens(x))
            # Find length of stemmed and stop words removed tokens and create it as new attribute/feature
            self.data['length'] = self.data['stemmed_text_tokens'].apply(lambda x: self.finding_length(x))
            # Find length of original tokens with stop words and create it as new attribute/feature
            self.data['length_original_tokens'] = self.data['text_tokens'].apply(lambda x: self.finding_length(x))
            # Find length of original text with stop words and create it as new attribute/feature
            self.data['length_original_text'] = self.data['cleaned_text'].apply(lambda x: self.finding_length(x))
            # Find number of non words and create it as new attribute/feature
            self.data['number_non_words'] = self.data['stemmed_text_tokens'].apply(lambda x:
                                                                                   self.finding_non_words(x))
        # Saving to pickle
        if self.save:
            self.save_to_pickle(self.default_name)
        return self.data.loc[:, ['text', 'cleaned_stemmed_text', 'length', 'length_original_tokens', 'keyword',
                                 'length_original_text', 'number_non_words', 'target']]

    # Function to remove @usernames
    @staticmethod
    def remove_username(text):
        return " ".join(re.split('[@][a-zA-Z0-9_]+', text))

    # Removing Punctuations
    @staticmethod
    def remove_punctuation(text):
        # Removing "&amp"
        return_text = " ".join(re.split('[&][a][m][p]', text))
        # Removing '-' with whitespace around it - to protect 'I-77'
        # return_text = " ".join(re.split('\s-\s', return_text))
        # Punctuations without ' & -
        punctuations = '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'
        # Removing regular punctuations
        return_text = " ".join(re.split('[' + punctuations + ']+', return_text))
        # sometimes ' connect words - "I'm" - below line outputs - "Im"
        return_text = "".join(re.split('[\'’]', return_text))
        #  Removing other punctuations
        return_text = " ".join(re.split('[ุ…“”—⁣️‼„‘´, ･─•⠀❓¡˃－！❞〝≧¨❛⏪≦、―«：–❝˂-︿¿˽।‍ ‍‍̣]+', return_text))
        return_text = " ".join(re.split(r'\\', return_text))
        return_text = " ".join(re.split(r'-', return_text))
        return return_text

    # Function to remove links
    @staticmethod
    def remove_links(text):
        # return_text = " ".join(re.split(r'http\S+', text))
        return_text = " ".join(re.split(r'http[a-zA-Z0-9.\/:]+|https[a-zA-Z0-9.\/:&]+', text))
        return return_text

    # Function to Tokenize
    @staticmethod
    def tokenize(text):
        # tokens = " ".join(re.findall('[\w]+', text))
        tokens = text.split()
        tokens = [word.lower() for word in tokens]
        return tokens

    @staticmethod
    def join_tokens(text):
        return " ".join(text)

    # Finding emojis
    def finding_emojis(self):
        emojis = pd.DataFrame()
        for line in self.data.text_nopunct:
            non_characters = [char for char in re.findall('\W', line) if char not in [' ', '\n']]
            if non_characters:
                emojis = emojis.append(non_characters)
        if emojis.empty:
            emojis[0] = ['😂']
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

    # Function to remove stop words
    def remove_stop_words(self, text):
        return_text = [word for word in text if word not in self.stopwords]
        return return_text

    def stem_words(self, text):
        return_text = [self.ps.stem(word) for word in text]
        return return_text

    @staticmethod
    def finding_length(text):
        return len(text)

    @staticmethod
    def finding_non_words(text):
        regex = re.compile(r'[\d][\d][\d][\d][\d][\d]')
        sum_non_words = sum([1 for word in text if word in regex.findall(word)])
        regex = re.compile(r'[\d][\d][\d][\d]')
        sum_non_words = sum_non_words + sum([1 for word in text if word in regex.findall(word)])
        regex = re.compile(r'[\W]')
        sum_non_words = sum_non_words + sum([1 for word in text if word in regex.findall(word)])
        return sum_non_words

    @staticmethod
    def clean_text(text):
        return text.split()

    # Save DataFrame to pickle
    def save_to_pickle(self, default_name):
        os.makedirs(self.path, exist_ok=True)
        if not default_name:
            self.data.to_pickle(self.path + 'DisasterSituation_DataFrame_' +
                                datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl')
        else:
            if self.stem:
                self.data.to_pickle(self.path + 'Data-DS-Stemmed-DF.pkl')
            else:
                self.data.to_pickle(self.path + 'Data-DS-DF.pkl')
        print('Saved DataFrame to Pickle')
        return


