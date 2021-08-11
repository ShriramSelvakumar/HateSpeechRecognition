import pandas as pd
import nltk
import re

# Setting display width for panda outputs
pd.set_option('display.max_colwidth', 200)
# Getting stopwords from NLTK corpus
stopwords = nltk.corpus.stopwords.words('english')
# Reading Hate Speech data
data_Disaster = pd.read_csv('../Data/DS_DATA.csv', sep=',', index_col=0)


# Function to remove @usernames
def remove_username(text):
    return_text = " ".join(re.split('[@][a-zA-Z0-9_]+', text))
    return return_text


# Function to remove 'RT' word
def remove_RT(text):
    return_text = " ".join(re.split('[R][T]', text))
    return return_text


# Removing Punctuations
def remove_punctuation(text):
    # Removing "&amp"
    return_text = " ".join(re.split('[&][a][m][p]', text))
    # Removing '-' with whitespace around it - to protect 'I-77'
    # return_text = " ".join(re.split('\s-\s', return_text))
    # Punctuations without ' & -
    punctuations = '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'
    # Removing regular punctuations
    return_text = " ".join(re.split('['+punctuations+']+', return_text))
    # sometimes ' connect words - "I'm" - below line outputs - "Im"
    return_text = "".join(re.split('[\'’]', return_text))
    #  Removing other punctuations
    return_text = " ".join(re.split('[ุ…“”—⁣️‼„‘´, ･─•⠀❓¡˃－！❞〝≧¨❛⏪≦、―«：–❝˂-︿¿˽।‍ ‍‍̣]+', return_text))
    return_text = " ".join(re.split(r'\\', return_text))
    return_text = " ".join(re.split(r'-', return_text))
    return return_text


# Function to remove links
def remove_links(text):
    # return_text = " ".join(re.split(r'http\S+', text))
    return_text = " ".join(re.split(r'http[a-zA-Z0-9.\/:]+|https[a-zA-Z0-9.\/:&]+', text))
    return return_text


# Function to Tokenize
def tokenize(text):
    # tokens = " ".join(re.findall('[\w]+', text))
    tokens = text.split()
    tokens = [word.lower() for word in tokens]
    return tokens


# Finding emojis
def finding_emojis(data):
    emojis = pd.DataFrame()
    for line in data:
        non_characters = [char for char in re.findall('\W', line) if char not in [' ', '\n']]
        if non_characters != []:
            emojis = emojis.append(non_characters)
    return emojis[0].unique()


# Function to make separate emojis as individual tokens
def separate_emojis(text):
    return_text = ''
    for char in text:
        if char in emojis_array:
            return_text = return_text + ' ' + char + ' '
        else:
            return_text = return_text + char
    return return_text


# Function to make words lower case and remove stop words
def remove_stop_words(text):
    return_text = [word for word in text if word not in stopwords]
    return return_text


# Save DataFrame to pickle
def save_to_pickle():
    data_Disaster.to_pickle('../Data/Data-Disaster.pkl')
    print('Saved data to Pickle')
    return


# Removing @usernames
data_Disaster['text_nousername'] = data_Disaster['text'].apply(lambda x: remove_username(x))
# Removing RT word
data_Disaster['text_RT'] = data_Disaster['text_nousername'].apply(lambda x: remove_RT(x))
# Removing links
data_Disaster['text_links'] = data_Disaster['text_RT'].apply(lambda x: remove_links(x))
# Removing Emojis
# data_Disaster['text_emoji'] = data_Disaster['text_links'].apply(lambda x: remove_emoji(x))

# Removing Punctuations
data_Disaster['text_nopunct'] = data_Disaster['text_links'].apply(lambda x: remove_punctuation(x))
# Get array of emojis
emojis_array = finding_emojis(data_Disaster.text_nopunct)
# Separating Emojis - to make each emoji as a single token
data_Disaster['text_emoji'] = data_Disaster['text_nopunct'].apply(lambda x: separate_emojis(x))
# Tokenizing
data_Disaster['text_tokens'] = data_Disaster['text_emoji'].apply(lambda x: tokenize(x))
# Remove stop words and make text lower case
data_Disaster['text_no_stop_words_tokens'] = data_Disaster['text_tokens'].apply(lambda x: remove_stop_words(x))
# Saving to pickle
save_to_pickle()



















