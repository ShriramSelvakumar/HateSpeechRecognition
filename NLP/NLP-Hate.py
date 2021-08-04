import pandas as pd
import re

# Setting display width for panda outputs
pd.set_option('display.max_colwidth', 200)
# Reading Hate Speech data
data_Hate = pd.read_csv('../Data/HS_DATA.csv', sep=',', index_col=0)


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
    # Punctuations without '
    punctuations = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
    # punctuations = '!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~'
    # Remove punctuations
    return_text = " ".join(re.split('['+punctuations+']+', text))
    # sometimes ' connect words - "I'm" - below line outputs - "Im"
    return_text = "".join(re.split('[\']+', return_text))
    # return_text = "".join(re.split('[’]+', return_text))
    return_text = "".join(re.split('[’…“”—⁣️]+', return_text))

    # return_text = " ".join(re.split(r'[.]+/^(?!.*[0-9][.]).*/', return_text))
    # return_text = " ".join([char for char in text if char not in string.punctuation])
    return return_text


# Function to remove links
def remove_links(text):
    # return_text = " ".join(re.split(r'http\S+', text))
    return_text = " ".join(re.split(r'http[a-zA-Z0-9.\/:]+|https[a-zA-Z0-9.\/:&]+', text))
    return return_text


# Function to remove emoji strings
def remove_emoji(text):
    return_text = " ".join(re.split('[&#][0-9]+', text))
    return return_text


# Function to Tokenize
def tokenize(text):
    # tokens = " ".join(re.findall('[\w]+', text))
    tokens = text.split()
    return tokens


# Removing @usernames
data_Hate['text_nousername'] = data_Hate['text'].apply(lambda x: remove_username(x))
# Removing RT word
data_Hate['text_RT'] = data_Hate['text_nousername'].apply(lambda x: remove_RT(x))
# Removing links
data_Hate['text_links'] = data_Hate['text_RT'].apply(lambda x: remove_links(x))
# Removing Emojis
# data_Hate['text_emoji'] = data_Hate['text_links'].apply(lambda x: remove_emoji(x))
# Removing Punctuations
data_Hate['text_nopunct'] = data_Hate['text_links'].apply(lambda x: remove_punctuation(x))
# Tokenizing
data_Hate['text_tokens'] = data_Hate['text_nopunct'].apply(lambda x: tokenize(x))




















