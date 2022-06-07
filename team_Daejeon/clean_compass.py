import pandas as pd
import spacy
import nltk
import pycountry

spacy.load("en_core_web_sm")
from spacy.lang.en import English

country_names = [country.name for country in pycountry.countries]

# Tokenize text
parser = English()


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        elif str(token) in country_names:
            pass
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


# Get meaning and root word
nltk.download('wordnet')
from nltk.corpus import wordnet as wn


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


from nltk.stem.wordnet import WordNetLemmatizer


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


# Filter stopwords
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

# Prepares text


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


# download the dataset from file
df = pd.read_csv("https://stip.oecd.org/assets/downloads/STIP_Survey.csv", sep='|', encoding='UTF-8-SIG',
                 header=0, low_memory=False)
df['ShortDescription'] = df['ShortDescription'].fillna("")

# Clean text and combine ShortDescription, Background, and Objectives 1-4
text_clean_arrays = []
for ind in df.index:
    background_tokens = []
    obj_tokens = []
    obj_tokens2 = []
    obj_tokens3 = []
    obj_tokens4 = []
    desc_tokens = prepare_text_for_lda(df['ShortDescription'][ind])
    if isinstance(df['Background'][ind], str):
        background_tokens = prepare_text_for_lda(df['Background'][ind])
    if isinstance(df['Objectives1'][ind], str):
        obj_tokens = prepare_text_for_lda(df['Objectives1'][ind])
    if isinstance(df['Objectives2'][ind], str):
        obj_tokens2 = prepare_text_for_lda(df['Objectives2'][ind])
    if isinstance(df['Objectives3'][ind], str):
        obj_tokens3 = prepare_text_for_lda(df['Objectives3'][ind])
    if isinstance(df['Objectives4'][ind], str):
        obj_tokens4 = prepare_text_for_lda(df['Objectives4'][ind])

    tokens = desc_tokens + background_tokens + \
        obj_tokens + obj_tokens2 + obj_tokens3 + obj_tokens4
    text_clean_arrays.append(tokens)

# Save it to a pkl file
df['text_clean'] = text_clean_arrays
df.to_pickle('compass_clean.pkl')
