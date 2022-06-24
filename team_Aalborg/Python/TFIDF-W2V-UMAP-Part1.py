#!/usr/bin/env python
# coding: utf-8

# # 1. TFIDF Weighted W2V Embeddings

# In[33]:


import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec

import json
import gensim
from nltk import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
stopwords = nltk.corpus.stopwords.words('english')

import h5py
import tqdm

import itertools
import gensim, logging
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gc
from multiprocessing import Pool
# from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.parsing.preprocessing import remove_stopwords


# In[34]:


def ele0(x):
    return x[0]
    
# Loads from JSON
def json_l(x):
    try:
        return json.loads(x)
    except ValueError:
        return []
        
# Preprocessing (sentence tokenizer + basik Gensim - lowercases, tokenizes, de-accents (optional))
def prepro(x):
    x = sent_tokenize(x)
    clean = []
    for j in x:
        clean.append(gensim.utils.simple_preprocess(j, min_len=1))
    return clean
    
def bigrammer(x):
    bigram = gensim.models.phrases.Phraser.load('bigram.m')
    sents = []
    for i in x:
        sents.append(bigram[i])
    return sents
    
def tfidfer(para):
    model_tfidf = gensim.models.TfidfModel.load('model_tfidf_11_2.m')
    docs_dict = Dictionary.load('docs_dict_11_2.d')
    return model_tfidf[docs_dict.doc2bow(itertools.chain(*para))] 

def preprocess_1 (data):
    data['appln_abstract_prepro'] = list(map(prepro, data.appln_abstract_st_r))
    data['appln_abstract_prepro'] = list(map(json.dumps, data.appln_abstract_prepro))
    return data
    
def preprocess_2 (data):
    data['no_json'] = list(map(json.loads, data['appln_abstract_prepro']))
    data['appln_abstract_prepro_bi'] = list(map(bigrammer, data['no_json']))
    data['appln_abstract_prepro_bi'] = list(map(json.dumps, data.appln_abstract_prepro_bi))
    return data

def preprocess_3 (data):
    data['no_json_tf'] = list(map(json.loads, data['appln_abstract_prepro_bi']))
    data['tfidf'] = list(map(tfidfer, data['no_json_tf']))
    data['tfidf'] = list(map(json.dumps, data['tfidf']))
    return data

def get_data_path(filename):
    data_dir = '/home/ubuntu/storage_data_new/python_scripts/STIP'
    path = os.path.join(data_dir, filename)
    if data_dir != '.' and 'DEEP_QUANT_ROOT' in os.environ:
        path = os.path.join(os.environ['DEEP_QUANT_ROOT'], path)
    return path 
    
def abstract_generator_nonmult(df):
    go = 'OK'
    while go == 'OK':
        try:
            i = df['appln_abstract_prepro_bi']
        except StopIteration:
                return
        if i.isnull().sum() > 0:
            i = list(filter(lambda a: a != None, i))
            go = 'STOP'
        if i.isnull().sum() < 0:
            i = [x[0] for x in i]
        j = [json_l(x) for x in i]
        for h in j:
            try:
                yield list(itertools.chain(*h))
            except StopIteration:
                return


# In[50]:


data = pd.read_csv('data.csv')
data.head(1)


# In[55]:


len_appln_abstract_st_r_list = []

for index, row in data.iterrows():
    len_appln_abstract_st_r_list.append(len(row.appln_abstract_st_r.split())) 
data['len_appln_abstract_st_r'] = len_appln_abstract_st_r_list


# In[38]:


tokens_without_sw = [remove_stopwords(text) for text in data.text_clean]
data ['appln_abstract_st_r'] = tokens_without_sw
len(tokens_without_sw)


# In[39]:


data_1 = preprocess_1(data)
data_1.to_csv('data.csv')


# In[44]:


sentences = Text8Corpus(get_data_path('data.csv'))


# In[41]:


first_sentence = next(iter(sentences))
                      
print(first_sentence[:10])


# In[45]:


bigram = Phrases(sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)


# In[46]:


bigram = gensim.models.phrases.Phraser(bigram)


# In[47]:


bigram.save('bigram.m')


# In[2]:


data_2 = preprocess_2 (data_1)


# In[49]:


data_2.to_csv('data.csv', index=False)


# In[59]:


from ast import literal_eval

docs_text_list = []
for itm in range(len(data['appln_abstract_prepro_bi'])):
    docs_text_list.append(literal_eval(data['appln_abstract_prepro_bi'][itm])[0])


# In[3]:


model_with_phrases = Word2Vec(sentences=docs_text_list, vector_size=300, window=80, min_count=1, workers=14)


# In[61]:


len(docs_text_list)


# In[62]:


model_with_phrases.save('w2v_313.m')


# In[63]:


voc = model_with_phrases.wv.index_to_key


# In[64]:


voc[:10]


# In[65]:


len(voc)


# In[66]:


# Build Gensim Dictionary
docs_dict = Dictionary([voc])
docs_dict.compactify()
docs_dict.save('docs_dict_11_2_313.d')


# In[67]:


model_tfidf = TfidfModel((docs_dict.doc2bow(x) for x in sentences), id2word=docs_dict)
model_tfidf.save('model_tfidf_11_2_313.m')


# In[4]:


data_3 = preprocess_3 (data_2)
data_3.to_csv('data.csv', index=False)
iterator = abstract_generator_nonmult(data_3)


# In[69]:


# Document matrix TF-IDF weighted
docs_vecs = (sparse2full(c, len(docs_dict)) for c in ((model_tfidf[docs_dict.doc2bow(x)] for x in iterator)))


# In[70]:


# Selected Word-Embeddings
emb_vecs_selftrained = np.vstack([model_with_phrases.wv[docs_dict.get(i)] for i in range(len(docs_dict))])
n_abstracts = len(data_2)
h5f = h5py.File('docvecs_23_4_test_313.h5', 'a')
dataset = h5f.create_dataset('weighted_tfidf_docvecs', (n_abstracts,300))


# In[71]:


# Generates document-vectors
pbar = tqdm.tqdm(total=n_abstracts)
start = 0
#while docs_vecs:
for i in range(1):
    a = np.vstack(next(docs_vecs) for _ in range(n_abstracts))
    pbar.update(n_abstracts)
    b = np.dot(a,emb_vecs_selftrained)
    end = start + len(b)
    dataset[start:end] = b
    start = end
pbar.close()
h5f.close()


# In[5]:


h5f = h5py.File('docvecs_23_4_test_313.h5', 'r')
dataset = h5f['weighted_tfidf_docvecs']
dataset[1]


# In[73]:


len(model_with_phrases.wv.index_to_key)


# In[75]:


dataset.shape


# # 2. Using UMAP for Clustering

# In[76]:


data_nar =  np.array(dataset)


# In[78]:


# !pip install umap-learn -i https://mirrors.ustc.edu.cn/pypi/web/simple

import umap
import umap.umap_ as umap
import hdbscan

umap_reducer_sci = umap.UMAP(random_state=42, n_components=2)
embeddings_sci = umap_reducer_sci.fit_transform(data_nar)


# In[131]:


clusterer_sci = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
clusterer_sci.fit(embeddings_sci)
data['cluster'] = clusterer_sci.labels_


# In[6]:


# data.cluster.value_counts()


# In[7]:


df_plot = pd.DataFrame(embeddings_sci, columns=['x','y'])
df_plot


# In[134]:


df_plot['Title'] = data['title']
df_plot['text_original'] = data['text_original']
df_plot['text_clean'] = data['text_clean']
df_plot['language'] = data['language']
df_plot['year'] = data['year']
df_plot['country'] = data['country']
df_plot['cluster'] = clusterer_sci.labels_


# In[135]:


df_plot = df_plot[df_plot['cluster']!= -1]


# In[136]:


df_plot.shape


# In[137]:


import altair as alt
alt.data_transformers.enable(max_rows=None)


# In[138]:


alt.Chart(df_plot).mark_circle(size=60).encode(
    x='x',
    y='y',
    color=alt.Color('cluster', scale=alt.Scale(scheme='category20')),
    tooltip=['Title', 'text_original', 'text_clean', 'language', 'year', 'country', 'cluster']
).properties(
    width=800,
    height=600
).interactive()


# In[139]:


data.head()


# In[140]:


df_community_100 = pd.DataFrame(data.cluster.value_counts())
df_community_100 = df_community_100.reset_index()
df_community_100


# In[141]:


# Dataset
from sklearn.datasets import fetch_20newsgroups
# Data manipulation
import numpy as np
import pandas as pd
from collections import defaultdict
pd.options.display.max_colwidth = 100
# Text preprocessing and modelling
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import train_test_split
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='talk')
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
# Warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
# Stopwords
stop_words = set(STOPWORDS).union(stopwords.words("english"))
stop_words = stop_words.union(['let', 'mayn', 'ought', 'oughtn', 
                               'shall'])
print(f"Number of stop words: {len(stop_words)}")


# In[143]:


import pandas as pd
import matplotlib.pyplot as plt

df_topic = pd.DataFrame(data.cluster.value_counts()[5:])
df_topic = df_topic.reset_index()

fig, ax = plt.subplots(1, len(df_topic), figsize=(20, 8))
for i in range(len(df_topic)):
    topic = df_topic['index'][i]
    text = ' '.join(data.loc[data['cluster']==topic, 'appln_abstract_prepro_bi'].values)    
    wordcloud = WordCloud(width=1000, height=1000, random_state=1, background_color='White', 
                          colormap='Set2', collocations=False, stopwords=stop_words).generate(text)
    ax[i].imshow(wordcloud) 
    ax[i].set_title(f"Cluster {topic}")
    # No axis details
    ax[i].axis("off");


# In[83]:


result = len(df.text_original[0].split())
result


# In[158]:


df = df.reset_index()
df.head(2)


# In[162]:


df_community_new = pd.merge(df, df_community, how='inner', left_on='index', right_on='query_paper')


# In[8]:


# df_community_new


# In[165]:


# df_community_new.to_excel('df_community_new_313_75.xlsx')
# df_community_new.to_csv('df_community_new_313_75.csv')


# In[6]:


import pandas as pd

df_community_new = pd.read_csv('df_community_new_313_75.csv')
df_community_new.head()


# In[7]:


list_new_stopwords = ['taking', 'account', 'john wiley', 'wiley sons', 'sons ltd', 'emerald publishing', 'elsevier', 'accompany', 'according', 'advantage', 'also', 'apparatus', 'application', 'based', 'belong', 'benefit', 'list', 'cliam', 'column', 'common', 'comprise', 'concern', 'copyright', 'describe', 'diagram', 'document', 'embodiment', 'example', 'exist', 'fig', 'figure', 'follow', 'formula', 'include', 'identify', 'invention', 'investigate', 'involve', 'kind', 'large', 'least', 'literature', 'lower', 'method', 'module', 'number', 'obtain', 'one', 'perform', 'prepare', 'present', 'prior', 'problem', 'propose', 'purpose', 'refer', 'reference', 'related', 'represent', 'require', 'result', 'row', 'said', 'schematic', 'score', 'select', 'solve', 'technical', 'technology', 'through', 'total', 'two', 'upper', 'use', 'utility', 'wherein', 'whether']


# In[8]:


import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')

# stopwords_default = stopwords.words('english')
print(len(stop_words))

# for adding multiple words
stop_words = stop_words.union(list_new_stopwords)
print(len(stop_words))


# In[9]:


def preprocess_text(text, stop_words, pos_to_keep=None):
    """Preprocess document into normalised tokens."""
    # Tokenise into alphabetic tokens with minimum length of 3
    tokeniser = RegexpTokenizer(r'[A-Za-z]{3,}')
    tokens = tokeniser.tokenize(text)
    
    # Lowercase and tag words with POS tag
    tokens_lower = [token.lower() for token in tokens]
    pos_map = {'J': 'a', 'N': 'n', 'R': 'r', 'V': 'v'}
    pos_tags = pos_tag(tokens_lower)
    
    # Keep tokens with relevant pos
    if pos_to_keep is not None:
        pos_tags =  [token for token in pos_tags if token[1][0] in pos_to_keep]  
    
    # Lemmatise 
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(t, pos=pos_map.get(p[0], 'v')) for t, p in pos_tags]
    
    # Remove stopwords
    keywords= [lemma for lemma in lemmas if lemma not in stop_words]
    return keywords
# documents = [preprocess_text(document, stop_words) for document in example]
# documents


# In[10]:


# documents_pre = [preprocess_text(document, stop_words) for document in df_community_new['text_original']]


# In[177]:


import pandas as pd
import matplotlib.pyplot as plt

df_community_100 = pd.DataFrame(df_community_new.paper_community.value_counts()[:12])
df_community_100 = df_community_100.reset_index()

fig, ax = plt.subplots(1, len(df_community_100), figsize=(20, 8))
for i in range(len(df_community_100)):
    topic = df_community_100['index'][i]
    text = ' '.join(df_community_new.loc[df_community_new['paper_community']==topic, 'text_clean'].values)    
    wordcloud = WordCloud(width=1000, height=1000, random_state=1, background_color='White', 
                          colormap='Set2', collocations=False, stopwords=stop_words).generate(text)
    ax[i].imshow(wordcloud) 
    ax[i].set_title(f"Community {topic}")
    # No axis details
    ax[i].axis("off");


# In[201]:


import pandas as pd
import matplotlib.pyplot as plt

df_community_100 = pd.DataFrame(df_community_new.paper_community.value_counts()[:6])
df_community_100 = df_community_100.reset_index()

fig, ax = plt.subplots(1, len(df_community_100), figsize=(20, 8))
for i in range(len(df_community_100)):
    topic = df_community_100['index'][i]
    text = ' '.join(df_community_new.loc[df_community_new['paper_community']==topic, 'text_clean'].values)    
    wordcloud = WordCloud(width=1000, height=1000, random_state=1, background_color='White', 
                          colormap='Set2', collocation_threshold = 45, stopwords=stop_words).generate(text)
    ax[i].imshow(wordcloud) 
    ax[i].set_title(f"Community {topic}")
    # No axis details
    ax[i].axis("off");


# In[26]:


df_community_new.paper_community.value_counts().nlargest(15)


# # 2.1 KT and KCC Keywords analysis

# In[11]:


keywords_both = ['invention', 'patenting', 'product_development', 'product', 'development', 'user producer','user driven', 'joint', 'partnership', 'co-creation', 'cooperation', 'value_chain', 'commercialisation', 'co-patenting', 'co-invention', 'co-investment', 'triangle', 'product_development', 'triple helix', 'user-producer', 'user-driven', 'transfer', 'contract', 'right', 'diffusion', 'flows', 'sharing', 'spillover', 'trade_secrets', 'trade secrets', 'trade', 'secrets', 'trademark', 'science-industry links', 'science', 'industry', 'links', 'science-industry_links', 'spin-off', 'spin off', 'access', 'adopt', 'absorptive capacity', 'absorptive', 'capacity', 'absorptive_capacity', 'licens', 'tacit', 'vocational skills', 'vocational_skills', 'vocational', 'skills', 'mobility', 'brain gain', 'brain_gain', 'brain', 'gain', 'brain_drain', 'brain drain', 'drain']


# In[27]:


communities_largest = [6, 15, 12, 7, 2, 14, 13, 0, 4, 24, 10, 30, 31, 19, 20]


# In[28]:


communities_largest_cnt = df_community_new.paper_community.value_counts().nlargest(15).to_list()


# In[29]:


list_list_topic_both = []
for i in range(len(communities_largest)):
    topic = communities_largest[i]
    text = ' '.join(df_community_new.loc[df_community_new['paper_community']==topic, 'text_clean'].values)    
    nltk_tokens = nltk.word_tokenize(text)
    data_analysis = nltk.FreqDist(nltk_tokens)
    # Let's take the specific words only if their frequency is greater than 3.
    filter_words = dict([(m, n) for m, n in data_analysis.items() if len(m) > 3])
    print('\033[1mtopic\033[0m', topic)
    list_topic = []
    for word in keywords_both:
        if word in filter_words:
            list_topic.append(filter_words[word])
#             print("%s: %s" % (word, filter_words[word]))
        else:
            list_topic.append(0)  
#             print('not found: ', word)
    list_list_topic_both.append(list_topic)


# In[30]:


df_topics_keywords = pd.DataFrame()

for item in range(len(communities_largest)):
    df_topics_keywords[communities_largest[item]] = list_list_topic_both[item]


# In[31]:


df_topics_keywords_norm = pd.DataFrame()
for item in range(len(communities_largest)):
    df_topics_keywords_norm[communities_largest[item]] = [value / communities_largest_cnt[item] for value in list_list_topic_both[item]]
   


# In[32]:


df_topics_keywords['keywords'] = keywords_both
df_topics_keywords_norm['keywords'] = keywords_both


# In[34]:


df_topics_keywords = df_topics_keywords[['keywords', 6, 15, 12, 7, 2, 14, 13, 0, 4, 24, 10, 30, 31, 19, 20]]
df_topics_keywords_norm = df_topics_keywords_norm[['keywords', 6, 15, 12, 7, 2, 14, 13, 0, 4, 24, 10, 30, 31, 19, 20]]


# In[35]:


df_topics_keywords


# In[46]:


df_topics_keywords.iloc[:21, :2]


# In[48]:


df_topics_knowledge_creation_final = pd.DataFrame()

df_topics_keywords_creation_final['co_creation'] = df_topics_keywords.iloc[:21, 1:2].sum()


# In[83]:


df_topics_knowledge_creation_final = pd.DataFrame()

for i in range(len(communities_largest)):
    df_topics_knowledge_creation_final[str(communities_largest[i])] = df_topics_keywords.iloc[:21, (1+i):(2+i)].sum()


# In[91]:


df_topics_knowledge_creation_final = pd.DataFrame()

df_topics_knowledge_creation_final['6'] = df_topics_keywords.iloc[:21, (1):(2)].sum()
df_topics_knowledge_creation_final['15'] = df_topics_keywords.iloc[:21, (2):(3)].sum()
df_topics_knowledge_creation_final['12'] = df_topics_keywords.iloc[:21, (3):(4)].sum()
df_topics_knowledge_creation_final['7'] = df_topics_keywords.iloc[:21, (4):(5)].sum()
df_topics_knowledge_creation_final['2'] = df_topics_keywords.iloc[:21, (5):(6)].sum()
df_topics_knowledge_creation_final['14'] = df_topics_keywords.iloc[:21, (6):(7)].sum()
df_topics_knowledge_creation_final['13'] = df_topics_keywords.iloc[:21, (7):(8)].sum()
df_topics_knowledge_creation_final['0'] = df_topics_keywords.iloc[:21, (8):(9)].sum()
df_topics_knowledge_creation_final['4'] = df_topics_keywords.iloc[:21, (9):(10)].sum()
df_topics_knowledge_creation_final['24'] = df_topics_keywords.iloc[:21, (10):(11)].sum()
df_topics_knowledge_creation_final['10'] = df_topics_keywords.iloc[:21, (11):(12)].sum()
df_topics_knowledge_creation_final['30'] = df_topics_keywords.iloc[:21, (12):(13)].sum()
df_topics_knowledge_creation_final['31'] = df_topics_keywords.iloc[:21, (13):(14)].sum()
df_topics_knowledge_creation_final['19'] = df_topics_keywords.iloc[:21, (14):(15)].sum()
df_topics_knowledge_creation_final['20'] = df_topics_keywords.iloc[:21, (15):(16)].sum()


# In[93]:


df_topics_knowledge_creation_final['15'] = df_topics_keywords.iloc[:21, (2):(3)].sum()


# In[108]:


df_topics_keywords.iloc[:21, (1):(2)].sum()


# In[95]:


df_topics_knowledge_creation_final['15'] = 1


# In[96]:


df_topics_knowledge_creation_final


# In[197]:


def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]


# In[199]:


import pandas as pd
import matplotlib.pyplot as plt
import re

df_community_100 = pd.DataFrame(df_community_new.paper_community.value_counts()[:1])
df_community_100 = df_community_100.reset_index()

fig, ax = plt.subplots(1, len(df_community_100), figsize=(20, 8))
for i in range(len(df_community_100)):
    topic = df_community_100['index'][i]
    text = ' '.join(df_community_new.loc[df_community_new['paper_community']==topic, 'text_clean'].values)
    true_word = basic_clean(text)
    true_bigrams_series = (pd.Series(nltk.ngrams(true_word, 2)).value_counts())[:20]
    true_bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
    plt.title('20 Most Frequently Occuring Bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# of Occurances')


# In[180]:


import pandas as pd
import matplotlib.pyplot as plt

df_community_100 = pd.DataFrame(df_community_new.paper_community.value_counts()[:12])
df_community_100 = df_community_100.reset_index()

fig, ax = plt.subplots(1, len(df_community_100), figsize=(20, 8))
for i in range(len(df_community_100)):
    topic = df_community_100['index'][i]
    text = ' '.join(df_community_new.loc[df_community_new['paper_community']==topic, 'text_clean'].values)    
    wordcloud = WordCloud(width=1000, height=1000, random_state=1, background_color='White', 
                          colormap='Set2', collocations=False, stopwords=stop_words).generate(text)
    ax[i].imshow(wordcloud) 
    ax[i].set_title(f"Community {topic}")
    # No axis details
    ax[i].axis("off");


# In[184]:


# def inspect_term_frequency(corpus, n=30):
#     """Show top n frequent terms in corpus."""
#     # Preprocess text
#     tokens = [preprocess_text(document, stop_words) for document in corpus]
#     corpus = [id2word.doc2bow(document) for document in tokens]
    
#     # Find term frequencies
#     frequency = defaultdict(lambda: 0)
#     for document in corpus:
#         for codeframe, count in document:
#             frequency[codeframe] += count        
#     frequency_list = [(codeframe, count) for codeframe, count in frequency.items()]
#     frequency_list.sort(key=lambda x: x[1], reverse=True)
#     codeframe_lookup = {value:key for key, value in id2word.token2id.items()}
#     data = {codeframe_lookup[codeframe]: count for codeframe, count in frequency_list[:n]}
#     return pd.DataFrame(pd.Series(data), columns=['frequency'])
# fig, ax = plt.subplots(1, 6, figsize=(16,12))
# for i in range(len(df_community_100)):
#     topic = 'topic' + str(i+1)
#     topic_df = df_community_new.loc[df_community_new['paper_community']==topic, 'text_clean']
#     frequency = inspect_term_frequency(topic_df)
#     sns.barplot(data=frequency, x='frequency', y=frequency.index, ax=ax[i])
#     ax[i].set_title(f"Topic {i+1} - Top words")
# plt.tight_layout()


# In[ ]:




