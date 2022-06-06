#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:57:50 2022

@author: motolab
"""

#%%
#import pyreadr
import pandas as pd
import math
import numpy as np
import gensim
from tqdm import tqdm
import matplotlib.pyplot as plt

data = pd.read_csv('strategies_final.csv')


#%%
import re
from tqdm import tqdm

def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('\\n', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


def normalize_text(text, lower=True):
    text = str(text)
    text = text_standardize(text)
    if lower:
        text = text.lower()
    return ' '.join(filter(None, (''.join(c for c in w if c.isalnum())
                                  for w in text.split())))
#%%
#abstract = [normalize_text(i) for i in tqdm(data.text_translated)]
abstract = [normalize_text(i) for i in tqdm(data.text_clean)]


#%%
from nltk.corpus import stopwords
exclude_punc = '!"#$%&\'()*+,-.。/:;<=>?@[\\]^_`{|}~、，；：'
stopword_list = stopwords.words('english')
#%%
def clean(text):
    return ' '.join([token for token in text.split() if (token not in exclude_punc) and (token not in stopword_list)])

#%%
abstract_cleaned = [clean(i) for i in tqdm(abstract)]
#%%
abstract_cleaned = [i.split() for i in tqdm(abstract_cleaned)]


#%%
import gensim

dictionary = gensim.corpora.Dictionary(abstract_cleaned)
len(dictionary)
#%%
doc_term_matrix = [dictionary.doc2bow(doc) for doc in abstract_cleaned]

model = gensim.models.ldamodel.LdaModel(corpus=doc_term_matrix, num_topics=14, id2word=dictionary, passes=10)

#%%
def find_optimal_number_of_topics(dictionary, corpus, texts, max_num, start = 2, step = 3, coherence = 'c_v', passes = 10):
    """
    dictionary: Gensim dictionary
    corpus: Gensim corpus / doc_term_matrix
    texts: documents
    max_num: maximum number of topics
    start:  number of topics to start
    step: search step
    
    -------
    Returns: A list of LDA topic models with corresponding coherence scores.
    
    """
    coherence_scores = []
    model_list = []
    for num_topics in range(start, max_num, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
        model_list.append(model)
        coherence_model = gensim.models.CoherenceModel(model = model, texts = texts, dictionary=dictionary, coherence=coherence)
        coherence_scores.append(coherence_model.get_coherence())
        
    return model_list, coherence_scores

#%%
max_num = 20
start = 5
step = 1
ldamodel_list, lda_coherence_scores = find_optimal_number_of_topics(dictionary=dictionary, corpus=doc_term_matrix, texts = abstract_cleaned, passes = 10, max_num=max_num, step = step)


#%%
# plot the coherence
start = 2
x = range(start, max_num, step)
plt.figure(figsize=(8, 6), dpi=80)
plt.plot(x, lda_coherence_scores)
plt.plot(x[np.argmax(lda_coherence_scores)], max(lda_coherence_scores), 'o', c = 'red', markersize = 4)
plt.xlabel('Num Topics')
plt.ylabel('Coherence score')
plt.title('LDA - coherence scores')
plt.xticks(np.arange(20), np.arange(20))
#plt.legend(("coherence_scores"), loc = 'best')
plt.savefig('lda_coherence_score.png')
plt.show()





#%%
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

# set Chinese font
font = r'Ubuntu-Regular.ttf'

for i in range(14):
    plt.figure(figsize=(10,10))
    plt.imshow(WordCloud(max_words=50, font_path=font, background_color='white').
               fit_words(dict(model.show_topic(i, 200))), interpolation='bilinear')
    plt.axis('off')
    #plt.title('Topic 8: Genomics',fontdict=dict(size=25))
    plt.title('Topic ' + str(i),fontdict=dict(size=25))
    plt.savefig('file_new/word_cloud_topic_'+ str(i) +'.png', bbox_inches='tight')

#%%
    
from gensim.test.utils import datapath
# Save model to disk.
temp_file = datapath("stip_model40")
model.save(temp_file)
#%%
# Load a potentially pretrained model from disk.
import gensim
from gensim.test.utils import datapath
temp_file = datapath("stip_model40")
model = gensim.models.ldamodel.LdaModel.load(temp_file)
    
#%%
import numpy as np
num_topics = 14
meaningful_topic_index = list(range(num_topics))
topic_percs = model[doc_term_matrix]


def charaVec(count_dict):
    count_dict = dict(count_dict)
    return [count_dict[i] if i in count_dict else 0 for i in meaningful_topic_index]


def Comp2Vec(coun_doc_ids):
    return np.array([charaVec(topic_percs[i]) for i in coun_doc_ids]).mean(0)


country_vectors = []
countries = list(set(data.country))

for i in countries:
    coun_doc_ids = list(data[data.country == i]['Unnamed: 0'])
    coun_doc_ids = [(i-1) for i in coun_doc_ids]
    country_vectors.append(Comp2Vec(coun_doc_ids))    
    
#%%
    
country_vectors_df = pd.DataFrame(country_vectors)
country_vectors_df['country'] = countries
country_vectors_df.to_csv('country_vectors_df.csv', index=False)
#%%

country_vectors_df = pd.read_csv('country_vectors_df.csv')

country_vectors = np.array(country_vectors_df.iloc[:, [2, 6, 7, 10, 11, 12, 13]])

from sklearn.metrics.pairwise import cosine_similarity

country_cs = cosine_similarity(np.array(country_vectors))


country_cs = pd.DataFrame(country_cs)
country_cs.to_csv('country_cs_7.csv', index=False)



