#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 22:00:40 2022

@author: motolab
"""



#%%


import pandas as pd
from tqdm import tqdm
import numpy as np

data = pd.read_csv('strategies_final.csv')


strategies_set = ['strategy', 'plan', 'agenda', 'policy', 'program']
goal_set = ['goal', 'directionality', 'aim', 'target', 'purpose', 'objective', 'vision']
timeline_set = ['milestone', 'loadmap']
budget_set = ['budget', 'fund', 'grant', 'investment', 'budget allocation']
action_set =  ['implement', 'execut', 'act']
governance_set = ['monitor', 'foresight', 'impact assessment', 
                  'policy intelligence', 'evaluation', 'policy coordination', 'feedback', 'lessons']


criterion = [strategies_set, goal_set, timeline_set, budget_set, action_set, governance_set]

text = data.text_translated[1]

def keywordcount(text, keywords):
    return np.sum([text.count(word) for word in keywords])

def keywordfreq(text, keywords):
    total_len = len(text.split())
    return np.sum([text.count(word)/total_len for word in keywords])


kwdcount_feat = []
kwdfreq_feat = []

for text in tqdm(data.text_translated):
    kwdcount_feat.append([keywordcount(text, keywords) for keywords in criterion])
    kwdfreq_feat.append([keywordfreq(text, keywords) for keywords in criterion])

#%%
kwdcount_feat = pd.DataFrame(np.array(kwdcount_feat), columns=['strategies', 'goal', 'timeline', 'budget', 'action', 'governace'])
kwdfreq_feat = pd.DataFrame(np.array(kwdfreq_feat),  columns=['strategies', 'goal', 'timeline', 'budget', 'action', 'governace'])
#%%
kwdcount_feat['country'] = data.country
kwdfreq_feat['country'] = data.country

#%%

kwdfreq_feat_agg = kwdfreq_feat.groupby('country').mean().reset_index()
kwdcount_feat_agg = kwdcount_feat.groupby('country').mean().reset_index()


kwdfreq_feat_agg.to_csv('kwdfreq_feat_agg.csv', index=False)
kwdcount_feat_agg.to_csv('kwdcount_feat_agg.csv', index=False)
