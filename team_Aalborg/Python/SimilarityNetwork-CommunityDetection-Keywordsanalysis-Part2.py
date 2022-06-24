#!/usr/bin/env python
# coding: utf-8

# # 3. Using TFIDF Weighted W2V Embeddings for Network analysis

# In[12]:


import pickle
import h5py
import numpy as np
#Store sentences & embeddings on disc
# with open('embeddings_313.pkl', "wb") as fOut:
#     pickle.dump({'id': df['doc_id'], 'sentences': df['text_original'], 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#Load sentences & embeddings from disc
# with open('embeddings_313.pkl', "rb") as fIn:
#     stored_data = pickle.load(fIn)
#     stored_id = stored_data['id']
#     stored_sentences = stored_data['sentences']
#     stored_embeddings = stored_data['embeddings']


# In[13]:


h5f = h5py.File('docvecs_23_4_test_313.h5', 'r')
embeddings = h5f['weighted_tfidf_docvecs']
embeddings.shape


# In[14]:


embeddings =  np.array(embeddings)


# In[15]:


# Python program to get average of a list
def Average(lst):
    return sum(lst) / len(lst)


# In[1]:


from sentence_transformers import SentenceTransformer, util
import pandas as pd
import scipy
# query = input("Please enter a new company description: ")
# query = 'News, off-beat stories and analysis of German and international affairs.'
df_nodes_papers = pd.DataFrame()

# top_n = len(table_test_2.description)
top_n = 313

for expid in range(len(embeddings)):
    query_embedding = embeddings[expid]
    correct_hits = util.semantic_search(query_embedding, embeddings, top_k=top_n)[0]
    correct_hits_ids = list([hit['corpus_id'] for hit in correct_hits])
    correct_hits_score = list([hit['score'] for hit in correct_hits])
    average = Average(correct_hits_score)
    query_paper = []
    average_sim = []
    
    for i in range(top_n):
        query_paper.append(expid)
        average_sim.append(average)

    top_5_similar_paper_df = pd.DataFrame({
        'query_paper': query_paper,
        'top_papers_id_all': correct_hits_ids,
        'cosine_similarity': correct_hits_score,
        'average_cosine_similarity': average_sim,

    })
    
    df_nodes_papers = df_nodes_papers.append(top_5_similar_paper_df)


# In[2]:


# df_nodes_papers


# In[18]:


df_80 = df_nodes_papers[df_nodes_papers['cosine_similarity'] > 0.75]


# In[19]:


# df_80[df_80['cosine_similarity']>0.9999][df_80[df_80['cosine_similarity']>0.9999]['query_paper'] != df_80[df_80['cosine_similarity']>0.9999]['query_paper']] 


# In[20]:


df_80_1 = df_80[df_80['cosine_similarity'] < 0.9999]


# In[80]:


# df_80_1.to_csv('df_75_1.csv', index=False)


# In[21]:


import community
import networkx as nx
# make the networkx graph object: directed graph
g = nx.from_pandas_edgelist(df_80_1, 'query_paper', 
                            'top_papers_id_all', 
                            edge_attr='cosine_similarity')
len(g.nodes())


# In[22]:


#first compute the best partition
partition_object = community.best_partition(g)


# In[23]:


import pickle
with open("partition_object_list", "wb") as fp:   #Pickling
    pickle.dump(partition_object, fp)


# In[24]:


# first elements of the partition dictionary
# this is a mapping between paper and community
list(partition_object.items())[0:10]


# In[25]:


# extract the communities for each album 
values = [partition_object.get(node) for node in g.nodes()]


# In[26]:


len(values)


# In[27]:


color_list = ["#019c3e",
"#9113a2", "#53e063", "#543abc", "#85d944", "#7f51d9", "#70c32b", "#5d5be3", "#b5ce1c", "#8471fd", "#61b10f", "#a75de6", "#49b529", "#b046cb",
"#27ca53", "#ae0099", "#01d069", "#c53abe", "#009d1c", "#ef68ea", "#7fdc57", "#a872fe", "#a6d636", "#014bc2", "#d5ca17", "#0063df",
"#7cb000", "#6b79ff", "#9fb300", "#6437af", "#b3d342", "#7532a0", "#63de79", "#e649c6", "#01b351", "#ff67e3", "#007f11", "#ee33a9",
"#00cc83", "#cd008f", "#00b96a", "#ee2599", "#2ae0a2", "#fb2e95", "#8bda69", "#b70088", "#69a000", "#ff87fc", "#106c00", "#b987ff",
"#e2c52d", "#0285f7", "#fcb113", "#0092fe", "#dea500", "#0159bd", "#fcbb3a", "#006cd0", "#e59500", "#6e8bff", "#b4a000",
"#3e48aa", "#c2cf4e", "#822b92", "#a2d662", "#9f007a", "#93d878", "#8e2185", "#538600",
"#ec92ff", "#008b3c", "#ff3a8f", "#00a667", "#d0006c", "#00dac4", "#dd2231", "#29d9ed", "#b51600", "#3edbd1", "#d82d23", "#01c6a9",
"#c3002c", "#46dbc9", "#f74a3b", "#03c8c1", "#e04022", "#45b0ff", "#d85102", "#0278d3", "#ff9424", "#0279c8", "#e58400", "#8899ff",
"#fa7720", "#018cd1", "#ed5227", "#00bcbb", "#a41707", "#5dd8db", "#a70a1b", "#6adba4", "#bd006f", "#8cd88b", "#c9005d", "#01aa7d",
"#bf004a", "#68d9bf", "#a8042e", "#87d89c", "#8c277f", "#688600", "#b299ff", "#c09100", "#88abff", "#c98300", "#016aa6",
"#edc143", "#78378a", "#d8c85a", "#6f3e85", "#ae8e00", "#dba0ff", "#3e6f00", "#ff6ec4", "#017027", "#ff5ca1", "#206003",
"#ff95e1", "#006025", "#ff5584", "#017c48", "#ff5375", "#02a694", "#ff5958","#01afb8","#be4100","#0092c8","#c46200","#0096c2","#b24c00",
"#00adc6","#b05800","#b2a8ff","#958800","#d8aeff","#627200","#e4afff","#195f22","#ffaaf4","#006131","#ff8cca","#425b08","#ccb3ff",
"#bd7200","#a7bdff","#983b00","#017da8","#ffb554","#5d468a","#b8d076","#873076","#7ed8b0","#991c63","#017f5d","#ff6672","#00896d",
"#ff7b9c","#375c1e","#ecb3f0","#757100","#4b4d89","#ecc065","#2f5386","#ff9054","#018a84","#ff8164","#73cbbc","#9d1f3f","#a8d19b",
"#98264c","#bbce8b","#8b3161","#ddc573","#6e4276","#9a7b00","#cbbef9","#a16e00","#9a9ed5","#9e6300","#6e7eb2","#ff9a5a","#357f65",
"#ff7d79","#54865e","#ff89b0","#4d5801","#fdaee2","#6c5b00","#feb0d0","#4e571e","#ff8b9a","#728a57","#7b3d6d","#fbba69",
"#886b9b","#856200","#ffabb6","#5d5317","#ff9fa5","#6e4c10","#ff8c89","#94a771","#9a282c","#d9c587","#8e3343","#e8c07d","#955672",
"#f6bb71","#813e47","#ffad72","#844540","#f2bc88","#8d381b","#c3ac77","#853f03","#b36e82","#865100","#eca499","#7c4516","#feb591",
"#704b26","#ff8d61","#7a452c","#ffa891","#843f21","#ff967b","#823f33","#bc9268","#a86a59","#907144","#c98e73"]


# In[28]:


# the community algorithm creates around 241 communities
# (there is some randomness to the algorithm)
# we select as many colors as there are communities:
color_list = color_list[0:len(set(values))]
len(color_list)


# In[29]:


import numpy as np
# and we make a dictionary where we map each
# community number to a specific color code
color_dict = pd.Series(color_list,
        index=np.arange(0,len(set(values)))).to_dict()


# In[30]:


# use the color dictionary to update the partition object:
# we replace the number of the community with the
# color hex code
for key, value in partition_object.items():
    partition_object[key] = color_dict[value]


# In[31]:


# set the node attribute color in networkx object
# using the above mapping
nx.set_node_attributes(g, partition_object, 'color')


# In[32]:


# https://gist.github.com/maciejkos/e3bc958aac9e7a245dddff8d86058e17
def draw_graph3(networkx_graph,notebook=True,output_filename='graph.html',show_buttons=True,only_physics_buttons=False,
                height=None,width=None,bgcolor=None,font_color=None,pyvis_options=None):
    """
    This function accepts a networkx graph object,
    converts it to a pyvis network object preserving its node and edge attributes,
    and both returns and saves a dynamic network visualization.
    Valid node attributes include:
        "size", "value", "title", "x", "y", "label", "color".
        (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_node)
    Valid edge attributes include:
        "arrowStrikethrough", "hidden", "physics", "title", "value", "width"
        (For more info: https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.network.Network.add_edge)
    Args:
        networkx_graph: The graph to convert and display
        notebook: Display in Jupyter?
        output_filename: Where to save the converted network
        show_buttons: Show buttons in saved version of network?
        only_physics_buttons: Show only buttons controlling physics of network?
        height: height in px or %, e.g, "750px" or "100%
        width: width in px or %, e.g, "750px" or "100%
        bgcolor: background color, e.g., "black" or "#222222"
        font_color: font color,  e.g., "black" or "#222222"
        pyvis_options: provide pyvis-specific options (https://pyvis.readthedocs.io/en/latest/documentation.html#pyvis.options.Options.set)
    """

    # import
    from pyvis import network as net

    # make a pyvis network
    network_class_parameters = {"notebook": notebook, "height": height, "width": width, "bgcolor": bgcolor, "font_color": font_color}
    pyvis_graph = net.Network(**{parameter_name: parameter_value for parameter_name, parameter_value in network_class_parameters.items() if parameter_value})

    # for each node and its attributes in the networkx graph
    for node,node_attrs in networkx_graph.nodes(data=True):
        pyvis_graph.add_node(node,**node_attrs)

    # for each edge and its attributes in the networkx graph
    for source,target,edge_attrs in networkx_graph.edges(data=True):
        # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
        if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
            # place at key 'value' the weight of the edge
            edge_attrs['value']=edge_attrs['weight']
        # add the edge
        pyvis_graph.add_edge(source,target,**edge_attrs)

    # turn buttons on
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons(filter_=['physics'])
        else:
            pyvis_graph.show_buttons()

    # pyvis-specific options
    if pyvis_options:
        pyvis_graph.set_options(pyvis_options)

    # return and also save
    return pyvis_graph.show(output_filename)


# In[3]:


# g.nodes(data=True)


# In[33]:


# make the pyviz interactive plot
# this will save out an html file to the directory
# where this script is
# plot will also be shown in the notebook
draw_graph3(g, height = '1000px', width = '1000px', 
            show_buttons=False,  
            output_filename='graph_output_communities_313_75.html', notebook=True)


# In[4]:


# g.nodes


# In[5]:


degree_centrality_node_list = []
degree_centrality_score_list = []


for node in g.nodes():
    print(node, nx.degree_centrality(g)[node])
    degree_centrality_node_list.append(node)
    degree_centrality_score_list.append(nx.degree_centrality(g)[node])


# In[36]:


d = {'node':degree_centrality_node_list,'score':degree_centrality_score_list}
df_degree_centrality = pd.DataFrame(d)
df_degree_centrality


# In[37]:


df_degree_centrality.to_csv('df_degree_centrality_community.csv', index=False)


# In[6]:


closeness_centrality_node_list = []
closeness_centrality_score_list = []

for node in g.nodes():
    print(node, nx.closeness_centrality(g, node))
    closeness_centrality_node_list.append(node)
    closeness_centrality_score_list.append(nx.closeness_centrality(g, node))


# In[7]:


d = {'node':closeness_centrality_node_list,'score':closeness_centrality_score_list}
df_closeness_centrality = pd.DataFrame(d)
df_closeness_centrality


# In[40]:


df_closeness_centrality.to_csv('df_closeness_centrality_community.csv', index=False)


# In[9]:


eigenvector_centrality_node_list = []
eigenvector_centrality_score_list = []

for node in g.nodes(): 
    print(node, nx.eigenvector_centrality(g, max_iter=1000)[node])
    eigenvector_centrality_node_list.append(node)
    eigenvector_centrality_score_list.append(nx.eigenvector_centrality(g, max_iter=1000)[node])


# In[10]:


d = {'node':eigenvector_centrality_node_list,'score':eigenvector_centrality_score_list}
df_eigenvector_centrality = pd.DataFrame(d)
df_eigenvector_centrality


# In[43]:


df_eigenvector_centrality.to_csv('df_eigenvector_centrality_community.csv', index=False)


# In[11]:


betweenness_centrality_node_list = []
betweenness_centrality_score_list = []

for node in g.nodes(): 
    print(node, nx.betweenness_centrality(g)[node])
    betweenness_centrality_node_list.append(node)
    betweenness_centrality_score_list.append(nx.betweenness_centrality(g)[node])


# In[12]:


d = {'node':betweenness_centrality_node_list,'score':betweenness_centrality_score_list}
df_betweenness_centrality = pd.DataFrame(d)
df_betweenness_centrality


# In[46]:


df_betweenness_centrality.to_csv('df_betweenness_centrality_community.csv', index=False)


# In[47]:


with open("partition_object_list", "rb") as fp:   # Unpickling
       partition_object_list = pickle.load(fp)


# In[48]:


paper_node_list = []

for item in range(len(list(partition_object_list.items()))):
    paper_node_list.append(list(partition_object_list.items())[item][0])


# In[49]:


paper_community_list = []

for item in range(len(list(partition_object_list.items()))):
    paper_community_list.append(list(partition_object_list.items())[item][1])


# In[50]:


d = {'paper_node':paper_node_list,'paper_community':paper_community_list}


# In[13]:


df_paper_community = pd.DataFrame(d)
df_paper_community


# In[52]:


df_community = pd.merge(df_80_1, df_paper_community, how='inner', left_on='query_paper', right_on='paper_node')


# In[14]:


df_community


# In[54]:


df_community.paper_community.value_counts()


# In[56]:


# df_community.to_excel('df_community_313_75.xlsx')
# df_community.to_csv('df_community_313_75.csv')


# In[15]:


df_community = pd.read_csv('df_community_313_75.csv')
df_community


# In[16]:


df_community_num = df_community[['query_paper', 'paper_community']]
df_community_num


# In[90]:


df_community_num = df_community_num.drop_duplicates(subset='query_paper', keep="last")


# In[58]:


import pandas as pd

df = pd.read_csv('/home/ubuntu/storage_data_new/python_scripts/STIP/data_final_v4.csv')


# In[74]:


df = df.reset_index()
df.shape


# In[86]:


df.head()


# In[91]:


df_community_new = pd.merge(df, df_community_num, how='inner', left_on='index', right_on='query_paper')


# In[100]:


df_community_new.to_csv('df_community_new_75.csv', index=False)


# In[93]:


df_community_new.shape


# In[94]:


df_community_new.head(2)


# In[95]:


df_community_new.paper_community.value_counts()


# In[98]:


df_community_100


# In[97]:


import pandas as pd
import matplotlib.pyplot as plt

df_community_100 = pd.DataFrame(df_community_new.paper_community.value_counts()[:6])
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


# In[99]:


import pandas as pd
import matplotlib.pyplot as plt

df_community_100 = pd.DataFrame(df_community_new.paper_community.value_counts()[6:12])
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


# In[170]:


df_community_100 = pd.DataFrame(df_community_new.paper_community.value_counts()[:12])
df_community_100 = df_community_100.reset_index()
df_community_100


# In[173]:


listoftextclusters = []
cluster_num_list = []

for i in range(len(df_community_100)):
    text = ''
    cluster = df_community_100['index'][i]
    text = ' '.join(df_community_new.loc[df_community_new['paper_community']==cluster, 'text_clean'].values)
    cluster_num_list.append(cluster)
    listoftextclusters.append(text)


# In[174]:


cluster = df_community_100['index'][2]
cluster 


# In[175]:


len(listoftextclusters)


# In[176]:


tokens_without_sw = [remove_stopwords(text) for text in listoftextclusters]
# df_plot['appln_clean_text_st_r'] = tokens_without_sw
len(tokens_without_sw)


# In[177]:


data_cluter = pd.DataFrame()
data_cluter ['appln_abstract_st_r'] = tokens_without_sw
data_cluter ['community_num'] = cluster_num_list


# In[178]:


data_cluter.head(20)


# # 4. KT and KCC Keywords analysis

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


# In[ ]:




