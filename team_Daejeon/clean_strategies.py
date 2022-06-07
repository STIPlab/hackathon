import pyreadr
import collections
import pandas as pd

# Get dataframe
target_rdata_path = "strategies_final.Rdata"
strategies_od = pyreadr.read_r(target_rdata_path)
df = strategies_od["data_final"]
pd.set_option('display.max_colwidth', 200)

# Convert cleaned text into list and add to df
text_clean_arrays = []

for ind in df.index:
    arr = df['text_clean'][ind].split()
    text_clean_arrays.append(arr)

df['text_clean_list'] = text_clean_arrays

# Words of interest
keywords = ['open_data', 'data_access', 'open_access',
            'research_data', 'research_infrastructure', 'transparency']

# Compare documents with words of interest
filtered_df = pd.DataFrame(
    columns=['country', 'year', 'period', 'doc_id', 'title', 'text_clean', 'text_clean_list'])

keywords_count = {'open_science': 0, 'open_data': 0, 'data_access': 0, 'open_access': 0,
                  'research_data': 0, 'research_infrastructure': 0, 'transparency': 0}
doc_count = 0
word_count = 0

for ind in df.index:
    match = set(df['text_clean_list'][ind]).intersection(keywords)
    if bool(match) == True:
        doc_count += 1
        filtered_df.loc[ind] = df.loc[ind]
        for key in match:
            word_count += 1
            keywords_count[key] += 1
print("total docs: " + str(len(df)))
print("doc hits: " + str(doc_count))
print("keyword hits: " + str(word_count))
print(keywords_count)

# Outputting to csv
filtered_df.to_pickle('filtered_strategies.pkl')
