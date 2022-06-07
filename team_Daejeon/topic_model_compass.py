import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pandas as pd

df = pd.read_pickle('compass_clean.pkl')

# Create Corpus - all the documents
texts = df['text_clean'].values.tolist()
data_words = list(texts)

print(data_words[:1][0][:30])

# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1][0][:30])

# number of topics
num_topics = 5

# Build LDA model
if __name__ == "__main__":
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)

    lda_model.save('model10.gensim')
    topics = lda_model.print_topics(num_words=4)
    for topic in topics:
        print(topic)

    # Visualize
    lda_display = gensimvis.prepare(
        lda_model, corpus, id2word, sort_topics=False)
    pyLDAvis.save_html(lda_display, 'lda_th10.html')
