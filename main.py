import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import pandas as pd

def load_manual_stopwords(path):
    with open(path, 'r') as stopwords_file:
        output = stopwords_file.readlines()
    output = [word.strip("\n") for word in output]
    return output

def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')

def preprocess(text):
    result = []
    manual_stopwords = load_manual_stopwords("stopwords_industry.txt")
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in manual_stopwords and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

if __name__ == "__main__":
    data = pd.read_csv('jobs.csv')
    role_and_descriptions = data[['role_title','description']]
    role_and_descriptions['index'] = role_and_descriptions.index
    documents = role_and_descriptions
    # print(documents[documents['index'] == 0])

    # preprocess description text
    processed_docs = documents['description'].map(preprocess)
    # print(processed_docs[:5])

    dictionary = gensim.corpora.Dictionary(processed_docs)
    count = 0
    # just printing to see
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 5:
            break

    # minimum number of docs token appears in
    token_appearance_min_num = 3
    top_frequency = 100000
    dictionary.filter_extremes(no_below=token_appearance_min_num, no_above=0.5, keep_n=top_frequency)
    
    # use bag-of-words corpus
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # use TF-IDF model
    from gensim import corpora, models

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    number_of_topics = 10
    print("BAG OF WORDS model------------------------------")
    lda_model_bow = gensim.models.LdaMulticore(bow_corpus, num_topics=number_of_topics, id2word=dictionary, passes=2, workers=2)
    for idx, topic in lda_model_bow.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # print("TF-IDF model------------------------------")
    # lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=number_of_topics, id2word=dictionary, passes=2, workers=4)
    # for idx, topic in lda_model_tfidf.print_topics(-1):
    #     print('Topic: {} Word: {}'.format(idx, topic))
    
    # test_doc = documents['description'][documents['index'] == 0]
    role_dict = {}
    for row in range(len(documents)):
        role = documents.iloc[row]['role_title']
        description = documents.iloc[row]['description']
        bow_vec = dictionary.doc2bow(preprocess(description))
        scores = sorted(lda_model_bow[bow_vec], key=lambda tup: -1*tup[1])
        # for idx, score in scores:
        #     topic = lda_model_bow.print_topic(idx, 5)
        #     print(f"Role: {role}\t Model_num: {idx}\t Score: {score}\t Topic: {topic}")
        
        role_dict[role] = scores[0]
    # print(role_dict)

    topic_dict = {}
    for key, value in role_dict.items():
        if value[0] in topic_dict:
            topic_dict[value[0]].append(key)
        else:
            topic_dict[value[0]] = [key]
    
    print(topic_dict)

    test_doc = "I want to design things"
    test_doc = "I want to research more machine learning algorithms"
    print(test_doc)
    test_bow_vec = dictionary.doc2bow(preprocess(test_doc))
    for idx, score in sorted(lda_model_bow[test_bow_vec], key=lambda tup: -1*tup[1]):
        print(f"Model_num: {idx}\t Score: {score}\t Topic: {topic}")
