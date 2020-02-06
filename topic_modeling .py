#%%
# importing the libraries
# basic stuff
import pandas as pd
import numpy as np
import math
import re
import nltk #swiss knife army for nlp
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem.snowball import SnowballStemmer 
from nltk.corpus import stopwords
from tqdm import tqdm


# regularization 
import artm


# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#lemmatizer = WordNetLemmatizer()

#%%
# made data more suitable for json parsing mechanism


# nltk stemmers
stemmerRu = SnowballStemmer("russian") 
stemmerEn = PorterStemmer()

# uploading data
df0 = pd.read_json(r'/Users/apple/BDML/id-psy_posts.json/0_28c5a7ee_id-psy_posts.json')
df1 = pd.read_json(r'/Users/apple/BDML/id-psy_posts.json/1_18f10508_id-psy_posts.json')
df2 = pd.read_json(r'/Users/apple/BDML/id-psy_posts.json/2_8e726921_id-psy_posts.json')
df3 = pd.read_json(r'/Users/apple/BDML/id-psy_posts.json/3_a5e719df_id-psy_posts.json')
#df_test = pd.read_json(r'/Users/apple/BDML/id-psy_posts.json/3_a5e719df_id-psy_posts.json')
# alternative
#df = pd.read_csv(r'/Users/apple/BDML/НИР/train.csv')
#%%
#union all and dropping empty lines 
df = pd.concat([df0[['text', 'owner_id']], df1[['text', 'owner_id']], df2[['text', 'owner_id']], df3[['text', 'owner_id']]])
df['text'].replace('', np.nan, inplace=True)
df.dropna(subset=['text'], inplace=True)
df.reset_index(drop=True, inplace=True)


#df_test['text'].replace('', np.nan, inplace=True)
#df_test.dropna(subset=['text'], inplace=True)
#%% 
#defining preprocessing function 


def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('russian')]
    stem_words=[stemmerRu.stem(w) for w in filtered_words]
    stem_words=[stemmerEn.stem(w) for w in stem_words]
    return " ".join(stem_words)
#%%
# cleaning text
df['text'] = df['text'].map(lambda s:preprocess(s)) 


#%%
# creating the function for transformation to vowpal_wabbit format


def df_to_vw_regression(df, filepath='in.txt', columns=None, target=None, namespace='namespace'):
    if columns is None:
        columns = df.columns.tolist()
    columns.remove(target)

    with open(filepath, 'w') as f:
        for _, row in tqdm(df.iterrows()):
            if namespace:
                f.write('{0} |{1} '.format(row[target], namespace))
            else:
                f.write('{0} | '.format(row[target]))
            last_feature = row.index.values[-1]
            for idx, val in row.iteritems():
                if idx not in columns:
                    continue
                if isinstance(val, str):
                    f.write('{0}'.format(val.replace(' ', '_').replace(':', '_')))
                elif isinstance(val, float) or isinstance(val, int):
                    if not math.isnan(val):
                        f.write('{0}:{1}'.format(idx.replace(' ', '_').replace(':', '_'), val))
                    else:
                        continue
                else:
                    f.write('{0}'.format(val.replace(' ', '_').replace(':', '_')))
                if idx != last_feature:
                    f.write(' ')
            f.write('\n')


def df_to_vw_classification(
        df,
        filepath='mc.txt',
        columns=None,
        target=None,
        tag=None,
        namespace='n'
):
    if columns is None:
        columns = df.columns.tolist()
    columns.remove(target)
    columns.remove(tag)

    with open(filepath, 'w') as f:
        for _, row in tqdm(df.iterrows()):
            if namespace:
                f.write(f"{row[target]} \'{row[tag]} |{namespace} ")
            else:
                f.write(f"{row[target]} \'{row[tag]} | ")
            last_feature = columns[-1]
            for idx, val in row.iteritems():
                if idx not in columns:
                    continue
                if isinstance(val, str):
                    f.write(f"{val.replace(' ', '_').replace(':', '_')}")
                elif isinstance(val, float) or isinstance(val, int):
                    if not math.isnan(val):
                        f.write(f"{idx.replace(' ', '_').replace(':', '_')}:{val}")
                    else:
                        continue
                else:
                    f.write(f"{val.replace(' ', '_').replace(':', '_')}")
                if idx != last_feature:
                    f.write(' ')
            f.write('\n')

#%%
# changing the type of data created
vw = df_to_vw_regression(df, filepath='/Users/apple/BDML/topic_modeling/TopicModeling/data.txt', target='text')
# vw1 = df_to_vw_classification(df, filepath='data.txt', target='target')

#%%
# batching data for applying it to our model
batch_vectorizer = artm.BatchVectorizer(data_path='/Users/apple/BDML/topic_modeling/TopicModeling/data.txt',
                                        data_format='vowpal_wabbit',
                                        collection_name='vw',
                                        target_folder='batches')

#batch_vectorizer = artm.BatchVectorizer(data_path='koselniy_batches', data_format='batches')


#%% md

# LDA (BigARTM package) Model

#%%

# setting up lda parameters


lda = artm.LDA(num_topics=20,
               alpha=0.001, beta=0.019,
               cache_theta=True,
               num_document_passes=5,
               dictionary=batch_vectorizer.dictionary)



#Phi is the ‘parts-versus-topics’ matrix, and theta is the ‘composites-versus-topics’ matrix
lda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)


#checking up the parametrs of the matricies
lda.sparsity_phi_last_value
lda.sparsity_theta_last_value
lda.perplexity_last_value

#%%


#top tokens for each class of all the clusters we've got
top_tokens = lda.get_top_tokens(num_tokens=200)
for i, token_list in enumerate(top_tokens):
    print('Topic #{0}: {1}'.format(i, token_list))

d = pd.DataFrame(top_tokens)
top_tokens_vec= d.apply(lambda x: ' '.join(x), axis=1)
#%%

lda_phi = lda.phi_
lda_theta = lda.get_theta()

#%% 
# alternaive lda via sklearn
tf_idf = TfidfVectorizer(stop_words=stopwords.words('russian'), ngram_range = (1, 2), max_df=.25, max_features=5000) 
#count = CountVectorizer(stop_words=stopwords.words('russian'), max_df=.1, max_features=5000) 
X = tf_idf.fit_transform(df['text'].values)
lda = LatentDirichletAllocation(n_components=25, random_state=123, learning_method='batch', n_jobs = -1, max_iter = 10)

#%%
X_topics = lda.fit_transform(X)
n_top_words = 10
feature_names = tf_idf.get_feature_names()
#%%
# leading topic
X_topics = pd.DataFrame(X_topics)
X_topics.reset_index(drop=True, inplace=True)

#%%
# top tokens

for topic_idx, topic in enumerate(lda.components_):
  print("Topic %d:" % (topic_idx + 1))
  print(' '.join([feature_names[i]
  for i in topic.argsort()
    [:-n_top_words - 1:-1]]))
    
#%%
phi_numbers = pd.DataFrame(lda.components_).transpose()
f_names = pd.DataFrame(feature_names)
lda_skl_phi = pd.concat([ f_names, phi_numbers], axis=1)
#%%
# concating initial dataset with topics

messages_topics = pd.concat([df, X_topics], axis = 1)

# adding leading topic

messages_topics['leading_topic'] = X_topics.idxmax(axis=1)
#%%
import markovify
mt  = messages_topics[messages_topics['leading_topic'] == 1]

# simple state markov chain 

text_model = markovify.Text(mt['text'], state_size= 2 )
text_model.compile(inplace = True)
for i in range(5):
    print(text_model.make_short_sentence(280))
#%%  
# combining 2 models 

model_a = markovify.Text(messages_topics[messages_topics['leading_topic'] == 11]['text'], state_size = 2 )
model_b = markovify.Text(messages_topics[messages_topics['leading_topic'] == 7]['text'], state_size= 2)

model_combo = markovify.combine([ model_a, model_b ], [ 1.5, 1 ])
for i in range(5):
    print(model_combo.make_sentence())
#%%
# ARTM (BigARTM package) Model

dictionary = batch_vectorizer.dictionary

#%%

topic_names = ['topic_{}'.format(i) for i in range(100)]
#inial objects cration
model_plsa = artm.ARTM(topic_names=topic_names, cache_theta=True,
                       scores=[artm.PerplexityScore(name='PerplexityScore',
                                                    dictionary=dictionary)])
model_artm = artm.ARTM(topic_names=topic_names, cache_theta=True,
                       scores=[artm.PerplexityScore(name='PerplexityScore',
                                                    dictionary=dictionary)],
                       regularizers=[artm.SmoothSparseThetaRegularizer(name='SparseTheta',
                                                                       tau=0.05)])

#%%
# adding some scores for our future model
# PLSA
model_plsa.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
model_plsa.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
model_plsa.scores.add(artm.TopicKernelScore(name='TopicKernelScore',probability_mass_threshold=0.3))
model_plsa.scores.add(artm.TopTokensScore(name='Top_words', num_tokens=20, class_id='text'))
model_plsa.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=300))


# ARTM
model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
model_artm.scores.add(artm.TopicKernelScore(name='TopicKernelScore',probability_mass_threshold=0.3))
model_artm.scores.add(artm.TopTokensScore(name='Top_words', num_tokens=20, class_id='text'))
model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=300))


#regulizers
model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=-0.05))
#model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=1.5e+4))

#%%
#setting up the number of tokens
model_plsa.num_document_passes = 1
model_artm.num_document_passes = 1

#%%
#initializing the model we've set up
model_plsa.initialize(dictionary=dictionary)
model_artm.initialize(dictionary=dictionary)

#%%

#fitting the model
model_plsa.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)

#%%
artm_phi = model_artm.get_phi()
artm_theta = model_artm.get_theta()

#%%
%matplotlib inline
import glob
import os
import matplotlib.pyplot as plt

import artm
#%%
# adding scores crhecking and plot
def print_measures(model_plsa, model_artm):
    print('Sparsity Phi: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['SparsityPhiScore'].last_value,
        model_artm.score_tracker['SparsityPhiScore'].last_value))

    print('Sparsity Theta: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['SparsityThetaScore'].last_value,
        model_artm.score_tracker['SparsityThetaScore'].last_value))

    print('Kernel contrast: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['TopicKernelScore'].last_average_contrast,
        model_artm.score_tracker['TopicKernelScore'].last_average_contrast))

    print('Kernel purity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['TopicKernelScore'].last_average_purity,
        model_artm.score_tracker['TopicKernelScore'].last_average_purity))

    print('Perplexity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['PerplexityScore'].last_value,
        model_artm.score_tracker['PerplexityScore'].last_value))

    plt.plot(range(model_plsa.num_phi_updates),
             model_plsa.score_tracker['PerplexityScore'].value, 'b--',
             range(model_artm.num_phi_updates),
             model_artm.score_tracker['PerplexityScore'].value, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('PLSA perp. (blue), ARTM perp. (red)')
    plt.grid(True)
    plt.show()


# printing results
print_measures(model_plsa, model_artm)
#%%
#comparing Phi & Theta matricies


plt.plot(range(model_plsa.num_phi_updates),
         model_plsa.score_tracker['SparsityPhiScore'].value, 'b--',
         range(model_artm.num_phi_updates),
         model_artm.score_tracker['SparsityPhiScore'].value, 'r--', linewidth=2)

plt.xlabel('Iterations count')
plt.ylabel('PLSA Phi sp. (blue), ARTM Phi sp. (red)')
plt.grid(True)
plt.show()

plt.plot(range(model_plsa.num_phi_updates),
         model_plsa.score_tracker['SparsityThetaScore'].value, 'b--',
         range(model_artm.num_phi_updates),
         model_artm.score_tracker['SparsityThetaScore'].value, 'r--', linewidth=2)

plt.xlabel('Iterations count')
plt.ylabel('PLSA Theta sp. (blue), ARTM Theta sp. (red)')
plt.grid(True)
plt.show()


#%%

#checking out the sparsity of our matricies
#print(model_artm.score_tracker["SparsityPhiScore"].last_value)
#print(model_artm.score_tracker["SparsityThetaScore"].last_value)

print(model_artm.score_tracker["SparsityPhiScore"].last_value)
print(model_artm.score_tracker["SparsityThetaScore"].last_value)

#checking out topic kernel score
print(model_artm.score_tracker["TopicKernelScore"])

#print(model_plsa.score_tracker["TopicKernelScore"])



#checking up the top words
#print(model_artm.score_tracker["Top_words"])
#print(model_artm.score_tracker["Top_words"])
#print(model_artm.)

#%%

for topic_name in model_artm.topic_names:
    print(topic_name + ': '),
    print(model_artm.score_tracker['TopTokensScore'].last_tokens[topic_name])


#%%
artmTokens = model_artm.score_tracker['TopTokensScore'].last_tokens
#%%
artmPhi = model_artm.phi_

#%%
artmTokens = pd.DataFrame(artmTokens)
artmToks= artmTokens.apply(lambda x: ' '.join(x), axis=0)
artmToks

#%% md

#%%
## KMeans model

#%%

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords

#%%

# retrieving text: setting up stop words, making Tf-Idf
words = stopwords.words('russian')
vectorizer = TfidfVectorizer(stop_words=words)
X = vectorizer.fit_transform(df['text'])

#%%

# counting KMeans for 13 clusters and fitting vectorized data in to the model
true_k = 18
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10)
model.fit(X)

#%%

#figuring out centroids for our data
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
centers = model.cluster_centers_

#%%

for i in range(true_k):
    print('Cluster %d:' % i),
    for ind in order_centroids[i, :300]:
        print(' %s' % terms[ind])

#%%

kdf = pd.DataFrame(
    [i,  terms[ind]]
    for i in range(true_k)
    for ind in order_centroids[i, :30]).groupby([0])[1].transform(lambda x: ' '.join(x)).drop_duplicates()

#%%

print(kdf)

#%% md

# RuBETR classifier

#%%

import warnings
warnings.filterwarnings('ignore')

#%%

#from deeppavlov.dataset_iterators.basic_classification_iterator import BasicClassificationDatasetIterator
#iterator = BasicClassificationDatasetIterator(data, seed=42, shuffle=True)
#for batch in iterator.gen_batches(data_type="train", batch_size=13):
#    print(batch)
#    break

#%%

#importing more libraries
#import math

#downloading te right vocabulaty
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor
bert_preprocessor = BertPreprocessor(vocab_file="~/.deeppavlov/downloads/bert_models/rubert_cased_L-12_H-768_A-12_v2/vocab.txt",
                                     do_lower_case=False,
                                     max_seq_length=64)



#%%

from deeppavlov.models.bert.bert_classifier import BertClassifierModel
#from deeppavlov.metrics.accuracy import sets_accuracy
bert_classifier = BertClassifierModel(
    n_classes=15,
    return_probas=False,
    one_hot_labels=True,
    bert_config_file="~/.deeppavlov/downloads/bert_models/rubert_cased_L-12_H-768_A-12_v2/bert_config.json",
    pretrained_bert="~/.deeppavlov/downloads/bert_models/rubert_cased_L-12_H-768_A-12_v2/bert_model.ckpt",
    save_path="ru_bert_model/model",
    load_path="ru_bert_model/model",
    keep_prob=0.5,
    learning_rate=1e-05,
    learning_rate_drop_patience=5,
    learning_rate_drop_div=2.0
)

#%%

#changing vocabulary format
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
vocab = SimpleVocabulary(save_path="./binary_classes.dict")
#iterator.get_instances(data_type="train")
#vocab.fit(iterator.get_instances(data_type="train")[1])
#list(vocab.items())
#bert_classifier(bert_preprocessor(top_tokens_vec))
#bert_classifier(bert_preprocessor(artmToks))

#%% md

# Turning BERT classifier into pretrained PyTorch

#%%

from pytorch_transformers import BertForPreTraining, BertConfig, BertTokenizer, load_tf_weights_in_bert #BertModel

#%% md

#transformin deeppavlov BERT model into PyTorch pretrained model

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file):
    config = BertConfig.from_json_file(bert_config_file)
    model = BertForPreTraining(config)
    model = load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    return model

BERT_MODEL_PATH = "/Users/apple/.deeppavlov/downloads/bert_models/rubert_cased_L-12_H-768_A-12_v2/"
tf_checkpoint_path = BERT_MODEL_PATH + 'bert_model.ckpt'
bert_config_file = BERT_MODEL_PATH + 'bert_config.json'

downloaded = convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=False)

#%%

encoder = downloaded.bert

#%%

#working summarizer

#needs customization:
# 1) adding Rus from spacy
# 2) adding multilang bert to bertParent
# 3) changing model and language all over the preproccessing file
from summarizer import Summarizer
model = Summarizer()

#%%

body = '''
🏅Главная тема. 
Комитет ВАДА рекомендовал максимально жестко наказать Россию за обнаруженные недавно манипуляции с базой данных Московской антидопинговой лаборатории: 
отстранить от Олимпийских и Паралимпийских игр и чемпионатов мира по всем видам спорта (футбол тоже считается, ага) 
на четыре года, а также запретить принимать любые международные соревнования. 
Пока это лишь рекомендации — окончательное решение должен принять исполком ВАДА 9 декабря. 
Надежд, прямо скажем, почти никаких: МОК уже поддержал введение «самых жестких санкций». 

🎯Нерегулярная армейская рубрика «Кто бросил валенок на пульт».
Суд взыскал 31 миллион рублей с двух военнослужащих Черноморского флота, которые случайно запустили авиационную ракету «воздух — поверхность».
Инцидент произошел еще в 2017 году, при полете ракета снесла ворота и часть стены ангара и уничтожила различное оборудование и еще одну ракету. 
'''


result = model(body, min_length=60, max_length = 500)
full = ''.join(result)
print(full)

#%%

#working summarizer
text = body

from summa import keywords
lables = keywords.keywords(text, language = 'russian',ratio = 0.02, scores = False)
print(lables)

#%%

from summa import summarizer
sumar = summarizer.summarize(text)
print(sumar)

#%% md

# TFIDF Classifier based on LentaRu dataset

#%%

# uploading data
lentaRu = pd.read_csv(r'/Users/apple/Downloads/lenta-ru-news.csv')
lentaRuTxt = lentaRu['text']
lentaRu.info()

# importing all libraries needed
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#scikit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

#defining classes stemmer and stop words
stemmer = SnowballStemmer('russian')
words = stopwords.words('russian')

#%%

#cleaning the data from odd symbols, stand alone letters
#lowercasing every document
lentaRu['cleaned'] = lentaRu['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^а-яА-Я]",
                                                                                               " ",
                                                                                               str(x)).split() if i not in words]).lower())
#checking out if any cell is NA
lentaRu.isna()

#%%

#getting rid of NAs
lentaRuWon = lentaRu.dropna()
lentaRuWon
print(lentaRuWon['tags'].unique())
print(lentaRuWon['topic'].unique())

#%%

#partitioning the data in to test and train frames
X_train, X_test, y_train, y_test = train_test_split(lentaRuWon['cleaned'],
                                                    lentaRuWon.topic,
                                                    test_size=0.05)

#%%

#creating pipeline
#create TFIDF vector for single words and bigrams
#selecting only good features for our vector
#choosing SVC algorithm
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words=words, sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

#%%

#fitting the model into the pipeliine
model = pipeline.fit(X_train, y_train)

#%%

#preparation
#getting instances
vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']
#transforming features
feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)

#%%

#checking out accuracy score of the model
print("accuracy score: " + str(model.score(X_test, y_test)))
#accuracy score: 0.8038830593120476

#%%

import dill as pickle

#with open('testing_model.pk', 'wb') as file:
#    pickle.dump(model, file)
with open('testing_model.pk','rb') as f:
    model = pickle.load(f)


#%%

lda_predict = model.predict(top_tokens_vec)
lda_predict = pd.DataFrame(lda_predict)
lda_result = pd.concat([top_tokens_vec, lda_predict], axis=1)
print(lda_result)

#%%

artm_predict = model.predict(artmToks)
artm_predict = pd.DataFrame(artm_predict)
artm_result = pd.concat([artmToks, artm_predict], axis=1)
print(artm_result)

#%%


