# %%
from IPython import get_ipython


# %%
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
from pymystem3 import Mystem

# nltk stemmers
stemmerRu = SnowballStemmer("russian") 
stemmerEn = PorterStemmer()
# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering

#uploading data needed
df = pd.read_csv(r'/Users/apple/BDML/data/train.csv')
df['text'].replace('', np.nan, inplace=True)
df.dropna(subset=['text'], inplace=True)
df.reset_index(drop=True, inplace=True)
# %%
# preprocessing function
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
    #lem_words=[Mystem().lemmatize(w) for w in filtered_words]
    #stem_words=[stemmerRu.stem(w) for w in filtered_words]
    return " ".join(filtered_words)


# cleaning text
df['clean'] = df['text'].map(lambda s:preprocess(s))
# %%
#%time df.to_csv('/Users/apple/BDML/data/trained_clean.csv')
df = pd.read_csv(r'/Users/apple/BDML/data/trained_clean.csv')
df = df.drop(['Unnamed: 0', 'text'], axis = 1)
df.head()


# %%
import artm


# creating the function for transformation to vowpal_wabbit format

def df_to_vw_regression(df, filepath='in.txt', columns=None, target=None, namespace='clean'):
    if columns is None:
        columns = df.columns.tolist()
    columns.remove(target)

    with open(filepath, 'w') as f:
        for _, row in tqdm(df.iterrows()):
            if namespace:
                f.write('|{0} '.format( namespace))
            else:
                f.write('{0} | '.format(row[target]))
            last_feature = row.index.values[-1]
            for idx, val in row.iteritems():
                if idx not in columns:
                    continue
                if isinstance(val, str):
                    f.write('{0}'.format(val.replace(' ', ' ').replace(':', ' ')))
                elif isinstance(val, float) or isinstance(val, int):
                    if not math.isnan(val):
                        f.write('{0}:{1}'.format(idx.replace(' ', ' ').replace(':', ' '), val))
                    else:
                        continue
                else:
                    f.write('{0}'.format(val.replace(' ', ' ').replace(':', ' ')))
                if idx != last_feature:
                    f.write(' ')
            f.write('\n')


# %%
# changing the type of data created
vw = df_to_vw_regression(df, filepath='data_df.txt', target='doc_id')


# %%

# batching data for applying it to our model
batch_vectorizer = artm.BatchVectorizer(data_path='data_df.txt',
                                        data_format='vowpal_wabbit',
                                        collection_name='vw',
                                        target_folder='batches2')

#batch_vectorizer = artm.BatchVectorizer(data_path='batches2', data_format='batches')


# %%
# setting up dictionary
dictionary = batch_vectorizer.dictionary


# %%
number_of_topics = 60
topic_names = ['topic_{}'.format(i) for i in range(number_of_topics)]

# inial objects creation
model_artm = artm.ARTM(topic_names=topic_names,
                       cache_theta=True,
                       dictionary=dictionary, 
                       seed = 123,
                       show_progress_bars = True)

model_artm.scores.add(artm.PerplexityScore(name='PerplexityScore',dictionary = dictionary))
#model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
#model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
#model_artm.scores.add(artm.TopicKernelScore(name='TopicKernelScore',probability_mass_threshold=0.3))
#model_artm.scores.add(artm.TopTokensScore(name='Top_words', num_tokens=20, class_id='text'))
model_artm.scores.add(artm.TopTokensScore(name='top_words',num_tokens = 10))

# additional regulizers
#model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=2.5e+4))
model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparseTheta', tau=- 0.05))
#model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SmoothPhi', tau= 0.3, topic_names= topic_names))#[35:]))

#setting up the number of tokens
model_artm.num_document_passes = 10

#initializing the model we've set up
model_artm.initialize(dictionary=dictionary)


# %%
# fitting the model
get_ipython().run_line_magic('time', 'model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=16)')


# %%
# phi and theta
artm_phi = model_artm.get_phi()
artm_theta = model_artm.get_theta()
## top_tokens
top_tokens = model_artm.score_tracker['top_words']
for topic_name in model_artm.topic_names:
    print ('\n',topic_name)
    for (token, weight) in zip(top_tokens.last_tokens[topic_name][:number_of_topics],top_tokens.last_weights[topic_name][:number_of_topics]):
        print (token, '-', weight)


# %%
import glob
import os
import matplotlib.pyplot as plt

perplexityScore = list(model_artm.score_tracker['PerplexityScore'].value)
get_ipython().run_line_magic('matplotlib', 'inline')

# visualizing perplexity
plt.scatter(range(len(perplexityScore)), perplexityScore)
plt.xlabel('number of iterations')
plt.ylabel('perplexity score')


# %%
lables = dict()
def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


# lemmatisation of top tokens

for topic_name in model_artm.topic_names:
    
    lables[topic_name] = ' '.join(word for word in top_tokens.last_tokens[topic_name][0:6])
    lables[topic_name] = ''.join(Mystem().lemmatize(lables[topic_name]))
    lables[topic_name] = ' '.join(unique_list(lables[topic_name].split()))
    
    print(lables[topic_name])
    
    artm_theta.rename({topic_name:lables[topic_name]}, inplace = True)

    
lables


# %%
artm_theta


# %%
theta_transposed = artm_theta.transpose()

#odd_themes = []
#theta_transposed.drop(columns = odd_themes)


#theta_transposed.to_csv('theta_transposed.csv')
#theta_transposed = pd.read_csv(r'theta_transposed.csv')

df['theme'] = theta_transposed.idxmax(axis=1)
df.head()


# %%
group_user = pd.read_csv(r'/Users/apple/BDML/data/group_user.csv')
lables = pd.read_csv(r'/Users/apple/BDML/data/train_ids.csv')
df = df.merge(lables, on = 'doc_id')
df.head()


# %%
df.theme.unique()


# %%
interests = group_user     .merge(df,left_on = 'group', right_on = '_id' )     .drop(['clean', 'doc_id'], axis = 1)     .sort_values('user')

interests = interests.reset_index()


# %%

interests = interests.pivot_table(index = 'user', columns='theme')     .replace(np.nan, 0) 

    


# %%
interests

# 1) топики надо будет почистить:
#     a) убрать однокорренные слова 
#     б) избавиться от лишних топиков 
    
# 2) думаю зафиксировать состояние топиков,вручную выкинуть лишние
# 3) сгруппировать пользователей по возрасту, пол и построить распределения этих векторов интересов внутри группы
# 4) потом генерировать похожие синтетические распределения как для группы, так и отдельный вектор для человека df['theme'] = theta_transposed.apply(lambda s: s.abs().nlargest(5).index.tolist(), axis=1)
# df['theme'] = df['theme'].astype(str).str.replace("'",'')
# df['theme'] = df['theme'].str.replace("[",'')
# df['theme'] = df['theme'].str.replace("]",'')
# pd.concat([df.drop(['theme'], axis =1), df['theme'].str.split(',', expand=True)], axis=1)

# df = df.merge(group_user, how = 'right',left_on = '_id', right_on = 'group')# trainin agglomeative clustering 
# model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
# model.fit(theta_transposed)
# labels = model.labels_
