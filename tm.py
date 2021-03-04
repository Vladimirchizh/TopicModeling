# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
# %%
# importing the libraries
# basic stuff
import numpy as np
import math
import re
import nltk 
from nltk.corpus import stopwords
from tqdm import tqdm
#from pymystem3 import Mystem

# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import normalize
#import langid
from tqdm import tqdm

import artm
# %%
#uploading data needed
df = pd.read_parquet(r'/Users/apple/BDML/topic_modeling/TopicModeling/theta_transposed_ND_big_clean.parquet.gzip')

#df['text'].replace('', np.nan, inplace=True)


# %%
print(df.columns)


# %%


data_topic = 'игра команда'

a = pd.DataFrame()
a['theme'] = df.drop(columns = ['text', 'owner_id']).idxmax(axis=1)
a['text'] = df['text']
a['coef'] = df[data_topic]
a = a.loc[a['theme'] == data_topic ] \
    .drop(columns = ['theme']) \
    .drop_duplicates(subset = 'text') \
    .sort_values('coef', ascending=False) \
    .reset_index(drop = True) \
    .loc[a['coef'] < 0.95]

a = a.head(round(len(a)*0.33))
#a = a.iloc[200:].reset_index(drop = True)










# %%



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
vw = df_to_vw_regression(df, filepath='/Users/apple/BDML/topic_modeling/TopicModeling/data_.txt', target='owner_id')


# %%

# batching data for applying it to our model
batch_vectorizer = artm.BatchVectorizer(data_path='/Users/apple/BDML/topic_modeling/TopicModeling/data_.txt',
                                        data_format='vowpal_wabbit',
                                        collection_name='vw',
                                        target_folder='/Users/apple/BDML/topic_modeling/TopicModeling/batchesND')


# %%
batch_vectorizer = artm.BatchVectorizer(data_path='/Users/apple/BDML/topic_modeling/TopicModeling/batchesND', data_format='batches')
# setting up dictionary

dictionary = batch_vectorizer.dictionary


# %%
number_of_topics = 80
topic_names = ['topic_{}'.format(i) for i in range(number_of_topics)]

# inial objects creation
model_artm = artm.ARTM(topic_names=topic_names,
                       cache_theta=True,
                       dictionary=dictionary, 
                       seed = 123,
                       show_progress_bars = True)

model_artm.scores.add(artm.PerplexityScore(name='PerplexityScore',dictionary = dictionary))
model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
#model_artm.scores.add(artm.TopicKernelScore(name='TopicKernelScore',probability_mass_threshold=0.3))
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
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=16)


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
perplexityScore = list(model_artm.score_tracker['PerplexityScore'].value)
perplexityScore[1:]


# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import os
import matplotlib.pyplot as plt

# visualizing perplexity [1:]
plt.scatter(range(len(perplexityScore[1:])), perplexityScore[1:])
plt.xlabel('number of iterations')
plt.ylabel('perplexity score')


# %%
plt.scatter(range(len(perplexityScore[8:])), perplexityScore[8:])
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
artm_theta.head()

цена наличие руб цвет рубль


игра матч футбол команда мир
гороскоп сегодня рождаться июнь август июль
радоваться бог господь христос
область год россия район город дтп
хотеть лайк нахуй любить блять репост


год век дизайн место дом музей
озеро город мир метр остров парк


від ціна або але характеристика буде
iphone apple видео новый android google
человек жизнь ремарк
альбом amp love рок группа
руб работа тело требоваться дом квартира
человек бизнес жизнь должный
год мама сказать говорить дверь домой
день нога мышца упражнение тело

минута масло вода затем добавлять

хотеть любовь друг жизнь рядом


человек имя энергия деньги жизнь сила


день праздник год август июль сентябрь

секс член порно рот девушка
компания рубль год млн
хотеть писать искать друг фото год
женщина муж жена мужчина спать
фото фотография фотограф свадьба художник автор
жизнь любовь мир счастие жить
любить глаз лишь снова сердце твой
сердце любовь жизнь мир лишь

любить человек хотеть лишь знать хотеться
мужчина женщина говорить самый bir

россия год сша страна война
серия сезон видео фильм онлайн смотреть
здравствовать пожалуйста подсказывать сколько добрый день

продавать цена размер руб отдавать состояние


bmw ряд петля mercedes лицо ford
например поэтому работа время несколько часто

новость книга запрещать интернет правда мир
лишь свет глаз душа счастие взгляд
книга роман жизнь автор

# %%
theta_transposed = artm_theta.transpose()
theta_transposed.head()


# %%

odd_themes = [
            'человек именно часто являться',
            'человек жизнь сегодня жить лишь',
            'человек жизнь значит твой',
            'сегодня друг день группа утро добрый',
            'человек любить жизнь',
            'александр год работа сергей андрей',
            'ребенок жизнь',
            'руб amp вылет день ночь завтрак',
            'человек лишь смерть мир',
            'что это как так я ты',
            'год сша нью',
            'ученый год иметь собака длина',
            'срок год доставка право день',
            'день сегодня стоять постараться отношение',
            'волос sex big tits anal',
            'жанр que драма режиссер роль não',
            'почему сегодня день делать знать говорить',
            'деньги день человек жизнь год',
            'грн являться мир габардин жизнь льон',
            'делать вопрос год поэтому часто ставить',
            'несколько становиться дома снова смерть'
    ]



#theta_transposed.drop(odd_themes,axis = 1, inplace = True)

theta_transposed.to_csv('theta_transposed.csv')


# %%
#theta_transposed = pd.read_csv(r'theta_transposed.csv').drop(['Unnamed: 0'], axis = 1)
df['theme'] = theta_transposed.idxmax(axis=1)
df['coef'] = 1 #theta_transposed.max(axis = 1)
df.tail(50)


# %%
group_user = pd.read_csv(r'/Users/apple/BDML/data/group_user.csv').rename(columns = {'group':'group_id'})
lables = pd.read_csv(r'/Users/apple/BDML/data/train_ids.csv').rename(columns = {'_id':'group_id'})


# %%
lables.tail()


# %%
df = df.merge(lables, on = 'doc_id')
df.head(10)


# %%
interests = group_user     .merge(df, on = 'group_id' )     .drop(['clean', 'doc_id', 'group_id'], axis = 1)     .sort_values('user')

interests.set_index('user', inplace = True)
interests.head()


# %%
interests = interests     .reset_index()     .pivot_table(index = 'user', 
                 columns='theme', 
                 aggfunc='sum') \
    .replace(np.nan, 0) 

interests.head()


# %%
interests.columns = interests.columns.get_level_values(1)
interests.columns = [''.join(col).strip() for col in interests.columns.values]
interests.tail(30)


# %%
interests = interests.T
interests = (100. * interests / interests.sum())
interests = interests.T
interests.head()


# %%
# creating age groups
vk_user = pd.read_csv(r'/Users/apple/BDML/data/vk_profiles.csv')
age_sex = vk_user.loc[:, :'sex'].rename(columns = {'id': 'user'})
age_sex['age_group'] = pd.cut(age_sex.age, [0, 14, 20, 27, 36, 45, 54, 63, 72, 81])
#age_sex.head()

cut = age_sex
cut['count'] = 1
cut.pivot_table('count', index='age_group', columns='sex', aggfunc='sum')


# %%
groups = age_sex.loc[:,'age_group'].unique().astype(str)
age_sex['age_group'] = age_sex['age_group'].astype(str)


# %%
grouping = dict()
ingroup_clusters = dict()
for o in range(2):
    for i in groups:
        
        grouping[str(i)+"_"+str(o)] = age_sex             .loc[(age_sex['sex'] == o) & (age_sex['age_group'] == i)]
        
        
        grouping[str(i)+"_"+str(o)] = grouping[str(i)+"_"+str(o)].merge(interests, on = 'user')
        


# %%
grouping['(36.0, 45.0]_1'].loc[:, (grouping['(36.0, 45.0]_1'] != 0).any(axis=0)].head(10)


# %%
for a in grouping:
    ingroup_clusters[a] = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')         .fit(grouping[a].drop(columns = ['user','age','sex','age_group','count']))
    grouping[a]['ag_lables'] = ingroup_clusters[a].labels_
    # for b in range(4):
        # grouping[a].loc[grouping[a]['ag_lables'] == b]

# %% [markdown]
# ## Мужчины от 36 до 45 включительно кластер 0

# %%
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.pyplot import figure
figure(num=None, figsize=(14, 10), dpi=300)
a = grouping['(36.0, 45.0]_1']                    .drop(columns = ['user','age','sex','age_group','count'])                    .loc[grouping['(36.0, 45.0]_1']['ag_lables'] == 0]                    .drop(columns = 'ag_lables')                    .iloc[:15,:]
ax = plt.axes()
ax.set_title('Мужчины от 36 до 45 включительно кластер 0')
sb.heatmap(a.T, annot = True)

# %% [markdown]
# ## Мужчины от 20 до 27 включительно кластер 0

# %%
figure(num=None, figsize=(14, 10), dpi=300)

a = grouping['(20.0, 27.0]_1']                    .drop(columns = ['user','age','sex','age_group','count'])                    .loc[grouping['(20.0, 27.0]_1']['ag_lables'] == 0]                    .drop(columns = 'ag_lables')                    .iloc[:15,:]
ax = plt.axes()
ax.set_title('Мужчины от 20 до 27 включительно кластер 0')
sb.heatmap(a.T, annot = True)

# %% [markdown]
# ## Женщины от 27 до 36 включительно кластер 0

# %%
figure(num=None, figsize=(14, 10), dpi=300)
a = grouping['(27.0, 36.0]_0']                     .drop(columns = ['user','age','sex','age_group','count'])                    .loc[grouping['(27.0, 36.0]_0']['ag_lables'] == 0]                    .drop(columns = 'ag_lables')                    .iloc[:15,:]
ax = plt.axes()
ax.set_title('Женщины от 27 до 36 включительно кластер 0')
sb.heatmap(a.T, annot = True)


# %%
grouping.pop('(0.0, 14.0]_0')
grouping.pop('(0.0, 14.0]_1')
grouping.pop('nan_0')
grouping.pop('nan_1')


# %%
hm = pd.DataFrame()
for i in grouping:
    hm[i] =  grouping[i]                 .drop(columns = ['user','age','sex','age_group','count'])                 .loc[grouping[i]['ag_lables'] == 0]                 .drop(columns = 'ag_lables')                 .mean()


figure(num=None, figsize=(10, 10), dpi=300)
#hm.iloc[:40,:]
heat_map = sb.heatmap(hm.iloc[:40,:], annot=True)
plt.show(heat_map)


# %%
hm2 = hm
hm2['mean'] = list(hm2.T.mean())
hm2 = hm2[hm2['mean'] >1.7].drop(columns = 'mean')
figure(num=None, figsize=(13, 10), dpi=300)
heat_map = sb.heatmap(hm2, annot=True)
plt.show(heat_map)


# %%

a4_dims = (11.7, 10.27)
fig, ax = plt.subplots(figsize=a4_dims)
#hm2.hist(bins=20,grid = True, ax=ax)
hm2.iloc[:,8:].hist(bins=20, ax=ax)


# %%
# '(27.0, 36.0]_0'
plt.hist(hm2['(27.0, 36.0]_0'], density=True, bins=30)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Data');
#plt.xticks(range(len(hm2['(27.0, 36.0]_0'])), hm2['(27.0, 36.0]_0'], size='small')
plt.show()


# %%
a4_dims = (9.7, 7.27)
fig, ax = plt.subplots(figsize=a4_dims)
hm2     .rename(columns = {'(27.0, 36.0]_0':'Women from 27 to 36','(27.0, 36.0]_1':'Men from 27 to 36'})     .loc[:,['Women from 27 to 36','Men from 27 to 36']]     .plot.barh(ax = ax)
#(kind = 'bar', ax = ax,subplots = False)

plt.ylabel('Topics')
plt.xlabel('Probability in %');


# %%
a4_dims = (9.7, 7.27)
fig, ax = plt.subplots(figsize=a4_dims)
hm2     .rename(columns = {'(45.0, 54.0]_0':'Women from 45 to 54','(45.0, 54.0]_1':'Men from 45 to 54'})     .loc[:,['Women from 45 to 54','Men from 45 to 54']]     .plot.barh(ax = ax)

plt.ylabel('Topics')
plt.xlabel('Probability in %');

figure(num=None, figsize=(14, 10), dpi=300)

sb.heatmap(grouping['(36.0, 45.0]_1'] \
           .drop(columns = ['user','age','sex','age_group','count']) \
           .loc[grouping['(36.0, 45.0]_1']['ag_lables'] == 1].iloc[:10,:].T, annot = True)from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(ingroup_clusters['(72, 81]1'], 
                        ingroup_clusters['(72, 81]0']))df['theme'] = theta_transposed.apply(lambda s: s.abs().nlargest(5).index.tolist(), axis=1)
df['theme'] = df['theme'].astype(str).str.replace("'",'')
df['theme'] = df['theme'].str.replace("[",'')
df['theme'] = df['theme'].str.replace("]",'')
pd.concat([df.drop(['theme'], axis =1), df['theme'].str.split(',', expand=True)], axis=1)

df = df.merge(group_user, how = 'right',left_on = '_id', right_on = 'group')# trainin agglomeative clustering 
model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
model.fit(theta_transposed)

