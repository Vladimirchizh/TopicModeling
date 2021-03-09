# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
# %%
#uploading data needed
df = pd.read_parquet(r'/Users/apple/BDML/topic_modeling/theta_transposed_ND_big_clean.parquet.gzip')

#df['text'].replace('', np.nan, inplace=True)

# %%
print(df.columns)


# %%

data_topic = 'авто автомобиль машина добавлять'

a = pd.DataFrame()
a['theme'] = df.drop(columns = ['text', 'owner_id']) \
    .idxmax(axis=1)

a['text'] = df['text']
a['coef'] = df[data_topic]
a = a.loc[a['theme'] == data_topic ] \
    .drop(columns = ['theme']) \
    .drop_duplicates(subset = 'text') \
    .sort_values('coef', ascending=False) \
    .reset_index(drop = True) \
    .rename({'owner_id':'id'})

# %%

a = a.loc[a['coef'] < 0.95].head(round(len(a)*0.25))
#a = a.iloc[200:].reset_index(drop = True)
#a = a[a['text'].str.contains('iphone')]
# %%

import langid
a['lang'] = a['text'].map(lambda s:langid.classify(s))
a = a[a['lang'].str.contains('ru', regex = False) == True]
a = a.drop(['lang'], axis = 1)
print(a.head())
# %%
file_test = open("test.txt","w")
file_train = open('train.txt', "w")

for i in a.head(round(len(a)*0.8))['text']:
    file_train.write(i+'. \n')
file_train.close()
for i in a.tail(round(len(a)*0.2))['text']:
    file_test.write(i+'. \n')
file_test.close() 
# %%
#a.to_csv('~/BDML/topic_modeling/posts.csv')
# %%
