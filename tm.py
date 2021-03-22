# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import markovify as markov_chain

# %%
#uploading data needed
df = pd.read_parquet(r'/Users/apple/BDML/topic_modeling/TopicModeling/theta_transposed_сс_rus.parquet.gzip')

#df['text'].replace('', np.nan, inplace=True)

# %%
print(df.columns)

# %% 
df['length'] = df.text.str.len()
df = df.loc[df['length'] > 70]
df.drop(columns = ['length'], inplace=True)
len(df)
# %%

# choosing topic
data_topic = 'работа год график день'

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

# taking top 25% of the topic data
#a = a.loc[a['coef'] < 0.95].head(round(len(a)*0.25))
a = a.loc[a['coef'] < 0.95].head(70000)

# %%
import random
df_sample = a.sample(frac=1).reset_index(drop=True)

train_data = df_sample.text[:round(len(df_sample)/4)]
test_data = df_sample.text[round(len(df_sample)/4):]


# %%
# saving the samples 
file_test = open("test.txt","w")
file_train = open('train.txt', "w")

for i in train_data:
    file_train.write(i+'. \n')
file_train.close()
for i in test_data:
    file_test.write(i+'. \n')
file_test.close() 


# %%

temp = pd.DataFrame()
temp['theme'] = df.drop(columns = ['text', 'owner_id']) \
    .idxmax(axis=1)

temp['text'] = df['text']
file_multitrain = open('multitrain.txt', "w")


for n in df.columns:

  print(n)
  d = temp
  d['coef'] = df[n]
  d = d.drop_duplicates(subset = 'text') \
    .sort_values('coef', ascending=False) \
    .reset_index(drop = True) \
    .loc[d['coef'] < 0.95].head(30000)
  for i in d['text']:
    file_multitrain.write(i+'. \n')

file_multitrain.close()
  

# %%
