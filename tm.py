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
data_topic = 'фото фотография шоу фотограф свадьба'

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

# taking top 25% of the topic data
#a = a.loc[a['coef'] < 0.95].head(round(len(a)*0.25))
a = a.loc[a['coef'] < 0.95].head(30000)

# %%
# saving the samples 
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




# creating markov chains for every topic in the dataset 
temp = pd.DataFrame()
temp['theme'] = df.drop(columns = ['text', 'owner_id']) \
    .idxmax(axis=1)

temp['text'] = df['text']
a = a

for n in df.columns:

  print(n)
  d = temp
  d['coef'] = df[n]
  d = d.drop_duplicates(subset = 'text') \
    .sort_values('coef', ascending=False) \
    .reset_index(drop = True) \
    .loc[d['coef'] < 0.95].head(5000)
  
  d['lang'] = d['text'].map(lambda s:langid.classify(s))
  d = d[d['lang'].str.contains('ru', regex = False) == True]
  d = d.drop(['lang'], axis = 1)   

  text_model = markov_chain \
      .Text(d[d['theme'] == n]['text'], state_size= 2, well_formed= True)

  text_model.compile(inplace = True)
  for i in range(4):
    print(text_model.make_short_sentence(160))


# %%
