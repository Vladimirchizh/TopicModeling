# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# %%
# uploading data needed
df = pd.read_parquet(r'/Users/apple/BDML/data/theta_transposed_сс_rus.parquet.gzip')
#df.drop(columns=['text'], inplace=True)

# %%
# text: raw before concatination
initial_text = pd.read_csv(r'/Users/apple/BDML/data/group_posts_divided.csv')

#
# %%

lst = df.columns[:len(df.columns)-2]

print(lst)

# %% 
import os
def create_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
    print("Created Directory : ", dir)
  else:
    print("Directory already existed : ", dir)
  return dir


def fix_length(df, length = 100):
    df['length'] = df.text.str.len()
    df = df.loc[df['length'] > length]
    df.drop(columns=['length'], inplace=True)
    len(df)
    return df


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext




def build_dataset(df, dest_path):
    f = open(dest_path, 'w')
    data = ''
    summaries = df['text'].tolist()
    for summary in summaries:
        summary = str(summary).strip()
        summary = re.sub(r"\s", " ", summary)
        bos_token = '<BOS>'
        eos_token = '<EOS>'
        data += bos_token + ' ' + summary + ' ' + eos_token + '\n'

    f.write(data)
# %%

# choosing topic

data_topic =  'игра команда место'
a = pd.DataFrame()
a['theme'] = df.drop(columns=['owner_id']) \
    .idxmax(axis=1)

# a['text'] = df['text']
a['owner_id'] = df['owner_id']
a['coef'] = df[data_topic]
a = a.loc[a['theme'] == data_topic] \
    .drop(columns=['theme']) \
    .sort_values('coef', ascending=False) \
    .reset_index(drop=True)

# taking top 25% of the topic data
# a = a.loc[a['coef'] < 0.95].head(round(len(a)*0.25))
a = a.loc[a['coef'] < 0.95]  # .head(40000)
a = a.loc[a['coef'] > 0.65]

# %%
# a = a.merge(initial_text,how='left', on=['owner_id'])
# a = fix_length(a, 100)
# a.drop_duplicates(inplace = True)
# a['text'] = a['text'].apply(lambda x: cleanhtml(x))
# a = fix_length(a, 100)
# a.drop_duplicates(inplace = True)
# %%
df_sample = a.sample(frac=1).reset_index(drop=True)

test_data = df_sample.text[:round(len(df_sample) / 4)]
train_data = df_sample.text[round(len(df_sample) / 4):]

# saving the samples 
file_test = open("validation.txt", "w")
file_train = open('train.txt', "w")

for i in train_data:
    file_train.write(i + '. \n')
file_train.close()
for i in test_data:
    file_test.write(i + '. \n')
file_test.close()

# %%

train_valid_ratio = 7 / 9
df_train, df_valid = train_test_split(a, train_size=train_valid_ratio, random_state=1)

build_dataset(df_train, 'train.txt')
build_dataset(df_valid, 'validation.txt')





# %%

# saving into the tree of the datasets

train_valid_ratio = 7 / 9

dataframes = dict()
temp = pd.DataFrame()
temp['theme'] = df.drop(columns=['text', 'owner_id']) \
    .idxmax(axis=1)

temp['text'] = df['text']

for n in lst:
    print(n)
    create_dir('pre_gpt/'+n)
    dataframes[n] = temp
    dataframes[n]['coef'] = df[n]

    dataframes[n] = dataframes[n].loc[dataframes[n]['coef'] < 0.95] \
        .loc[dataframes[n]['coef'] > 0.65] \
        .drop_duplicates(subset='text') \
        .sort_values('coef', ascending=False) \
        .reset_index(drop=True)
    
    df_train, df_valid = train_test_split(dataframes[n],
                                          train_size=train_valid_ratio,
                                          random_state=1)
    build_dataset(df_train, 'pre_gpt/'+n+'/train.txt')
    build_dataset(df_valid, 'pre_gpt/'+n+'/validation.txt')




# %%

# saving as the single file

file_multitrain = open('multitrain.txt', "w")
for d in dataframes.values():
    for i in d['text']:
        file_multitrain.write(i + '. \n')

file_multitrain.close()

# %% OR

dataframe = pd.DataFrame()
for i in dataframes.values():
    dataframe = pd.concat([dataframe, i])

dataframe.drop(['coef', 'theme'],
               axis=1,
               inplace=True)
print((len(dataframe)))

dataframe.to_parquet(r'train_high_score.parquet.gzip', compression='gzip')




# %%
temp = pd.DataFrame()
temp['theme'] = df.drop(columns=['text', 'owner_id']) \
    .idxmax(axis=1)

temp['text'] = df['text']
file_multitrain = open('multitrain.txt', "w")

for n in df.columns:

    print(n)
    d = temp
    d['coef'] = df[n]
    d = d.drop_duplicates(subset='text') \
        .sort_values('coef', ascending=False) \
        .reset_index(drop=True) \
        .loc[d['coef'] < 0.95] \
        .loc[d['coef'] > 0.65]
    for i in d['text']:
        file_multitrain.write(i + '. \n')

file_multitrain.close()

# %%
