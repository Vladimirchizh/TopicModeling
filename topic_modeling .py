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
# %%
# regularization 
import artm
# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
#from pymystem3 import Mystem

# nltk stemmers
stemmerRu = SnowballStemmer("russian") 
stemmerEn = PorterStemmer()

#%%
# made data more suitable for json parsing mechanism
# uploading data
df0 = pd.read_json(r'~/BDML/id-psy_posts.json/0_28c5a7ee_id-psy_posts.json')
df1 = pd.read_json(r'~/BDML/id-psy_posts.json/1_18f10508_id-psy_posts.json')
df2 = pd.read_json(r'~/BDML/id-psy_posts.json/2_8e726921_id-psy_posts.json')
df3 = pd.read_json(r'~/BDML/id-psy_posts.json/3_a5e719df_id-psy_posts.json')
# alternative
#df = pd.read_csv(r'/Users/apple/BDML/НИР/train.csv')
#union all and dropping empty lines 
#df = pd.concat([df0[['text', 'owner_id']], df1[['text', 'owner_id']], df2[['text', 'owner_id']], df3[['text', 'owner_id']]])
df['text'].replace('', np.nan, inplace=True)
df.dropna(subset=['text'], inplace=True)
df.reset_index(drop=True, inplace=True)
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
    #stem_words=[stemmerRu.stem(w) for w in filtered_words]
    #lemms = [Mystem.lemmatize(str(w)) for w in filtered_words]
    #stem_words=[stemmerEn.stem(w) for w in stem_words]
    return " ".join(filtered_words)

# cleaning text
df['clean'] = df['text'].map(lambda s:preprocess(s))
df.drop(['text'], axis = 1,  inplace = True)
df.drop_duplicates(inplace=True)
df.index.is_unique
df.dropna(subset=['clean'], inplace=True)
df.reset_index(drop=True, inplace=True)
#%%
# creating the function for transformation to vowpal_wabbit format


def df_to_vw_regression(df, filepath='in.txt', columns=None, target=None, namespace='text'):
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


# changing the type of data created

#vw = df_to_vw_regression(df, filepath='/Users/apple/BDML/topic_modeling/TopicModeling/data.txt',  target = 'owner_id')

#%%
# batching data for applying it to our model
batch_vectorizer = artm.BatchVectorizer(data_path='/Users/apple/BDML/topic_modeling/TopicModeling/data.txt',
                                        data_format='vowpal_wabbit',
                                        collection_name= 'vw',
                                        target_folder='my_collection_batches')

#batch_vectorizer = artm.BatchVectorizer(data_path='/Users/apple/BDML/topic_modeling/TopicModeling/my_collection_batches',
#                                        data_format='batches')
#%%
# LDA (BigARTM package) Model
# setting up lda parameters

lda = artm.LDA(num_topics=20,
               alpha=0.001, beta=0.019,
               cache_theta=True,
               num_document_passes=5,
               dictionary=batch_vectorizer.dictionary)

#Phi is the ‘parts-versus-topics’ matrix, and theta is the ‘composites-versus-topics’ matrix
lda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)

#checking up the parametrs of the matricies
lda.sparsity_phi_last_value
lda.sparsity_theta_last_value
lda.perplexity_last_value

#%%

#top tokens for each class of all the clusters we've got
top_tokens = lda.get_top_tokens(num_tokens=10)
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

dictionary = artm.Dictionary()
dictionary.gather(data_path='/Users/apple/BDML/topic_modeling/TopicModeling/my_collection_batches')
dictionary.save_text(dictionary_path='/Users/apple/BDML/topic_modeling/TopicModeling/my_collection_batches/my_dictionary.txt')

#%%
# intial objects creation
T=40
topic_names = ['topic_{}'.format(i) for i in range(T)]

model_artm = artm.ARTM(dictionary = dictionary, topic_names= topic_names, cache_theta= True)
model_artm.scores.add(artm.PerplexityScore(name='PerplexityScore',dictionary = dictionary))
model_artm.scores.add(artm.TopTokensScore(name='top_words',num_tokens = 10))
model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer', tau=-0.7,
                                                            topic_names=topic_names[:35]))
model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='smooth_phi_regularizer', tau=0.3,
                                                           topic_names=topic_names[35:]))

#%%
# initializing the model we've set up
model_artm.initialize(dictionary)
model_artm.num_document_passes = 3
#%%
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=16)

#%%
perplexityScore = list(model_artm.score_tracker['PerplexityScore'].value)

#%%
#%matplotlib inline
import glob
import os
import matplotlib.pyplot as plt

# %%
# visualizing perplexity
plt.scatter(range(len(perplexityScore)), perplexityScore)
plt.xlabel('number of iterations')
plt.ylabel('perplexity score')
# %%
top_tokens = model_artm.score_tracker['top_words']
for topic_name in model_artm.topic_names:
    print ('\n',topic_name)
    for (token, weight) in zip(top_tokens.last_tokens[topic_name][:40],
                               top_tokens.last_weights[topic_name][:40]):
        print (token, '-', weight)
# %%
# extraction of phi and theta to enviroment 

artm_phi = model_artm.get_phi()
artm_theta = model_artm.get_theta()
# %%
# transposing theta

theta_transposed = artm_theta.transpose()
theta_texts = pd.concat([df, theta_transposed], axis = 1)
# %%
# aglomerative clustering 
model = AgglomerativeClustering(n_clusters=16, affinity='euclidean', linkage='ward')
model.fit(theta_transposed)
labels = model.labels_
# %%
theta_transposed['labels'] = labels
#theta_texts['leading_topic'] = theta_transposed.idxmax(axis=1)

df['label'] = labels
merged = df.merge(initial_text, left_on='clean', right_on = 'clean',left_index=True )
# Checkpoint

file1 = open("myfile.txt","w")

for i in merged[ merged['label'] == 1]['text']:
    file1.write(i+'. \n')
#file1.writelines(df[df['label'] == 3]['clean']) 

file1.close() 
# %%

# starting pytorch 

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
import os
from argparse import Namespace


flags = Namespace(
    train_file='myfile.txt',
    seq_size=32,
    batch_size=16,
    embedding_size=64,
    lstm_size=64,
    gradients_norm=5,
    initial_words=['просто'],
    predict_top_k=5,
    checkpoint_path='checkpoint',
)

def get_data_from_file(train_file, batch_size, seq_size):
    with open(train_file, 'r') as f:
        text = f.read()
    text = text.split()

    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    print('Vocabulary size', n_vocab)

    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]

class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))
def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        flags.train_file, flags.batch_size, flags.seq_size)

    net = RNNModule(n_vocab, flags.seq_size,
                    flags.embedding_size, flags.lstm_size)
    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, 0.01)

    iteration = 0
    for e in range(50):
        batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
        state_h, state_c = net.zero_state(flags.batch_size)
        
        # Transfer data to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:
            iteration += 1
            
            # Tell it we are in training mode
            net.train()

            # Reset all gradients
            optimizer.zero_grad()

            # Transfer data to GPU
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()

            # Perform back-propagation
            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(
                net.parameters(), flags.gradients_norm)
            # Update the network's parameters
            optimizer.step()
            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, 200),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

            if iteration % 300 == 0:
                predict(device, net, flags.initial_words, n_vocab,
                        vocab_to_int, int_to_vocab, top_k=5)

def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    print(' '.join(words))
    
if __name__ == '__main__':
    main()
# %%

# %%
import markovify
#%%

# simple state markov chain 
tt  = theta_texts[theta_texts['leading_topic'] == 'topic_32']
text_model = markovify.Text(tt['text'], state_size= 2 )
text_model.compile(inplace = True)
for i in range(5):
    print(text_model.make_short_sentence(280))

#%%
# multistate markov chain 
model_a = markovify.Text(theta_texts[theta_texts['leading_topic'] == 'topic_18']['text'], state_size = 2 )
model_b = markovify.Text(theta_texts[theta_texts['leading_topic'] == 'topic_25']['text'], state_size= 2)

model_combo = markovify.combine([ model_a, model_b ], [ 1.5, 1 ])
for i in range(5):
    print(model_combo.make_sentence())

















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

