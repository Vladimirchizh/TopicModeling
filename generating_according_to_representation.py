import numpy as np
import torch
import re

np.random.seed(42)
torch.manual_seed(42)

from transformers import GPT2LMHeadModel, GPT2Tokenizer


data_samples = {

    .25:'art',
    .15:'photo',
    .14:'dresses'

    }

keys = list(data_samples.keys())
vals = list(data_samples.values())



for i in data_samples.values():
  print(i)
  tok = GPT2Tokenizer.from_pretrained("models/"+i)
  model = GPT2LMHeadModel.from_pretrained("models/"+i)
  text = "<BOS> "
  inpt = tok.encode(text, return_tensors="pt")


  for a in range(round(10*keys[vals.index(i)])):
    print('sample: ',a, 'message')
    
    out = model.generate(inpt.cpu(), max_length=90, 
                       repetition_penalty=5.0, 
                       do_sample=True, top_k=5, 
                       top_p=0.95, temperature=1)
    
    representations = open('models/%s/representations.txt'%i, "w")
    f_text = re.sub('expand text','',re.sub(r'(?:\s)<[^, ]*', '',  tok.decode(out[0])))
    print(f_text)
    representations.write(f_text + '. \n')

