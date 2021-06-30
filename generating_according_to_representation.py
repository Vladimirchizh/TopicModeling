import numpy as np
import torch
import re
import random

np.random.seed(42)
torch.manual_seed(42)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

art = ['искусство', 'концерт','филармония','сегодня']
beauty = ['волос' 'маникюр'   'запись']
cars = ['автомобиль', 'авто', 'машина']
clubs = ['билет','вход','музыка','клуб ']
competitive_sport = ['игра', 'команда', 'место']
cost_vacations = ['отдых','море', 'пляж']
dresses = ['размер', 'платье', 'наличие', 'ткань']
handmade = ['ряд', 'handmade', 'ручнаяработа' ]
massage = ['массаж', 'семинар', 'релакс']
music_consert = ['группа','концерт','альбом','песня','рок']
parenting = ['ребенок', 'родитель']
photo = ['фото', 'фотография', 'шоу', 'фотограф', 'свадьба']
renovation = ['дома', 'ремонт', 'работа', 'монтаж']
series = ['фильм', 'серия', 'кино']
shopping = ['магазин', 'вещь', 'одежда', 'обувь']
skin_hair_care = ['волос', 'маникюр', 'запись']
sport = ['футбол','матч','сегодня','победа']


data_samples = {

    .6: 'cost_vacations'
    #.6: 'series'

    }

keys = list(data_samples.keys())
vals = list(data_samples.values())



for i in data_samples.values():
  print(i)
  tok = GPT2Tokenizer.from_pretrained("models/"+i)
  model = GPT2LMHeadModel.from_pretrained("models/"+i)
  
  representations = open('models/%s/representations.txt'%i, "w")

  for a in range(round(10*keys[vals.index(i)])):
    print('sample: ',a, 'message')
    text = "<BOS> "+random.choice(eval(i))
    inpt = tok.encode(text, return_tensors="pt")
    out = model.generate(inpt.cpu(), max_length=90, 
                       repetition_penalty=5.0, 
                       do_sample=True, top_k=5, 
                       top_p=0.95, temperature=1)
    
    f_text = re.sub('expand text','',re.sub(r'(?:\s)<[^, ]*', '',  tok.decode(out[0])))
    print(f_text)
    representations.write(f_text + '. \n')
    
  representations.close()

