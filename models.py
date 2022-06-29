from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
from scipy.special import softmax
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import re

class NMTSeq2Seq(nn.Module):

    """ Neural Machine Translation """
    def __init__(self, mname = 'facebook/wmt19-ru-en'):
        super(NMTSeq2Seq, self).__init__()

        self.tokenizer = FSMTTokenizer.from_pretrained(mname)
        self.model = FSMTForConditionalGeneration.from_pretrained(mname)

    def predict(self, text):
        '''
        Translate
        '''
        self.model.eval()

        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        outputs = self.model.generate(
            input_ids = input_ids,
            num_beams = 15,
            do_sample = False,
            max_length = 256,
            length_penalty = 1.0,
            early_stopping = True,
            use_cache = True,
            num_return_sequences = 1)
        translation = self.tokenizer.decode(outputs.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return translation


class SentClassifier(nn.Module):

    """ Roberta enc model """

    def __init__(self):

        super(SentClassifier, self).__init__()

        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def preprocess(self, text):
        new_text = []
    
    
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    def predict(self, text):
        '''
        Returns 3 probability scores:
             (negative, neutral, positive)
        '''
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return scores


class NMTSent():

    """Neural Translation followed by Sentiment Classification"""

    def __init__(self, mname = 'facebook/wmt19-ru-en'):

        self.nmt_model = NMTSeq2Seq(mname=mname)
        self.sentiment_model = SentClassifier()
    
    def predict(self, text):
        '''
        Translate and sentiment classify
        '''
        self.nmt_model.eval()
        self.sentiment_model.eval()

        translation = self.nmt_model.predict(text)
        return self.sentiment_model.predict(translation)
    
    
    

class LangSentClassifier():

    """Sentiment Classifier for different languages"""

    def __init__(self, lang='ru'):

        self.lang = lang
        if lang == 'ru':
            mname = 'blanchefort/rubert-base-cased-sentiment-rusentiment'
        elif lang == 'de':
            mname = 'oliverguhr/german-sentiment-bert'
        elif lang == 'en':
            mname = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(mname)
        self.model = AutoModelForSequenceClassification.from_pretrained(mname)
    
    def replace_numbers(self,text: str) -> str:
            return text.replace("0"," null").replace("1"," eins").replace("2"," zwei").replace("3"," drei").replace("4"," vier").replace("5"," fünf").replace("6"," sechs").replace("7"," sieben").replace("8"," acht").replace("9"," neun")         

    def clean_text(self,text: str)-> str:   

        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\\S+', re.MULTILINE) 

        text = text.replace("\n", " ")        
        text = self.clean_http_urls.sub('',text)
        text = self.clean_at_mentions.sub('',text)        
        text = self.replace_numbers(text)                
        text = self.clean_chars.sub('', text) # use only text chars                          
        text = ' '.join(text.split()) # substitute multiple whitespace with single whitespace   
        text = text.strip().lower()
        return text
    
    def preprocess(self, text):
        new_text = []
    
    
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    @torch.no_grad()
    def predict(self, text, no_neutral = False):
        '''
            Returns probabilties as [neg, neu, pos]
        '''
        if self.lang == 'de':
            text = self.clean_text(text)
        if self.lang == 'en':
            text = self.preprocess(text)
        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        scores = outputs[0][0].detach().numpy()
        scores = softmax(scores)
        scores = [float(s) for s in scores]
        if self.lang == 'en':
            if no_neutral:
                return [scores[0], 0, scores[2]]
            return scores
        if self.lang == 'de':
            if no_neutral:
                return [scores[1], 0, scores[0]]
            return [scores[1], scores[2], scores[0]]
        if self.lang == 'ru':
            if no_neutral:
                return [scores[2], 0, scores[1]]
            return [scores[2], scores[0], scores[1]]

