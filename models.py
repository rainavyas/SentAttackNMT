from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
from scipy.special import softmax
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

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

    def __init__(self, mname='blanchefort/rubert-base-cased-sentiment-rusentiment'):

        self.tokenizer = AutoTokenizer.from_pretrained(mname)
        self.model = AutoModelForSequenceClassification.from_pretrained(mname)
    
    @torch.no_grad()
    def predict(self, text):
        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        scores = outputs[0][0].detach().numpy()
        scores = softmax(scores)
        return scores

