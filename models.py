from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch.nn as nn
import numpy as np
from scipy.special import softmax

class SentClassifier(nn.Module):

    """ T5 enc-dec model """

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

