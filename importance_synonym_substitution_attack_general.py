'''
Same as importance_synonym_substitition_attack.py but
allows for different source-target language pairs (not just en as target language)
'''

import torch
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize.treebank import TreebankWordDetokenizer
from wiki_ru_wordnet import WikiWordnet
import json
import sys
import os
import argparse
from collections import OrderedDict
from models import LangSentClassifier, NMTSent
import string
import math
from odenet import synonyms_word

def get_token_importances(tokens, model, sent_ind=2):
    '''
    Returns list of importances in token order

    Importance of a word is measured as the absolute change
    in prediction (of positive sentiment)
    with and without target token included
    '''

    ref_score = model.predict(sentence)[sent_ind] # select correct sentiment

    importances = []
    for i, token in enumerate(tokens):
        if token in string.punctuation:
            importances.append(0)
        else:
            new_sentence = TreebankWordDetokenizer().detokenize(tokens[:i]+tokens[i+1:])
            token_score = model.predict(new_sentence)[sent_ind]
            importances.append(abs(ref_score-token_score))

    return importances


def attack_sentence(sentence, model, wikiwordnet, max_syn=5, frac=0.1, lang='ru', sent_ind=2):
    '''
    Identifies the N most important words (N=frac*length)
    Finds synonyms for these words using Russian WordNet or German OdeNet (choose language)
    Selects the best synonym to replace with based on Forward Pass to maximise
    the selected sentiment score.

    If language passed is 'en', then attack is directly on the sentiment classifier
    and not the NMT system.

    Returns the original_sentence, updated_sentence, original_probs, updated_probs
    '''

    tokens = nltk.word_tokenize(sentence)
    token_importances = get_token_importances(tokens, model, sent_ind=sent_ind)
    inds = torch.argsort(torch.FloatTensor(token_importances), descending=True)

    # Number of tokens to substitute
    N = math.floor(frac*len(tokens))

    attacked_sentence = sentence[:]
    words_swapped = 0

    for ind in inds:
        attacked_tokens = nltk.word_tokenize(attacked_sentence)
        target_token = attacked_tokens[ind]

        synonyms = []
        if lang == 'ru':
            for syn in wikiwordnet.get_synsets(target_token):
                for lemma in syn.get_words():
                    synonyms.append(lemma.lemma())
        elif lang == 'de':
            all = synonyms_word(target_token)
            try:
                synonyms = [item for sublist in all for item in sublist]
            except:
                continue
        elif lang == 'en':
            for syn in wn.synsets(target_token):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
        else:
            raise ValueError('Language not recognised')

        if len(synonyms)==0:
            continue

        # Remove duplicates
        synonyms = list(OrderedDict.fromkeys(synonyms))

        if len(synonyms) > max_syn+1:
            synonyms = synonyms[:max_syn+1]

        best = (attacked_sentence, model.predict(attacked_sentence)[sent_ind]) # (sentence, positivity score)
        change = False
        for syn in synonyms:
            trial_sentence = TreebankWordDetokenizer().detokenize(attacked_tokens[:ind]+[syn]+attacked_tokens[ind+1:])
            score = model.predict(trial_sentence)[sent_ind]
            if score > best[1]:
                change = True
                best = (trial_sentence, score)
        if change:
            attacked_sentence = best[0]
            words_swapped += 1
        
        if words_swapped >= N:
            break

    original_probs = model.predict(sentence)
    attacked_probs = model.predict(attacked_sentence)

    # import pdb; pdb.set_trace()
    return attacked_sentence, original_probs.tolist(), attacked_probs.tolist()


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Source Data file - source if NMT attack and target if sent classifier attack')
    commandLineParser.add_argument('OUT', type=str, help='Directory to store results of attack, e.g. Attacked_Data/Imp-Ru')
    commandLineParser.add_argument('--max_syn', type=int, default=6, help="Number of synonyms to search")
    commandLineParser.add_argument('--frac', type=float, default=0.1, help="Fraction of words to substitute")
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="start index in data file")
    commandLineParser.add_argument('--end_ind', type=int, default=100, help="end index in data file")
    commandLineParser.add_argument('--sent_ind', type=int, default=2, help="sentiment index to attack e.g. 2 is positive")
    commandLineParser.add_argument('--source_lang', type=str, default='ru', help="Source language")
    commandLineParser.add_argument('--target_lang', type=str, default='en', help="Target language")
    commandLineParser.add_argument('--sent_attack', type=str, default='no', help="Attack sentiment classifier using target?")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/importance_synonym_substitution_attack_general.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    wikiwordnet = WikiWordnet()
    nltk.download('wordnet')

    # Load the data
    with open(args.IN, 'r') as f:
        sentences = f.readlines()
    sentences = [s.rstrip('\n') for s in sentences]
    sentences = sentences[args.start_ind:args.end_ind]

    # Create end-to-end stacked model
    if args.sent_attack == 'no':
        model = NMTSent(mname = f'facebook/wmt19-{args.source_lang}-{args.target_lang}')
    else:
        model = LangSentClassifier(lang=args.target_lang)

    # Create directory to save files in
    dir_name = f'{args.OUT}_frac{args.frac}'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


    for i, sentence in enumerate(sentences):

        # Attack and save the sentence attack
        attacked_sentence, original_probs, attacked_probs = attack_sentence(sentence, model, wikiwordnet, max_syn=args.max_syn, frac=args.frac, lang=args.source_lang, sent_ind=args.sent_ind)
        info = {"sentence":sentence, "attacked_sentence":attacked_sentence, "original_probs":original_probs, "attacked_probs":attacked_probs}
        filename = f'{dir_name}/{args.start_ind + i}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(info))
