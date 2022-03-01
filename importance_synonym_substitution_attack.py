'''
Importantce ranked synonym substitution attack
'''

import torch
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from wiki_ru_wordnet import WikiWordnet
import json
import sys
import os
import argparse
from collections import OrderedDict
from models import NMTSent
import string

def get_token_importances(tokens, model):
    '''
    Returns list of importances in token order

    Importance of a word is measured as the absolute change
    in prediction (of positive sentiment)
    with and without target token included
    '''

    ref_score = model.predict(sentence)[2] # select probability of positive

    importances = []
    for i, token in enumerate(tokens):
        if token in string.punctuation:
            importances.append(0)
        else:
            new_sentence = TreebankWordDetokenizer().detokenize(tokens[:i]+tokens[i+1:])
            token_score = model.predict(new_sentence)[2]
            importances.append(abs(ref_score-token_score))

    return importances


def attack_sentence(sentence, model, wikiwordnet, max_syn=5, N=1):
    '''
    Identifies the N most important words
    Finds synonyms for these words using Russian WordNet
    Selects the best synonym to replace with based on Forward Pass to maximise
    the positivity score.

    Returns the original_sentence, updated_sentence, original_probs, updated_probs
    '''

    tokens = nltk.word_tokenize(sentence)
    token_importances = get_token_importances(tokens, model)
    inds = torch.argsort(torch.FloatTensor(token_importances), descending=True)

    attacked_sentence = sentence[:]
    words_swapped = 0

    for ind in inds:
        attacked_tokens = nltk.word_tokenize(attacked_sentence)
        target_token = attacked_tokens[ind]

        synonyms = []
        for syn in wikiwordnet.get_synsets(target_token):
            for lemma in syn.get_words():
                synonyms.append(lemma.lemma())
        if len(synonyms)==0:
            continue

        # Remove duplicates
        synonyms = list(OrderedDict.fromkeys(synonyms))

        if len(synonyms) > max_syn+1:
            synonyms = synonyms[:max_syn+1]

        best = (attacked_sentence, model.predict(attacked_sentence)[2]) # (sentence, positivity score)
        change = False
        for syn in synonyms:
            trial_sentence = TreebankWordDetokenizer().detokenize(attacked_tokens[:ind]+[syn]+attacked_tokens[ind+1:])
            score = model.predict(trial_sentence)[2]
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

    import pdb; pdb.set_trace()
    return attacked_sentence, original_probs.tolist(), attacked_probs.tolist()


if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Source Data file')
    commandLineParser.add_argument('OUT', type=str, help='Directory to store results of attack, e.g. Attacked_Data/Imp-Ru')
    commandLineParser.add_argument('--max_syn', type=int, default=6, help="Number of synonyms to search")
    commandLineParser.add_argument('--N', type=int, default=1, help="Number of words to substitute")
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="start index in data file")
    commandLineParser.add_argument('--end_ind', type=int, default=100, help=" end index in data file")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/importance_synonym_substitution_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    wikiwordnet = WikiWordnet()

    # Load the data
    with open(args.IN, 'r') as f:
        sentences = f.readlines()
    sentences = sentences[args.start_ind:args.end_ind]

    # Create end-to-end stacked model
    model = NMTSent()

    # Create directory to save files in
    dir_name = f'{args.OUT}_N{args.N}'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


    for i, sentence in enumerate(sentences):

        # Attack and save the  sentence attack
        attacked_sentence, original_probs, attacked_probs = attack_sentence(sentence, model, wikiwordnet, max_syn=args.max_syn, N=args.N)
        info = {"sentence":sentence, "attacked_sentence":attacked_sentence, "original_probs":original_probs, "attacked_probs":attacked_probs}
        filename = f'{dir_name}/{args.start_ind + i}.txt'
        with open(filename, 'w') as f:
            f.write(json.dumps(info))
