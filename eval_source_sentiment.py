'''
Evaluate sentiment of SOURCE language using output files from attack script:
importance_synonym_substitution_attack.py
'''

import json
import sys
import os
import argparse
from eval_sentiment import print_stats
from models import LangSentClassifier

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DIR', type=str, help='Directory with stored results of attack, e.g. Attacked_Data/Imp-Ru_N2')
    commandLineParser.add_argument('--lang', type=str, default='ru', help='source language')
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="start index in data")
    commandLineParser.add_argument('--end_ind', type=int, default=2000, help=" end index in data")
    commandLineParser.add_argument('--original', type=str, default='no', help=" is it unattacked data?")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_source_sentiment.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    if args.lang == 'ru':
        mname = 'blanchefort/rubert-base-cased-sentiment-rusentiment'
        neg=2
        neu=0
        pos=1
    model = LangSentClassifier(mname=mname)

    # Evaluate
    negatives = []
    neutrals = []
    positives = []
    missed = 0
    counts = [0, 0, 0]

    for ind in range(args.start_ind, args.end_ind):
        filename = f'{args.DIR}/{ind}.txt'
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                info = json.load(f)
        except:
            print(f'Failed to load {ind}.txt')
            missed +=1
            continue

        
        source_text = info['attacked_sentence']
        if args.original == 'yes':
            source_text = info['sentence']
        scores = model.predict(source_text)
        # import pdb; pdb.set_trace()
        negatives.append(float(scores[neg]))
        neutrals.append(float(scores[neu]))
        positives.append(float(scores[pos]))
        
        ind_max = max(enumerate(scores), key=lambda x: x[1])[0]
        counts[ind_max] += 1
    
    # Return stats
    # import pdb; pdb.set_trace()
    print_stats('Negative', negatives)
    print_stats('Neutral', neutrals)
    print_stats('Positive', positives)
    print()

    tot = (args.end_ind-args.start_ind) - missed
    fracs = [c/tot for c in counts]
    print(f'Fraction Negative: {fracs[neg]}')
    print(f'Fraction Neutral: {fracs[neu]}')
    print(f'Fraction Positive: {fracs[pos]}')
    print()

    print(f"Missed {missed}/{tot} samples")

