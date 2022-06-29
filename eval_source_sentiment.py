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
    model = LangSentClassifier(lang=args.lang)

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
        negatives.append(float(scores[0]))
        neutrals.append(float(scores[1]))
        positives.append(float(scores[2]))
        
        ind_max = max(enumerate(scores), key=lambda x: x[1])[0]
        counts[ind_max] += 1
    
    # Return stats
    # import pdb; pdb.set_trace()
    print_stats('Negative', negatives)
    print_stats('Neutral', neutrals)
    print_stats('Positive', positives)
    print()

    tot = (args.end_ind-args.start_ind)
    tot_adj = tot - missed
    fracs = [c/tot_adj for c in counts]
    print(f'Fraction Negative: {fracs[0]}')
    print(f'Fraction Neutral: {fracs[1]}')
    print(f'Fraction Positive: {fracs[2]}')
    print()

    print(f"Missed {missed}/{tot} samples")

