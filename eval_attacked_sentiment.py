'''
Evaluate sentiment using output files from attack script:
importance_synonym_substitution_attack.py
'''

import json
import sys
import os
import argparse
from eval_sentiment import print_stats

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DIR', type=str, help='Directory to store results of attack, e.g. Attacked_Data/Imp-Ru_N2')
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="start index in data")
    commandLineParser.add_argument('--end_ind', type=int, default=2000, help=" end index in data")
    commandLineParser.add_argument('--original', type=str, default='no', help=" end index in data")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_attacked_sentiment.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    key = 'attacked_probs'
    if args.original == 'yes':
        key = 'original_probs'
    
    # Evaluate
    negatives = []
    neutrals = []
    positives = []
    missed = 0
    for ind in range(args.start_ind, args.end_ind):
        filename = f'{args.DIR}/{ind}.txt'
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                info = json.load(f)
        except:
            print(f'Failed to load {ind}.txt')
            missed +=1

        scores = info[key]
        negatives.append(scores[0])
        neutrals.append(scores[1])
        positives.append(scores[2])
    
    # Return stats
    print_stats('Negative', negatives)
    print_stats('Neutral', neutrals)
    print_stats('Positive', positives)
    print(f"Missed {missed}/{args.end_ind-args.start_ind} samples")
