'''
Evaluate sentiment using output files from attack script:
importance_synonym_substitution_attack.py or general version of script
'''

import json
import sys
import os
import argparse
from eval_sentiment import print_stats

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DIR', type=str, help='Directory with stored results of attack, e.g. Attacked_Data/Imp-Ru_frac0.1')
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="start index in data")
    commandLineParser.add_argument('--end_ind', type=int, default=2000, help=" end index in data")
    commandLineParser.add_argument('--original', type=str, default='no', help=" is it unattacked data?")
    commandLineParser.add_argument('--no_neutral', type=str, default='yes', help=" don't use neutral")
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

        scores = info[key]
        if args.no_neutral == 'yes':
            scores[1] = 0
        negatives.append(scores[0])
        neutrals.append(scores[1])
        positives.append(scores[2])
        
        ind_max = max(enumerate(scores), key=lambda x: x[1])[0]
        counts[ind_max] += 1
    
    # Return stats
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


