'''
Evaluate sentiment and get prediction sequence using output files from attack script:
importance_synonym_substitution_attack.py
'''

import json
import sys
import os
import argparse
from models import NMTSeq2Seq

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DIR', type=str, help='Directory with stored results of attack, e.g. Attacked_Data/Imp-Ru_N2')
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="start index in data")
    commandLineParser.add_argument('--end_ind', type=int, default=2000, help=" end index in data")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/view_prediction.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load model
    model = NMTSeq2Seq()
    
    # Evaluate
    for ind in range(args.start_ind, args.end_ind):
        filename = f'{args.DIR}/{ind}.txt'
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                info = json.load(f)
        except:
            print(f'Failed to load {ind}.txt')
            continue

        original_sentiment = info['original_probs']
        attacked_sentiment = info['attacked_probs']

        original_source = info['sentence']
        attacked_source = info['attacked_sentence']

        original_prediction = model.predict(original_source)
        attacked_prediction = model.predict(original_source)

        print(f'{ind}\n')
        print(f'Original Source: {original_source}')
        print(f'Original Prediction: {original_prediction}')
        print(f'Original Sentiment: {original_sentiment}')
        print()
        print(f'Attacked Source: {attacked_source}')
        print(f'Attacked Prediction: {attacked_prediction}')
        print(f'Attacked Sentiment: {attacked_sentiment}')
        print()
        print()
        



