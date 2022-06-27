'''
Everything this file does can be achieved using 'eval_attacked_sentiment.py'

Input file:

sentence 1
sentence 2
.
.
.
sentence N


Output: Average (mean) + standard deviation of sentiment scores:
 - negative
 - neutral
 - positive
'''

import sys
import os
import argparse
from models import LangSentClassifier
from statistics import mean, stdev

def print_stats(text, values):
    avg = mean(values)
    std = stdev(values)
    print(f'\n{text}:\t{avg} +- {std}')

if __name__ == "__main__":
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to data')
    commandLineParser.add_argument('--lang', type=str, default='en', help='language')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_sentiment.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = LangSentClassifier(lang=args.lang)

    # Load the data
    with open(args.IN, 'r') as f:
        sentences = f.readlines()
    sentences = [s.rstrip('\n') for s in sentences]
    
    # Evaluate
    negatives = []
    neutrals = []
    positives = []
    counts = [0, 0, 0]
    for i, sent in enumerate(sentences):
        print(f'Evaluating {i}/{len(sentences)}')
        scores = model.predict(sent)
        # import pdb; pdb.set_trace()
        negatives.append(scores[0].item())
        neutrals.append(scores[1].item())
        positives.append(scores[2].item())
    
        ind_max = max(enumerate(scores), key=lambda x: x[1])[0]
        counts[ind_max] += 1
    
    # Return stats
    print_stats('Negative', negatives)
    print_stats('Neutral', neutrals)
    print_stats('Positive', positives)
    print()

    tot = len(sentences)
    fracs = [c/tot for c in counts]
    print(f'Fraction Negative: {fracs[0]}')
    print(f'Fraction Neutral: {fracs[1]}')
    print(f'Fraction Positive: {fracs[2]}')
    print()
