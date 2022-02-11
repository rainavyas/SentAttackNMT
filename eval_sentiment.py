'''
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
from models import SentClassifier
from statistics import mean, stdev

def print_stats(text, values):
    avg = mean(values)
    std = stddev(values)
    print(f'\n{text}:\t{avg} +- {std}')

if __name__ == "__main__":
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to data')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_sentiment.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the model
    model = SentClassifier()

    # Load the data
    with open(args.IN, 'r') as f:
        sentences = f.readlines()
    
    # Evaluate
    negatives = []
    neutrals = []
    positives = []
    for i, sent in enumerate(sentences):
        print(f'Evaluating {i}/{len(sentences)}')
        scores = model.predict(sent)
        negatives.append(scores[0])
        neutrals.append(scores[1])
        positives.append(scores[2])
    
    # Return stats
    print_stats('Negative', negatives)
    print_stats('Neutral', neutrals)
    print_stats('Positive', positives)
