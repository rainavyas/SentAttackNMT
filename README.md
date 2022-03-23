# Objective

Use [Facebook FAIR's BPE Transformer model](https://huggingface.co/facebook/wmt19-de-en) submitted to the WMT19 translation task.
Evaluation is performed on the WMT19 test set using the [sacreBLEU tool](https://github.com/mjpost/sacreBLEU).

Evaluation of this model is described [here](https://github.com/rainavyas/NMTFAIRWMT19/blob/main/README.md)

The aim is to perform a sequence-to-sequence adversarial attack, where an attack in the source language achieves an increased _positive_ sentiment in the predicted sequence in the target language. The sentiment score for English as a target language is measured using a [pre-trained Roberta-base model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment). Source language used is Russian.

The following types of adversarial attacks are considered:

- an importance based synonym substitution attack (where N words are substituted).

# Requirements

python3.6 or above

## Install with PyPI
`pip install torch transformers scipy`

### Russian Processing
`pip install nltk wiki-ru-wordnet` 

### German Processing

Use [odenet](https://github.com/hdaSprachtechnologie/odenet). To install, clone the repository and then run `pip install .` from within the repo. Further install: `pip install networkx matplotlib`.


