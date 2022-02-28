# Objective

Use [Facebook FAIR's BPE Transformer model](https://huggingface.co/facebook/wmt19-de-en) submitted to the WMT19 translation task.
Evaluation is performed on the WMT19 test set using the [sacreBLEU tool](https://github.com/mjpost/sacreBLEU).

Evaluation of this model is described [here](https://github.com/rainavyas/NMTFAIRWMT19/blob/main/README.md)

The aim is to perform a sequence-to-sequence adversarial attack, where an attack in the source language achieves an increased _positive_ sentiment in the predicted sequence in the target language. The sentiment score for English as a target language is measured using a [pre-trained Roberta-base model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment).

The following types of adversarial attacks are considered:

- an importance based synonym substitution attack (where N words are substituted).

# Requirements

python3.6 or above

## Install with PyPI
`pip install torch transformers nltk pygermanet`

Other setup required to use germanet: https://pypi.org/project/pygermanet/#setup
