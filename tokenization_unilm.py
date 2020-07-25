# coding=utf-8
"""Tokenization classes for UniLM."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from transformers.tokenization_bert import BertTokenizer, whitespace_tokenize

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'unilm-large-cased': "",
        'unilm-base-cased': ""
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'unilm-large-cased': 512, 
    'unilm-base-cased': 512
}


class UnilmTokenizer(BertTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)
