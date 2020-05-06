#!/usr/bin/env python
# coding: utf-8

import torch.utils.data as utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.hub import load
import torchvision.transforms as transforms

from sklearn.metrics import f1_score, accuracy_score

import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages
from fastai.text import *
from fastai.metrics import *


from transformers import AdamW
from functools import partial

import os
from sklearn.model_selection import train_test_split

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig, 'bert-base-uncased'),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig, 'roberta-base')}



class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""

    def __init__(self, pretrained_tokenizer: BertTokenizer, model_type='bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t: str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] + [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens


class TransformersVocab(Vocab):
    def __init__(self, tokenizer: BertTokenizer):
        super(TransformersVocab, self).__init__(itos=[])
        self.tokenizer = tokenizer

    def numericalize(self, t: Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        # return self.tokenizer.encode(t)

    def textify(self, nums: Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(
            self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(
            nums)

    def __getstate__(self):
        return {'itos': self.itos, 'tokenizer': self.tokenizer}

    def __setstate__(self, state: dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int, {v: k for k, v in enumerate(self.itos)})


class CustomTransformerModel(nn.Module):

    def __init__(self, CustomModel):
        super(CustomTransformerModel, self).__init__()
        self.transformer = CustomModel

    def forward(self, input_ids):
        # Return only the logits from the transfomer
        logits = self.transformer(input_ids)[0]
        return logits


def main():
    path = './Data/Tweets.csv'
    dataset = pd.read_csv(path, engine='python')
    test_ratio = 0.3
    train, test = train_test_split(dataset, test_size = test_ratio)
    NUM_EPOCHS = 1
    for key in MODEL_CLASSES.keys():
        model_class, tokenizer_class, config_class, model_name = MODEL_CLASSES[key]
        transformer_tokenizer = tokenizer_class.from_pretrained(model_name)
        transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer=transformer_tokenizer,
                                                               model_type=key)
        fastai_tokenizer = Tokenizer(tok_func=transformer_base_tokenizer, pre_rules=[], post_rules=[])

        transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
        numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

        tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer,
                                               include_bos=False,
                                               include_eos=False)

        transformer_processor = [tokenize_processor, numericalize_processor]

        pad_first = bool(key in ['xlnet'])
        pad_idx = transformer_tokenizer.pad_token_id

        databunch = (TextList.from_df(train, cols='Tweets', processor=transformer_processor)
                     .split_by_rand_pct(0.1)
                     .label_from_df(cols= 'Labels')
                     .add_test(test)
                     .databunch(bs=1, pad_first=pad_first, pad_idx=pad_idx))

        config = config_class.from_pretrained(model_name)
        config.num_labels = 5

        transformer_model = model_class.from_pretrained(model_name, config = config)
        # transformer_model = model_class.from_pretrained(model_name, num_labels = 5)
        custom_transformer_model = CustomTransformerModel(CustomModel = transformer_model)

        CustomAdamW = partial(AdamW, correct_bias=False)
        learner = Learner(databunch,
                          custom_transformer_model,
                          opt_func=CustomAdamW,
                          metrics=[accuracy, fbeta])
        learner.fit_one_cycle(NUM_EPOCHS)
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))
        print(learner.sched.rec.rec_metrics)
        ax[1].plot(list(range(95)), learner.sched.rec_metrics)
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        plt.savefig('accuracy_vs_epoch_'+key+'.png')


if __name__ == "__main__":
    main()
