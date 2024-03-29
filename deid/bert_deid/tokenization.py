# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2019 Alistair Johnson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition on i2b2 2014 dataset """

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
import itertools
from bisect import bisect_left, bisect_right
from bert_deid.ensemble_feature import find_phi_location, create_extra_feature_vector

import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """Features are directly input to the transformer model."""
    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        input_subwords=None,
        input_offsets=None,
        input_lengths=None,
        extra_feature=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

        # offsets are used in evaluation to match predictions with text
        self.input_subwords = input_subwords
        self.input_offsets = input_offsets
        self.input_lengths = input_lengths

        self.extra_feature = extra_feature


def tokenize_with_labels(
    tokenizer, example, pad_token_label_id=-100, default_label='O'
):
    text = example.text

    # bert use ## to indicate subword
    special_characters = '##'

    word_tokens = tokenizer.tokenize(text)

    # make sure we lower case the text if the tokens are lower cased
    # necessary for aligning text with tokens
    do_lower_case = tokenizer.tokenize('A')[0]
    if 'a' in do_lower_case:
        text = text.lower()

    # now reverse engineer the locations of the tokens
    offsets = []
    lengths = []

    # also output a flag indicating if the token is a subword
    token_sw = []
    w = 0
    i = 0

    while i < len(text):
        if w >= len(word_tokens):
            # we have assigned all word tokens
            # what remains must be whitespace/filtered characters
            break

        token = word_tokens[w]

        if token == '[UNK]':
            # can't do this token
            raise ValueError(
                'Unable to get offsets for tokens due to unknown tokens.'
            )

        # ignore the special characters added by the tokenizer
        if token.startswith(special_characters):
            token = token[len(special_characters):]
            subword = True
        else:
            subword = False

        # end if we have reached the end of the text
        if (i + len(token)) > len(text):
            break

        # check if this token matches the original text
        if text[i:i + len(token)] == token:
            offsets.append(i)
            lengths.append(len(token))
            token_sw.append(subword)

            w += 1
            i += len(token)
            continue

        i += 1

    assert len(token_sw) == len(word_tokens)
    assert len(offsets) == len(word_tokens)
    assert len(lengths) == len(word_tokens)

    # initialize token labels as the default label
    # set subword tokens to padded token
    token_labels = [
        pad_token_label_id if sw else default_label for sw in token_sw
    ]

    # when building examples for model evaluation, there are no labels
    if example.labels is None:
        return word_tokens, token_labels, token_sw, offsets, lengths

    w = 0
    for label in example.labels:
        entity_type = label.entity_type
        start, offset = label.start, label.length
        stop = start + offset
        # force all labels to be upper case
        entity_type = entity_type.upper()
        # get the first offset > than the label start index
        i = bisect_left(offsets, start)
        if i == len(offsets):
            # we have labeled a token at the end of the text
            # also catches the case that we label a part of a token
            # at the end of the text, but not the entire token
            if not token_sw[-1]:
                token_labels[-1] = entity_type
        else:
            # find the last token which is within this label
            j = bisect_left(offsets, stop)

            # assign all tokens between [start, stop] to this label
            # *except* if it is a padding token (so the model ignores subwords)
            new_labels = [
                entity_type if not token_sw[k] else pad_token_label_id
                for k in range(i, j)
            ]
            token_labels[i:j] = new_labels

    return word_tokens, token_labels, token_sw, offsets, lengths



def convert_examples_to_features(
    examples,
    label_to_id,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token="[PAD]",
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    feature_overlap=None,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)

        `feature_overlap` - Split a single long example into multiple training observations. This is
        useful for handling examples containing very long passages of text.
            None (default): truncates each example at max_seq_length -> one InputFeature per InputExample.
            [0, 1): controls how much overlap between consecutive segments.
                e.g. `feature_overlap=0.1` means the last 10% of InputFeature 1 will equal first 10%
                of InputFeature 2, assuming that the InputExample is long enough to require splitting.
    """

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2

    features = []
    n_obs = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        patterns = example.patterns
        pattern_label = 1
        ex_phi_locs = []
        for pattern in patterns:
            ex_phi_locs.append(find_phi_location(pattern, pattern_label, example.text, stanfordNER=True))

        assert (len(patterns) == len(ex_phi_locs))

        ex_tokens, ex_labels, ex_token_sw, ex_offsets, ex_lengths = tokenize_with_labels(
            tokenizer, example, pad_token_label_id=pad_token_label_id
        )

        # assign labels based off the offsets
        ex_label_ids = [
            label_to_id[l] if l != pad_token_label_id else pad_token_label_id
            for l in ex_labels
        ]

        n_tokens = len(ex_tokens)
        if feature_overlap is None:
            # we will truncate the sequence by having the iterator only have start=0
            feature_len = n_tokens
        else:
            feature_len = int(
                (1 - feature_overlap) * (max_seq_length - special_tokens_count)
            )

        token_iterator = range(0, n_tokens, feature_len)

        for t in token_iterator:
            tokens = ex_tokens[t:t + max_seq_length - special_tokens_count]
            offsets = ex_offsets[t:t + max_seq_length - special_tokens_count]
            lengths = ex_lengths[t:t + max_seq_length - special_tokens_count]
            token_sw = ex_token_sw[t:t + max_seq_length - special_tokens_count]
            label_ids = ex_label_ids[t:t + max_seq_length -
                                     special_tokens_count]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            offsets += [-1]
            lengths += [-1]
            token_sw += [False]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                offsets += [-1]
                lengths += [-1]
                token_sw += [-1]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                offsets += [-1]
                lengths += [-1]
                token_sw += [False]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                offsets = [-1] + offsets
                lengths = [-1] + lengths
                token_sw = [False] + token_sw
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            pad_token_input_id = tokenizer.convert_tokens_to_ids([pad_token])[0]
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token_input_id] * padding_length) + input_ids
                input_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + input_mask
                segment_ids = (
                    [pad_token_segment_id] * padding_length
                ) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
                offsets = ([-1] * padding_length) + offsets
                lengths = ([-1] * padding_length) + lengths
                token_sw = ([False] * padding_length) + token_sw
            else:
                input_ids += [pad_token_input_id] * padding_length
                input_mask += [
                    0 if mask_padding_with_zero else 1
                ] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length
                offsets += [-1] * padding_length
                lengths += [-1] * padding_length
                token_sw += [False] * padding_length

            extra_features = []
            for i in range(len(ex_phi_locs)):
                extra_feature = create_extra_feature_vector(ex_phi_locs[i], offsets, lengths, token_sw)
                extra_features.append(extra_feature)


            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(offsets) == max_seq_length
            assert len(lengths) == max_seq_length
            assert len(token_sw) == max_seq_length
            if len(extra_features) > 0:
                assert len(extra_features[0]) == max_seq_length
                assert len(extra_features) == len(example.patterns)


            if n_obs < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info(
                    "subwords: %s", " ".join([str(x)[0] for x in token_sw])
                )
                logger.info(
                    "input_ids: %s", " ".join([str(x) for x in input_ids])
                )
                logger.info(
                    "input_mask: %s", " ".join([str(x) for x in input_mask])
                )
                logger.info(
                    "segment_ids: %s", " ".join([str(x) for x in segment_ids])
                )
                logger.info(
                    "label_ids: %s", " ".join([str(x) for x in label_ids])
                )
                logger.info("offsets: %s", " ".join([str(x) for x in offsets]))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_ids=label_ids,
                    input_offsets=offsets,
                    input_lengths=lengths,
                    input_subwords=token_sw, 
                    extra_feature=extra_features,
                )
            )
            n_obs += 1
    return features




