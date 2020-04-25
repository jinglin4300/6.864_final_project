from torch import nn
import torch
import numpy as np 
from bert_deid.crf import CRF
from transformers import BertModel, BertPreTrainedModel

class BERTBiLSTMCRF(BertPreTrainedModel):
    def __init__(self, config, method='concat_last_four', num_lstm_layers=2, lstm_bidirectional=True,dropout=0.1):
        super().__init__(config)
        method = method.lower()
        if method not in self._get_valid_methods():
            raise ValueError("Invalid method argument")
        config.output_hidden_states=True
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.method = method
        self.num_layers = num_lstm_layers
        self.bidirectional = lstm_bidirectional
        if self.bidirectional:
            hidden_size = self.hidden_size//2
        else:
            hidden_size = self.hidden_size
        self.rnn = nn.LSTM(
            bidirectional=self.bidirectional, 
            num_layers=self.num_layers, 
            input_size=self.hidden_size*4 if method=='concat_last_four' else self.hidden_size,
            hidden_size=hidden_size, 
            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        self.bert.eval()
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            hidden_states = outputs[2]
            embedding_output = hidden_states[0]
            hidden_layers = hidden_states[1:]

            if self.method == "embedding":
                encoded_layers = embedding_output # embedding
            elif self.method == "second_to_last":
                encoded_layers = hidden_layers[-2] # second to last layer 
            elif self.method == "sum_last_four":
                encoded_layers = torch.stack([hidden_layers[-4], hidden_layers[-3], hidden_layers[-2], hidden_layers[-1]], dim=0)
                encoded_layers = torch.sum(encoded_layers, dim=0)
            elif self.method == "sum_all": 
                # sum all 12 layers
                encoded_layers = torch.stack(list(hidden_layers), dim=0)
                encoded_layers = torch.sum(encoded_layers, dim=0)
            elif self.method == "last":
                encoded_layers = outputs[0] # last layer
            else:
                #concatenate last four layers
                encoded_layers = torch.cat([hidden_layers[-1], hidden_layers[-2], hidden_layers[-3], hidden_layers[-4]], dim=-1)

        enc, _ = self.rnn(encoded_layers)

        last_encoder_layer = enc # (batch_size, seq_length, hidden_size)
        # mask all -100
        mask = (labels>=0).long()
        # update all -100 to 0 to avoid indicies out-of-bound in CRF 
        labels = labels * mask
        mask = mask.to(torch.uint8) #.byte()

        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.hidden2label(last_encoder_layer)

        logits = torch.Tensor(self.crf.decode(emissions, mask=mask)).long() # (batch_size, seq_len, num_tags)
        print ('logit', logits.shape)

        # if labels is not None:
        #     log_likelihood = self.crf(emissions=emissions, tags=labels, mask=mask)
        #     tag_seqs = torch.Tensor(self.crf.decode(emissions, mask=mask)).long()# (batch_size, seq_len)
        #     print('log_likelihood.shape',log_likelihood.shape)
        #     print('tag_seqs.shape',tag_seqs.shape)
        #     print('log_likelihood',log_likelihood)
        #     print('tag_seqs',tag_seqs)
        #     return -1*log_likelihood, tag_seqs

        outputs = (logits,)
        print('outputs',outputs)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # only keep active parts of the loss
            attention_mask = None
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fn.ignore_index).type_as(labels)
                )
                loss = loss_fn(active_logits, active_labels)

            else:
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            
            outputs = (loss,) + outputs
        return outputs

    def _get_valid_methods(self):
        return set(["embedding", "second_to_last", "sum_last_four", "sum_all", "last", "concat_last_four"])

