
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class BERTBiLSTM(BertPreTrainedModel):
    # def __init__(self, bert, top_rnns=False, vocab_size=None, device='cpu', finetuning=False):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.num_layers = config.num_lstm_layers
        self.bidirectional = config.lstm_bidirectional
        self.top_rnns = config.top_rnns
        if self.top_rnns:
            if self.bidirection:
                hidden_size = self.hidden_size//2
            else:
                hidden_size = self.hidden_size
            self.rnn = nn.LSTM(
                bidirectional=self.bidirectional, 
                num_layers=self.num_layers, 
                input_size=self.hidden_size, 
                hidden_size=hidden_size, 
                batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.device = config.device
        self.finetuning = config.bert_finetuning

    def forward(self, input_ids, attention_mask, token_type_ids, labels):

        if self.training and self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            encoded_layers, _ = self.bert(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                enc = encoded_layers[-1]

        if self.top_rnns:
            enc, _ = self.rnn(enc)
        logits = self.classifier(enc)

        outputs = (logits,)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # only keep active parts of the loss
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