
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class BERTBiLSTM(BertPreTrainedModel):
    # def __init__(self, bert, top_rnns=False, vocab_size=None, device='cpu', finetuning=False):
    def __init__(self, config, num_lstm_layers=2, lstm_bidirectional=True, top_rnns=True, bert_finetuning=False):
        super().__init__(config)
        config.output_hidden_states=True
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.num_layers = num_lstm_layers
        self.bidirectional = lstm_bidirectional
        self.top_rnns = top_rnns
        if self.top_rnns:
            if self.bidirectional:
                hidden_size = self.hidden_size//2
            else:
                hidden_size = self.hidden_size
            self.rnn = nn.LSTM(
                bidirectional=self.bidirectional, 
                num_layers=self.num_layers, 
                # input_size=self.hidden_size, # last layer, embeddings
                input_size=self.hidden_size*4,  # concat last four layers
                hidden_size=hidden_size, 
                batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.finetuning = bert_finetuning

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):

        if self.training and self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            encoded_layers, _ = self.bert(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
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

                
                # encoded_layers = embedding_output # embedding
                
                # encoded_layers = hidden_layers[-2] # second to last layer 

                # sum last four layers
                # encoded_layers = torch.stack([hidden_layers[-4], hidden_layers[-3], hidden_layers[-2], hidden_layers[-1]], dim=0)
                # encoded_layers = torch.sum(encoded_layers, dim=0)

                # sum all 12 layers
                # encoded_layers = torch.stack(list(hidden_layers), dim=0)
                # encoded_layers = torch.sum(encoded_layers, dim=0)
                
                # encoded_layers = outputs[0] # last layer
                
                #concatenate last four layers
                encoded_layers = torch.cat([hidden_layers[-1], hidden_layers[-2], hidden_layers[-3], hidden_layers[-4]], dim=-1)

        if self.top_rnns:
            enc, _ = self.rnn(encoded_layers)
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