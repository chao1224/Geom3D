import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


class BertForSequenceRegression(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceRegression, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.loss_fct = torch.nn.MSELoss()

    def forward(self, 
        input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
        labels=None, output_attentions=None, output_hidden_states=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        outputs = self.regressor(pooled_output)
        if labels is not None:
            loss = self.loss_fct(outputs.view(-1), labels.view(-1))
            return loss
        else:
            return outputs