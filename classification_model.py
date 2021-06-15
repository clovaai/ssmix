'''
SSMix
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
---
Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import torch
import torch.nn as nn
from transformers import *
from transformers.modeling_bert import (BertEmbeddings,
                                        BertLayer,
                                        BertPooler,
                                        BertConfig,
                                        BertPreTrainedModel,
                                        SequenceClassifierOutput,
                                        BaseModelOutputWithPooling,
                                        BaseModelOutput)

class AlteredBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = AlteredBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            inputs=None,
            inputs2=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            trace_grad=False,
            mixup_lambda=0,
            mixup_layer=-1,
            mix_embedding=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            inputs=inputs,
            inputs2=inputs2,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            trace_grad=trace_grad,
            mixup_lambda=mixup_lambda,
            mixup_layer=mixup_layer,
            mix_embedding=mix_embedding,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class AlteredBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = AlteredBertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        inputs=None,
        inputs2=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        trace_grad=False,
        mixup_lambda=0,
        mixup_layer=-1,
        mix_embedding=False,
    ):
        # Add exception for RoBERTa
        if inputs.get('token_type_ids') is None:
            inputs['token_type_ids'] = None
        if inputs2 is not None and inputs2.get('token_type_ids') is None:
            inputs2['token_type_ids'] = None

        input_ids, attention_mask, token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs[
            'token_type_ids']
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs2 is not None:
            input_ids2, attention_mask2, token_type_ids2 = inputs2['input_ids'], inputs2['attention_mask'], inputs2['token_type_ids']
            input_shape2 = input_ids2.size()
            if attention_mask2 is None:
                attention_mask2 = torch.ones(input_shape2, device=device)
            if token_type_ids2 is None:
                token_type_ids = torch.zeros(input_shape2, dtype=torch.long, device=device)
            embedding_output2 = self.embeddings(
                input_ids=input_ids2, position_ids=position_ids, token_type_ids=token_type_ids2,
                inputs_embeds=inputs_embeds
            )
            extended_attention_mask2: torch.Tensor = self.get_extended_attention_mask(attention_mask2, input_shape2,
                                                                                      device)
        else:
            embedding_output2, extended_attention_mask2 = None, None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        if trace_grad:
            embedding_output = embedding_output.detach().requires_grad_(True)
        if mix_embedding:
            embedding_output = mixup_lambda * embedding_output + (1 - mixup_lambda) * embedding_output2
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask | attention_mask2,
                                                                                      input_shape, device)
            assert(mixup_layer is -1)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            hidden_states2=embedding_output2,
            attention_mask=extended_attention_mask,
            attention_mask2=extended_attention_mask2,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mixup_lambda=mixup_lambda,
            mixup_layer=mixup_layer
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:] + (embedding_output,)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,)

class AlteredBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states=None,
        hidden_states2=None,
        attention_mask=None,
        attention_mask2=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        mixup_lambda=0,
        mixup_layer=-1
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            # General step
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            # step according to hidden layer mixup
            if mixup_layer != -1:
                assert(hidden_states2 is not None)
                assert(attention_mask2 is not None)

                if i <= mixup_layer:
                    layer_outputs2 = layer_module(hidden_states2, attention_mask2, layer_head_mask, encoder_hidden_states,
                                                  encoder_attention_mask, output_attentions)
                    hidden_states2 = layer_outputs2[0]
                if i == mixup_layer:
                    hidden_states = mixup_lambda * hidden_states + (1-mixup_lambda) * hidden_states2

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict: # We usually fall into this category
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class ClassificationModel(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', num_labels=2):
        super(ClassificationModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.num_labels = num_labels
        config.max_length = 128 # GLUE task
        self.bert = AlteredBertForSequenceClassification.from_pretrained(pretrained_model, config=config)

    def forward(self, *args, **kwargs):
        outputs = self.bert(*args, **kwargs)
        return outputs[0]
