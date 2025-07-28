# src/model.py (FINAL and CORRECTED)
import torch
from torch import nn
from transformers import PreTrainedModel, AutoConfig, AutoModel

class MiniLayoutLM(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # --- CRITICAL CHANGE HERE ---
        # The name of this component MUST match the model type for weights to load.
        # For ELECTRA models, the name is "electra".
        self.electra = AutoModel.from_config(config)
        # ----------------------------
        
        # Layout embedding layers
        self.x_position_embeddings = nn.Embedding(1001, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(1001, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(1001, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(1001, config.hidden_size)

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        # Use the correctly named component
        text_outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = text_outputs[0]

        left_pos = self.x_position_embeddings(bbox[:, :, 0])
        upper_pos = self.y_position_embeddings(bbox[:, :, 1])
        right_pos = self.x_position_embeddings(bbox[:, :, 2])
        lower_pos = self.y_position_embeddings(bbox[:, :, 3])
        w_emb = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        h_emb = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])

        combined_embeddings = sequence_output + left_pos + upper_pos + right_pos + lower_pos + w_emb + h_emb
        
        pooled_output = self.dropout(combined_embeddings)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return (loss, logits) if loss is not None else (logits,)