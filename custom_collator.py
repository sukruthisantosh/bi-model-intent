#!/usr/bin/env python3
"""
Custom Data Collator for Entity Recognition Training
===================================================

Handles variable length sequences properly during training.
"""

import torch
from transformers import DataCollatorForLanguageModeling

class EntityRecognitionCollator(DataCollatorForLanguageModeling):
    """Custom collator for entity recognition training."""
    
    def __call__(self, features):
        # Get the maximum length in this batch
        max_length = max(len(feature['input_ids']) for feature in features)
        
        # Pad all sequences to the same length
        batch = {}
        for key in ['input_ids', 'attention_mask', 'labels']:
            batch[key] = []
            for feature in features:
                sequence = feature[key]
                # Pad with appropriate values
                if key == 'labels':
                    # Use -100 for padding in labels (ignored by loss function)
                    padded = sequence + [-100] * (max_length - len(sequence))
                else:
                    # Use pad_token_id for input_ids and 0 for attention_mask
                    if key == 'input_ids':
                        pad_value = self.tokenizer.pad_token_id
                    else:
                        pad_value = 0
                    padded = sequence + [pad_value] * (max_length - len(sequence))
                batch[key].append(padded)
        
        # Convert to tensors
        for key in batch:
            batch[key] = torch.tensor(batch[key])
        
        return batch
