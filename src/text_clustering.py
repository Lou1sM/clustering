import numpy as np
import math
import torch
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer
from sklearn.datasets import fetch_20newsgroups
from pdb import set_trace

from transformers import LongformerForSequenceClassification
with torch.cuda.device(3):
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True,output_hidden_states=True).cuda()
    model.longformer.encoder.layer = model.longformer.encoder.layer[:11]
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model.config.max_position_embeddings = 60000
    tokenizer.model_max_length = 60000
    model.eval()

    def get_hidden_feature_vec(batch_input):
        max_in_batch = max([len(item) for item in batch_input])
        attention_mask = torch.stack([torch.arange(max_in_batch) < len(item) for item in batch_input]).long().cuda()
        padded = torch.tensor([item + [1]*(max_in_batch - len(item)) for item in batch_input]).long().cuda()
        padded, attention_mask = pad_to_window_size(padded, attention_mask, 512, 1)
        all_outputs = model(padded, attention_mask=attention_mask)
        hidden = all_outputs[1][11]
        s = hidden*(attention_mask.unsqueeze(-1))
        # Take mean of non-zero vectors
        s = s.sum(axis=1)/(attention_mask.sum(axis=1).unsqueeze(1))
        return s

    ng = fetch_20newsgroups()
    tokenized_input = [tokenizer.encode(s) for s in ng.data]
    batch_size = 8
    feature_vecs_list = []
    num_batches = math.ceil(len(tokenized_input)/batch_size)
    for batch_idx in range(num_batches):
        try:
            print(batch_idx, num_batches)
            batch = tokenized_input[batch_idx*batch_size:(batch_idx+1)*batch_size]
            feature_vecs = get_hidden_feature_vec(batch)
            assert feature_vecs.shape == (batch_size,768) # batch_size * hidden_size
            feature_vecs_list.append(feature_vecs.detach().cpu().numpy())
        except: set_trace()
    feature_vecs_array = np.stack(feature_vecs_list)
    np.save('preprocessed_20ng.npy',feature_vecs_array)
