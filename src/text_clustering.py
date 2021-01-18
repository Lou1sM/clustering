import torch
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer
from transformers import LongformerModel
from sklearn.datasets import fetch_20newsgroups
from pdb import set_trace

#from transformers import LongformerForSequenceClassification
#model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True,output_hidden_states=True)
model = LongformerModel.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True,output_hidden_states=True).cuda()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model.config.max_position_embeddings = 10000
tokenizer.model_max_length = 10000


ng = fetch_20newsgroups()
tokenized_input = [tokenizer.encode(s) for s in ng.data[:10]]
max_in_batch = max([len(item) for item in tokenized_input])
attention_mask = torch.stack([torch.arange(max_in_batch) < len(item) for item in tokenized_input]).long().cuda()
padded = torch.tensor([item + [1]*(max_in_batch - len(item)) for item in tokenized_input]).long().cuda()
padded, attention_mask = pad_to_window_size(padded, attention_mask, 512, 1)
all_outputs = model(padded, attention_mask=attention_mask)
output = all_outputs[0]

print(max_in_batch)
print(output.shape)
# Zero out indices of padding
s = output*(attention_mask.unsqueeze(-1))
# Take mean of non-zero vectors
s = s.sum(axis=1)/(attention_mask.sum(axis=1).unsqueeze(1))
print(s.shape)
