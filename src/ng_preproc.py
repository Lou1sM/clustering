import numpy as np
import math
import torch
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer
from sklearn.datasets import fetch_20newsgroups
from pdb import set_trace
import torch.nn as nn

from transformers import LongformerForSequenceClassification

def chunk_tokenized_text(tokenized_text_and_label):
    """Expects tuples of text (str), label (int)."""
    tokenized_text,label = tokenized_text_and_label
    num_chunks = math.ceil(len(tokenized_text)/4096)
    chunks = [(tokenized_text[chunk_idx*4096:(chunk_idx+1)*4096],label) for chunk_idx in range(num_chunks)]
    if sum([len(x[0]) for x in chunks]) != len(tokenized_text): set_trace()
    return chunks


def get_hidden_feature_vec(model, batch_input, device):
    """Returns a triple, (prediction, hidden, text_feature_vec), each of which
    is batched along the first dimension.

    prediction (tensor) is the output of the classifier.
    hidden (list) is the hidden states from the 11th self-attention layer.
    text_feature_vec is the mean pool of the hidden states for each word in the text
    """

    max_in_batch = max([len(item) for item in batch_input])
    attention_mask = torch.stack([torch.arange(max_in_batch) < len(item) for item in batch_input]).long().to(device)
    padded = torch.tensor([item + [1]*(max_in_batch - len(item)) for item in batch_input]).long().to(device)
    padded, attention_mask = pad_to_window_size(padded, attention_mask, 512, 1)
    prediction, all_hiddens = model(padded, attention_mask=attention_mask)
    hidden = all_hiddens[11]
    text_feature_vec = hidden*(attention_mask.unsqueeze(-1))
    # Take mean of non-zero vectors
    text_feature_vec = text_feature_vec.sum(axis=1)/(attention_mask.sum(axis=1).unsqueeze(1))
    text_feature_vec = text_feature_vec.tolist()
    #set_trace()
    hidden = hidden.tolist()
    hidden = [h[:len(item)] for h,item in zip(hidden,batch_input)]
    return prediction, hidden, text_feature_vec


if __name__ == "__main__":
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True,output_hidden_states=True).cuda()
    model.longformer.encoder.layer = model.longformer.encoder.layer[:11]
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.model_max_length = 4096
    model.eval()
    model = nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ng = fetch_20newsgroups()
    tokenized_input = [tokenizer.encode(s) for s in ng.data]
    goodies = [(x,l) for x,l in zip(tokenized_input,ng.target) if len(x) <= 4096]
    badies = [(x,l) for x,l in zip(tokenized_input,ng.target) if len(x) > 4096]
    clipped_badies = [c for b in badies for c in chunk_tokenized_text(b)]
    clipped_tokenized_inputs_with_labels = goodies + clipped_badies
    inputs = [item[0] for item in clipped_tokenized_inputs_with_labels]
    labels = [item[1] for item in clipped_tokenized_inputs_with_labels]
    batch_size = 24
    feature_vecs_list = []
    hidden_list = []
    num_batches = math.ceil(len(inputs)/batch_size)
    for batch_idx in range(num_batches):
        print(batch_idx, num_batches)
        batch = inputs[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_labels = labels[batch_idx*batch_size:(batch_idx+1)*batch_size]
        #if batch_idx == 8: set_trace()
        prediction, hidden, feature_vecs = get_hidden_feature_vec(model,batch,device)
        #assert feature_vecs.shape == (batch_size,768) # batch_size * hidden_size
        feature_vecs_list.append(feature_vecs)
        for i,(h,l) in enumerate(zip(hidden,batch_labels)):
            dpoint_num = batch_idx*batch_size + i
            fname = f"../NG/preprocessed/ng{dpoint_num}.pt"
            t = torch.tensor(h)
            torch.save({'embeds':t, 'label':l},fname)
    #try:
        #feature_vecs_array = np.concatenate(feature_vecs_list,axis=0)
    #except: set_trace()
    #labels_array = np.array(labels)
    #np.save('preprocessed_20ng.npy',feature_vecs_array)
    np.save('preprocessed_20ng_hiddens.npy',hidden_list)
    np.save('preprocessed_20ng_labels.npy',labels)
    #with open('preprocessed_20ng_hiddens.npy','w') as f:
        #for _sublist1 in hidden_list:
            #f.write(' '.join(
