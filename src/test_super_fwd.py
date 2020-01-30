import utils
from time import time
from pdb import set_trace
import torch
import torch.nn as nn

s = utils.SuperAE(5,50,'cuda')
d = utils.get_mnist_dset()
x = d[0][0][None]
latents = s.encode(x)
pred = s.decode_list(latents)

dl = utils.get_mnist_dloader()


opt = torch.optim.Adam(s.parameters())
lf = nn.L1Loss()
start_time = time()
for epoch in range(3):
    total_loss = 0.
    for i,(xb,yb,idx) in enumerate(dl):
        preds = s(xb)
        loss = sum([lf(pred,xb) for pred in preds])
        total_loss = total_loss*(i+1)/(i+2) + loss.item()/(i+2)
        loss.backward(); opt.step(); opt.zero_grad()
    print(total_loss)
    print(utils.asMinutes(time() - start_time))

torch.save({'superae':s},f'superae_{num_epochs}.pt')
