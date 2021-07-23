# Code for the paper "Selective Pseudo-label Clustering"

The paper is available [here](https://arxiv.org/abs/2107.10692). The results it reports can be reproduced by cloning this repo and then running the following command once for each dataset, from the src/ directory:

    python train.py --dset <MNISTfull|USPS|FashionMNIST> --conc --num_aes 15

Downloading the datasets can fail with a HTTP error if there is a connection problem. This can normally be solved by simply rerunning the command a few times.

This command uses an ensemble size of 15, ie 15 autoencoders (aes), as do the results from the paper. If your gpu is not big enough for this and you get a cuda memory error, you can pass the ```--split``` argument to specify the maximum number of aes that are to be trained in parallel. Eg appending ```--split 5``` to the above would mean that 5 aes are trained in parallel, then another 5, then another 5. This will make training slower, but won't affect the results. 

Further cl args, and their descriptions, can be found in src/options.py.
