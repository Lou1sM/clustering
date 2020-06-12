import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--ae_range',type=int,nargs='+')
    group.add_argument('--aeids',type=int,nargs='+')
    group.add_argument('--num_aes',type=int)
    parser.add_argument('--NZ',type=int,default=50)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--clmbda',type=float,default=1.)
    parser.add_argument('--clusterer',type=str,default='HDBSCAN',choices=['HDBSCAN','GMM'])
    parser.add_argument('--conc',action='store_true')
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--disable_cuda',action='store_true')
    parser.add_argument('--dset',type=str,default='MNISTfull',choices=['MNISTfull','FashionMNIST','USPS','MNISTtest','CIFAR10','coil-100', 'letterAJ'])
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--exp_name',type=str,default='try')
    parser.add_argument('--gen_batch_size',type=int,default=100)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--inter_epochs',type=int,default=8)
    parser.add_argument('--max_meta_epochs',type=int,default=30)
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--noise',type=float,default=1.5)
    parser.add_argument('--patience',type=int,default=7)
    parser.add_argument('--pretrain_epochs',type=int,default=10)
    parser.add_argument('--rlmbda',type=float,default=1.)
    parser.add_argument('--save','-s',action='store_true')
    parser.add_argument('--scatter_clusters',action='store_true')
    parser.add_argument('--sections',type=int,nargs='+', default=[4])
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--short_epochs',action='store_true')
    parser.add_argument('--single',action='store_true')
    parser.add_argument('--ablation',type=str,choices=['none','sharing','filtering'],default='none')
    parser.add_argument('--show_gpu_memory',action='store_true')
    parser.add_argument('--split',type=int,default=-1)
    parser.add_argument('--reload_chkpt',type=str,default='none')
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--vis_pretrain',action='store_true')
    parser.add_argument('--vis_train',action='store_true')
    return parser.parse_args()