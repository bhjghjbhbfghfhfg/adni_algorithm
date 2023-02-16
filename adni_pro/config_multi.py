import argparse

def get_args_multi():

    parser = argparse.ArgumentParser()

    parser.add_argument('--AD_dir', type=str,
                        help='subfolder of train or test dataset', default='AD/')
    parser.add_argument('--CN_dir', type=str,
                        help='subfolder of train or test dataset', default='CN/')
    parser.add_argument('--MCI_dir', type=str,
                        help='subfolder of train or test dataset', default='MCI/')


    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--class_num', type=int, help='class_num', default=2)
    # parser.add_argument('--seed', type=int, help='Seed', default=541)
    parser.add_argument('--gpu', type=str, help='GPU ID', default='1')


    parser.add_argument('--train_root_path', type=str, help='Root path for train dataset',
                        default='./multi_data_0211/train/')
    parser.add_argument('--val_root_path', type=str, help='Root path for val dataset',
                        default='./multi_data_0211/val/')
    parser.add_argument('--test_root_path', type=str, help='Root path for test dataset',
                        default='./multi_data_0211/test/')
    parser.add_argument('--batch_size', type=int, help='batch_size of data', default=2)
    parser.add_argument('--nepoch', type=int, help='Total epoch num', default=100)
    #  0: mri 1: pet 2: multi
    parser.add_argument('--state', type=int, help='single modality or multi modality', default=2)

    args = parser.parse_args()
    # args.seed = 541
    return args
