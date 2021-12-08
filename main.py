import argparse
from scripts.train import train
from scripts.test import test
from scripts.eval import eval
if __name__ == '__main__':
    '''
    模型参数配置
    '''
    parser = argparse.ArgumentParser(description='FDANET')

    parser.add_argument('--model', nargs='?', type=str, default='fdanet',
                        choices=('fdanet', 'reg-only'),
                        help='Model to use [\'fdanet, reg-only\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='7S',
                        choices=('7S', '12S', 'my'), help='Dataset to use')
    parser.add_argument('--scene', nargs='?', type=str, default='heads', help='Scene')
    
    parser.add_argument('--flag', nargs='?', type=str, required=True,
                        choices=('train','test'), help='train or test')
    parser.add_argument('--eval', nargs='?', type=str, default=True, help='train or test')
    parser.add_argument('--init_lr', nargs='?', type=float, default=0.0004,
                        help='Initial learning rate')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--aug', nargs='?', type=bool, default=True,
                        help='w/ or w/o data augmentation')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to saved model to resume from')  
    parser.add_argument('--data_path', required=True, type=str,
                        help='Path to dataset')
    parser.add_argument('--log_summary', default='progress_log_summary.txt',
                        metavar='PATH',
                        help='txt where to save per-epoch stats')
    parser.add_argument('--train_id', nargs='?', type=str, default='',
                        help='An identifier string')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=2000,
                        help='the number of epoch to train')
    parser.add_argument('--output', nargs='?', type=str, default='./',
                        help='Output directory')        # 误差文件保存路径
    args = parser.parse_args()

    if args.dataset == '7S':
        if args.scene not in ['chess', 'heads', 'fire', 'office', 'pumpkin',
                              'redkitchen', 'stairs']:
            print('Selected scene is not valid.')
            sys.exit()

    if args.dataset == '12S':
        if args.scene not in ['apt1/kitchen','apt1/living','apt2/bed','apt2/kitchen',
                              'apt2/living','apt2/luke','office1/gates362',
                              'office1/gates381','office1/lounge','office1/manolis',
                              'office2/5a','office2/5b']:
            print('Selected scene is not valid.')
            sys.exit()
    
    if args.dataset == 'my':
        if args.scene not in ['room']:
            print('Selected scene is not valid.')
            sys.exit()
    
    if args.flag == 'train':
        train(args)
    elif args.flag == 'test':
        test(args)
    