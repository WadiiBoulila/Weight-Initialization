from main import main
from training import evaluation_summary
import pandas as pd
import argparse
import os


# dataset_name, model_name, init_name=None, train_opt=True, eval_opt=True, save=True, overwrite=True, batch_size=16, learning_rate=0.0001, epochs=100, printing=True
# dataset_name, model_name='', avg_num=10, summary_save_path=log_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("-dn",'--dataset_name', help='dataset name should be in [ucmerced, aid, ksa, patternnet, wadii]', default='ucmerced', type=str)
    parser.add_argument("-mn",'--model_name', help='model username should be in [vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2, mobilenet_v2]', default='mobilenet_v2', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2', 'mobilenet_v2'])
    parser.add_argument("-wi",'--weight_init', help='weight initialization method should be ether xavier or he to use the famouse method, and you can choose any other name to use our proposed method. ex: custom', default='custom', type=str)
    parser.add_argument("-is",'--image_size', help='image size for the transforms', default=224, type=int)
    # running option 
    parser.add_argument("-tr","--train", help="training option (default: False)", default=False, action="store_true")
    parser.add_argument("-ev","--eval", help="evaluation option (default: False)", default=False, action="store_true")
    parser.add_argument("-evs","--eval_summary", help="evaluation summary option (default: False)", default=False, action="store_true")
    # hyperparameters
    parser.add_argument("-ep",'--epochs', help='training iteration number', default=100, type=int)
    parser.add_argument("-bs",'--batch_size', help='training/evaluation batch size', default=16, type=int)
    parser.add_argument("-lr",'--learning_rate', help='training learning rate ', default=0.0001, type=float)
    # extra
    parser.add_argument("-sv","--save", help="save the model and training history (default: True)", default=True, action="store_true")
    parser.add_argument("-ow","--overwrite", help="overwrite the current model with the same dataset_name, model_name, and init_name (default: True)", default=True, action="store_true")
    parser.add_argument("-pr","--printing", help="print the used hyperparameters, dataset details, and model training progress (default: True)", default=True, action="store_true")
    # summary
    parser.add_argument("-an",'--avg_num', help='number of the evaluation for taking the average', default=10, type=int)
    parser.add_argument("-sp",'--summary_save_path', help='save path of the summary file', default='./results/log/', type=str)
    
    args = parser.parse_args()
    
    if not (args.train or args.eval or args.eval_summary):
        print('please run the file properly. for more information: https://github.com/WadiiBoulila/Weight-Initialization')
    if args.train or args.eval:
        main(args.dataset_name, args.model_name, args.weight_init, args.train, args.eval, args.save, args.overwrite, args.image_size, args.batch_size, args.learning_rate, args.epochs, args.printing)
    if args.eval_summary:
        evaluation_summary(args.dataset_name, args.model_name, avg_num=args.avg_num, summary_save_path=args.summary_save_path)
    
     
