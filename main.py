import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from networks import pretrained_network
from datasets import RemoteSensingDataset
from training import train, evaluate
import pandas as pd
import os

# save paths
checkpoint_path = './results/checkpoints'
log_path = './results/log'

def main(dataset_name, model_name, init_name=None, train_opt=True, eval_opt=True, save=True, overwrite=True, batch_size=16, learning_rate=0.0001, epochs=100, printing=True, return_acc=False):
    if printing:
        print('*'*50)
        print(' '*50, 'model name:    ', model_name.upper())
        print(' '*50, 'dataset name:  ', dataset_name.upper())
        print(' '*50, 'init name:     ', init_name.upper() if init_name else 'NONE')
        print(' '*50, 'batch size:    ', batch_size)
        print(' '*50, 'learning rate: ', learning_rate)
        print(' '*50, 'epochs:        ', epochs)
        print('*'*50)
    # save paths
    model_save_path = f'{checkpoint_path}/{dataset_name}-{model_name}-{init_name}-epochs-{epochs}-best-model.pt'
    history_save_path = f'{log_path}/{dataset_name}-{model_name}-{init_name}-epochs-{epochs}-history.csv'
    # exit the function if the overwrite not allowed
    if not overwrite and save and os.path.exists(history_save_path):
        if printing:
            print('[INFO]  The history file of this experiment is already exist.')
        if return_acc:
            return max(pd.read_csv(history_save_path)['val_accuracy'])
        else:
            return 0
    # create folders if does not exists
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # train/eval options
    if train_opt or eval_opt:
        # get train and val dataloaders
        dataset = RemoteSensingDataset(dataset_name, batch_size, printing=printing)
        # try to use gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # initialize the model
        model = pretrained_network(model_name, len(dataset.classes), init_name)
        model.to(device)
        # loss function and optimizer
        criterion = CrossEntropyLoss()
        optimizer_ft = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        if train_opt:
            # training
            if printing:
                print('\n\n[INFO]  Model Training...')
            model, history = train(model, criterion, optimizer_ft, scheduler, dataset, device, epochs, printing)
            if save:
                # save model and training history
                torch.save(model.state_dict(), model_save_path)
                pd.DataFrame.from_dict(history).to_csv(history_save_path, index=False)
                if printing:
                    print('\n[INFO]  Saved:  ', model_save_path)
                    print('[INFO]  Saved:  ', history_save_path)
        if eval_opt:
            # evaluation
            if printing:
                print('\n\n[INFO]  Model Evaluation...')
            accuracy, precision, recall, f1, loss = evaluate(model, criterion, dataset, device, printing=printing)
            if printing:
                print("[INFO]  Accuracy:  {:.4f}".format(accuracy))
                print("[INFO]  Precision: {:.4f}".format(precision))
                print("[INFO]  Recall:    {:.4f}".format(recall))
                print("[INFO]  F1 Score:  {:.4f}".format(f1))
                print("[INFO]  Loss:      {:.4f}\n".format(loss))

        if return_acc:
            return max(history['val_accuracy'])

        