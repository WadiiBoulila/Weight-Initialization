import torch
from torch.nn import CrossEntropyLoss
from networks import pretrained_network
from datasets import RemoteSensingDataset
from utils import cm_to_dict, performance_report
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import copy
import os


log_path = './results/log/'
checkpoints_path = './results/checkpoints/'


def train(model, criterion, optimizer, scheduler, dataset, device, num_epochs=10, printing=True):
    # history dict
    history = {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    }
    # initial variables
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # train/val variables
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    # main training/validation loop
    for epoch in range(num_epochs):
        if printing:
            print("Epoch {}/{}".format(epoch, num_epochs))
        # reset epoch accuracy and loss
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        # set model training status for the training
        model.train(True)
        # training iterations
        for i, data in enumerate(dataset.dataloaders['train']):
            # extract images and labels
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            # Clear gradients
            optimizer.zero_grad()
            # predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            # compute loss and back propagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # calculate training loss and accuracy
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)
            # free some memory
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        # average training loss and accuracy
        # * 2 as we only used half of the dataset
        avg_loss = loss_train / dataset.dataset_sizes['train']
        avg_acc = acc_train / dataset.dataset_sizes['train']
        # change model training status for the evaluation
        model.train(False)
        model.eval()
        # validation iterations
        for i, data in enumerate(dataset.dataloaders['val']):
            # extract images and labels
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            # Clear gradients
            optimizer.zero_grad()
            # predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            # compute loss
            loss = criterion(outputs, labels)
            # calculate training loss and accuracy
            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data)
            # free some memory
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        # average validation loss and accuracy
        avg_loss_val = loss_val / dataset.dataset_sizes['val']
        avg_acc_val = acc_val / dataset.dataset_sizes['val']
        # printing
        if printing:
            print("\tloss: {:.4f}  -  accuracy: {:.4f}  -  val_loss: {:.4f}  -  val_accuracy: {:.4f}".format(avg_loss, avg_acc, avg_loss_val, avg_acc_val))
        # update best accuracy
        if avg_acc_val > best_acc:
            if printing:
                print("\tval_accuracy improved from {:.4f} to {:.4f}".format(best_acc, avg_acc_val))
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            if printing:
                print("\tval_accuracy did not improve from {:.4f}".format(best_acc))
        # save progress in history
        history['epoch'].append(epoch)
        history['loss'].append(np.float64(avg_loss).item())
        history['accuracy'].append(np.float64(avg_acc).item())
        history['val_loss'].append(np.float64(avg_loss_val).item())
        history['val_accuracy'].append(np.float64(avg_acc_val).item())
    # calculate training time
    elapsed_time = time.time() - since
    if printing:
        print("\n[INFO]  Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        print("[INFO]  Best accuracy: {:.4f}".format(best_acc))
    # load best weight and return it
    model.load_state_dict(best_model_wts)
    return model, history


def evaluate(model, criterion, dataset, device, printing=True):
    # initial variables
    since = time.time()
    loss_test = 0
    acc_test = 0
    cm = torch.zeros(len(dataset.classes), len(dataset.classes))
    # testing iterations
    for i, data in enumerate(dataset.dataloaders['val']):
        # set model training status to False for the evaluation
        model.train(False)
        model.eval()
        # extract images and labels
        inputs, labels = data
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        # predictions
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        # calculate loss
        loss = criterion(outputs, labels)
        # calculate testing loss and accuracy
        loss_test += loss.item()
        acc_test += torch.sum(preds == labels.data)
        # calculate confusion matrix
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
        # free some memory
        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
    # average testing loss and accuracy
    loss_test = loss_test / dataset.dataset_sizes['val']
    acc_test = acc_test / dataset.dataset_sizes['val']
    # calculate precision, recall, and f1
    cm_dict = cm_to_dict(cm, dataset.classes)
    precision, recall, f1, _ = performance_report(cm_dict, mode = 'Macro')
    if printing:
        print()
        # calculate training time 
        elapsed_time = time.time() - since
        print("[INFO]  Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    return round(acc_test.item(), 4), round(precision.item(), 4), round(recall.item(), 4), round(f1.item(), 4), round(loss_test, 4)
    
    
def evaluation_summary(dataset_name, model_name='', avg_num=10, summary_save_path=log_path):
    # get path of all models that contains the entered init, model, and epochs 
    models_paths = [checkpoints_path + name for name in os.listdir(checkpoints_path) if dataset_name in name and model_name in name]
    # get dataset object
    dataset = RemoteSensingDataset(dataset_name, 16, printing=False)
    # define loss function
    criterion = CrossEntropyLoss()
    # df columns
    columns = ['filename', 'model_name', 'init_name', 'epochs', 'train_accuracy', 'val_accuracy', 'avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1', 'avg_loss']
    arr = []
    # loop over all saved weights
    for best_weights in tqdm(models_paths, desc='[INFO]  getting models evaluations:'):
        # record data
        filename = best_weights.split('/')[-1]
        model_name = filename.split('-')[1]
        init_name = filename.split('-')[2]
        train_epochs = filename.split('-')[4]
        # load the model
        model = pretrained_network(model_name, len(dataset.classes), init_name)
        model.load_state_dict(torch.load(best_weights, map_location=torch.device('cpu')))
        # compute average measures
        avg_accuracy, avg_precision, avg_recall, avg_f1, avg_loss = 0, 0, 0, 0, 0
        for i in range(avg_num):
            # get its evaluations
            accuracy, precision, recall, f1, loss = evaluate(model, criterion, dataset, torch.device('cpu'), False)
            avg_accuracy += accuracy
            avg_precision += precision
            avg_recall += recall
            avg_f1 += f1
            avg_loss += loss
        avg_accuracy = avg_accuracy / avg_num
        avg_precision = avg_precision / avg_num
        avg_recall = avg_recall / avg_num
        avg_f1 = avg_f1 / avg_num
        avg_loss = avg_loss / avg_num
        # read best val/train accuracy from history
        history = pd.read_csv(f'{log_path}/{dataset_name}-{model_name}-{init_name}-epochs-{train_epochs}-history.csv')
        best_row = history[history['val_accuracy'] == max(history['val_accuracy'])].head(1)
        # add row to array
        arr.append([filename, model_name, init_name, train_epochs, round(best_row['accuracy'].item(), 4), round(best_row['val_accuracy'].item(), 4), avg_accuracy, avg_precision, avg_recall, avg_f1, avg_loss])
    # arr to df
    df = pd.DataFrame(arr, columns=columns)
    df = df.sort_values(by=['model_name'])
    if not os.path.exists(summary_save_path):
        os.makedirs(summary_save_path)
    summary_file_save_path = os.path.join(summary_save_path, f'{dataset_name.upper()}-EVALUATION_SUMMARY.csv')
    df.to_csv(summary_file_save_path, index=False)
    print('[INFO]  Saved:  ', summary_file_save_path)