import torch
from networks import pretrained_network
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os


save_path = './results/vis'
log_path = './results/log'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    

def dist_plot(dataset_name, model_name, trained_epochs, filter_epoch):
    # get history files names    
    csv_files = [csv_file for csv_file in os.listdir(log_path) if f'{dataset_name}-{model_name}' in csv_file and str(trained_epochs) in csv_file]
    # load history files
    dfs = [pd.read_csv(os.path.join(log_path, csv_file)) for csv_file in csv_files]    
    # plot dist lines
    plt.figure(figsize=(12, 5))
    for i, df in enumerate(dfs):
        # filter epochs
        df = df.head(filter_epoch)
        # get init name and labels
        init_name = csv_files[i].split('-')[2]
        best_acc = round(max(df['val_accuracy']), 3)
        label = '{}:{}'.format(init_name, best_acc)
        # add current df val_accuracy progress to the figure
        plt.plot(df['epoch'], df['val_accuracy'], label=label)
    plt.title(' '.join([dataset_name, model_name, str(filter_epoch), 'epochs']))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f'{save_path}/dist-{dataset_name}-{model_name}-epochs-{filter_epoch}-of-{trained_epochs}.png', dpi=360, bbox_inches='tight')
    plt.show()


def progress_plot(dataset_name, model_name, init_name, epochs=None):
    if not epochs:
        csv_files = [csv_file for csv_file in os.listdir(log_path) if f'{dataset_name}-{model_name}' in csv_file]
        epochs = max(int(name.split('-')[5]) for name in csv_files)
    csv_path = f'{log_path}/{dataset_name}-{model_name}-{init_name}-epochs-{epochs}-history.csv'
    df = pd.read_csv(csv_path)
    # plot lines
    plt.figure(figsize=(12, 5))
    plt.plot(df['epoch'], df['accuracy'], label='train: ' + str(round(max(df['accuracy']), 4)))
    plt.plot(df['epoch'], df['val_accuracy'], label='val:     ' + str(round(max(df['val_accuracy']), 4)))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'{save_path}/progress-dist-{dataset_name}-{model_name}-{init_name}-epochs-{epochs}.png', dpi=360, bbox_inches='tight')
    plt.show()


def cm_plot(dataset, model_name, init_name, epochs):
    # load model
    model_path = f'./results/checkpoints/{dataset.name}-{model_name}-{init_name}-epochs-{epochs}-best-model.pt'
    model = pretrained_network(model_name, len(dataset.classes), init_name)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # prediction
    y_pred = []
    y_true = []
    for inputs, labels in dataset.dataloaders['val']:
        output = model(inputs) # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # normalize
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # to dataframe
    df_cm = pd.DataFrame(cmn/np.sum(cmn) *10, index = [i for i in dataset.classes], columns = [i for i in dataset.classes])
    
    # plot
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=dataset.classes, yticklabels=dataset.classes, cmap='Blues')
    plt.title(' '.join([dataset.name, model_name, str(epochs), 'epochs']))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    plt.savefig(f'{save_path}/cm-dist-{dataset.name}-{model_name}-{init_name}-{epochs}-epochs.png', dpi=360, bbox_inches='tight')