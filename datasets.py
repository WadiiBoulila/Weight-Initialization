from torch.cuda.random import seed
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import os

# fixed paths
base_download_path = 'data/compressed/'
base_images_folder_path = 'data/uncompressed/'
base_splitted_data_path = 'data/splitted/'

class RemoteSensingDataset():
    def __init__(self, dataset_name='ucmerced', batch_size=16, printing=False):
        # lowercase user input
        dataset_name = dataset_name.lower()
        # properties setup
        if dataset_name == 'ucmerced':
            self.name = dataset_name
            self.download_link = 'http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip'
            self.download_path = base_download_path
            self.zip_file = 'UCMerced_LandUse.zip'
            self.unzip_path = base_images_folder_path
            self.images_folder_path = os.path.join(base_images_folder_path, 'UCMerced_LandUse/Images')
            self.splitted_data_path = os.path.join(base_splitted_data_path, dataset_name)
            self.gdrive = False
        elif dataset_name == 'aid':
            self.name = dataset_name
            self.download_link = 'https://public.dm.files.1drv.com/y4m0a2dwQ9slMClrnr37pLBLwdoDeAtqb-HxoQhYrkMt0xmyfB_FqY6eWISm2nTspsQpunwTwMXfcxJ3zVo0Jb-4xoJ0jkIHAWKujQVkKn7FxFmwpqb0txsmf6PGmDBoIXEbwd4scXdg9tLxgKir-bB7Snm6jgP5BythY0SjdHEJtizPwIqoav3MfVzPNvjhJ1VIkn80TcHDMPKEjTdkHXm5FIFhgLm2-ReP8SfjUlayck'
            self.download_path = base_download_path
            self.zip_file = 'AID.zip'
            self.unzip_path = base_images_folder_path
            self.images_folder_path = os.path.join(base_images_folder_path, 'AID')
            self.splitted_data_path = os.path.join(base_splitted_data_path, dataset_name)
            self.gdrive = False
        elif dataset_name == 'ksa':
            self.name = dataset_name
            self.download_link = '1H400Qamkl7oVCvvMzcQ72N0-jEZuegk5'
            self.download_path = base_download_path
            self.zip_file = 'KSA.zip'
            self.unzip_path = base_images_folder_path
            self.images_folder_path = os.path.join(base_images_folder_path, 'KSA')
            self.splitted_data_path = os.path.join(base_splitted_data_path, dataset_name)
            self.gdrive = True
        elif dataset_name == 'pattern':
            self.name = dataset_name
            self.download_link = '1m-q6NU0I_VezSwExuiz4Z3Nf_kxH7XE8'
            self.download_path = base_download_path
            self.zip_file = 'PatternNet.zip'
            self.unzip_path = base_images_folder_path
            self.images_folder_path = os.path.join(base_images_folder_path, 'PatternNet')
            self.splitted_data_path = os.path.join(base_splitted_data_path, dataset_name)
            self.gdrive = True
        elif dataset_name == 'wadii':
            self.name = dataset_name
            self.download_link = ''
            self.download_path = base_download_path
            self.zip_file = 'SatelliteDataset.zip'
            self.unzip_path = base_images_folder_path
            self.images_folder_path = os.path.join(base_images_folder_path, 'SatelliteDatasetForEncryption')
            self.splitted_data_path = os.path.join(base_splitted_data_path, dataset_name)
            self.gdrive = True
        elif dataset_name == 'cifar10':
            self.name = dataset_name
            self.download_link = ''
            self.download_path = base_download_path
            self.zip_file = 'cifar10.zip'
            self.unzip_path = base_images_folder_path
            self.images_folder_path = os.path.join(base_images_folder_path, 'cifar10')
            self.splitted_data_path = os.path.join(base_splitted_data_path, dataset_name)
            self.gdrive = False
        elif dataset_name == 'cifar100':
            self.name = dataset_name
            self.download_link = ''
            self.download_path = base_download_path
            self.zip_file = 'cifar100.zip'
            self.unzip_path = base_images_folder_path
            self.images_folder_path = os.path.join(base_images_folder_path, 'cifar100')
            self.splitted_data_path = os.path.join(base_splitted_data_path, dataset_name)
            self.gdrive = False
        else:
            raise Exception("Your target dataset information is not found. Go to dataset.py and add your dataset information. Then, try again.")
        
        self.dataloaders, self.dataset_sizes, self.classes = self._get_data(batch_size, printing)

    def _get_data(self, batch_size, printing):
        # chack for data availability
        if not os.path.exists(os.path.join(self.splitted_data_path, 'train')):
            self._split()
        # transforms (data augmentation)
        data_transforms = {
            'train': transforms.Compose([
                # transforms.Resize(224),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            'val': transforms.Compose([
                # transforms.Resize(224),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        }
        # initialize dataseta
        image_datasets = {
            x: ImageFolder(
                os.path.join(self.splitted_data_path, x), 
                transform=data_transforms[x]
            )
            for x in ['train', 'val']
        }
        # initialize dataloaders
        dataloaders = {
            x: DataLoader(
                image_datasets[x], batch_size=batch_size,
                shuffle=True, num_workers=2,
            )
            for x in ['train', 'val']
        }
        # printing
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        classes = image_datasets['train'].classes
        if printing:
            for x in ['train', 'val']:
                print("[INFO]  Loaded {} images under {}".format(dataset_sizes[x], x))
            print("[INFO]  Classes: ", ''.join(['\n\t\t'+i for i in classes]), '\n\n')
        # return dataloaders to use it in the training
        return dataloaders, dataset_sizes, classes

    def _split(self):
        # check if the data folder is existed
        if not os.path.exists(self.images_folder_path):
            # self._download() # download zip data
            self._unzip() # unzip the data
        # split the data folder into train and val
        print('[INFO]  Splitting {} dataset...'.format(self.name))
        if not os.path.exists(self.images_folder_path):
            os.makedirs(self.images_folder_path)
        import splitfolders
        splitfolders.ratio(self.images_folder_path, output=self.splitted_data_path, ratio=(0.8, 0.2), seed=1998)
        print('')

    def _download(self):
        # download the dataset from its offecial url
        print('[INFO]  Downloading {} dataset...'.format(self.name))
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)
        if self.gdrive:
            import gdown
            gdown.download(id=self.download_link, output=self.download_path, quiet=False)
        else:
            import wget
            wget.download(self.download_link, out=self.download_path)

    def _unzip(self):
        import zipfile
        with zipfile.ZipFile(os.path.join(self.download_path, self.zip_file), 'r') as file:
            file.extractall(self.unzip_path)    
