import os
from datasets import RetinalDataset
import torch.utils.data as data


class Config(object):
    gpu = '0,1,2,3,4,5,6,7'

    def __getitem__(self, key):
        return self.__getattribute__(key)


class ConfigMultiMD(Config):
    """Configuration for multi-disease retinal diagnosis with paired CFP-FFA data."""

    disease_dict = ['AMD', 'ME', 'VH', 'HighMyopia', 'CSC', 'DR', 'RVO']

    # Data and label path
    datapath = '.../dataset/large'
    # Model save path
    resultPath = '../result/MD'

    fold = ['paired', '0']

    # Image modalities
    modals = ['CFP_enhanced', 'FFA_select']
    modal_format = {
        'CFP_enhanced': ['1', '2'],
        'FFA_select': ['35', '55', 'three']
    }

    rotate = False
    imgSize = 224
    batchSize = 16
    num_class = 7

    pretrained = True
    patch_num = 3

    # Threshold for FFA background ratio (frames with excessive black background are filtered)
    background_ratio = 0.8

    K = 4000
    dim_embedding = 512
    dim_vq = 512

    lengthFA = 3
    num_model = [1]
    classLabel = [0, 1, 2]
    num_enface = 1
    in_channels = 1

    # Transformer settings (for potential extensions)
    dim_emb = 512
    num_head = 4
    droopout_rate = 0.1
    former_layers = 3

    # Training hyperparameters
    num_epochs = 60
    base_lr = 1e-4
    momentum = 0.9
    weight_decay = 0.001

    saveName = '_' + 'BS' + str(batchSize) + '-lr' + str(base_lr) + '-PNum' + str(patch_num)

    workers = 5

    # Dataset and dataloader construction
    dataset_train = RetinalDataset(
        disease_dict, datapath, modals, modal_format, fold, lengthFA,
        back_ratio=background_ratio, imgSize=imgSize, isTraining='train', isRotate=rotate)
    dataset_train_push = RetinalDataset(
        disease_dict, datapath, modals, modal_format, fold, lengthFA,
        back_ratio=background_ratio, imgSize=imgSize, isTraining='train_push', isRotate=rotate)
    dataset_validation = RetinalDataset(
        disease_dict, datapath, modals, modal_format, fold, lengthFA,
        back_ratio=background_ratio, imgSize=imgSize, isTraining='validation')
    dataset_test = RetinalDataset(
        disease_dict, datapath, modals, modal_format, fold, lengthFA,
        back_ratio=background_ratio, imgSize=imgSize, isTraining='test')

    dataloader_train_push = data.DataLoader(
        dataset=dataset_train_push, batch_size=batchSize,
        shuffle=False, num_workers=workers, drop_last=True)
    dataloader_valid = data.DataLoader(
        dataset=dataset_validation, batch_size=8,
        num_workers=workers, shuffle=False)
    dataloader_test = data.DataLoader(
        dataset=dataset_test, batch_size=40,
        num_workers=workers, shuffle=False)


mapping = {
    'MRD': ConfigMultiMD
}

# Select configuration via environment variable
APP_ENV = os.environ.get('APP_ENV', 'MRD')
config = mapping[APP_ENV]()
