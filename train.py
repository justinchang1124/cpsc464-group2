import os
import time
import argparse
import traceback
import pickle
import random
import logging
import pprint

import sys
sys.path.append('C:/Users/justin/PycharmProjects/cpsc464-group2/nyu_algos')

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, MultiStepLR
from torch import amp

from utilities.callbacks import History, RedundantCallback, resolve_callbacks, EvaluateEpoch
from utilities.warmup import GradualWarmupScheduler
from utilities.utils import boolean_string, save_checkpoint
from networks import losses
from networks.models import MRIModels, MultiSequenceChannels


class ParamTest:
    pass

def create_tensor(np_array):
    return torch.from_numpy(np_array).float()

def create_empty_tensor(shape):
    return create_tensor(np.empty(shape))

def create_model():
    s_params = ParamTest()
    s_params.weights = False
    s_params.architecture = '3d_resnet18_fc'
    s_params.resnet_groups = 16 # default: 16
    s_params.input_type = 'birads'  # 5 classes (0, 1, 2, 3, 4)
    s_params.topk = 10 # default: 10
    s_params.network_modification = None
    return MRIModels(s_params, in_channels=1, num_classes=5, force_bottleneck=False, inplanes=64).model

# model = create_model()
#

# x = create_tensor((1, 1, 1, 256, 256))
# hmm = model(x)

loss = nn.CrossEntropyLoss()
predict = torch.tensor([[100, 100, 100, 100, 1000]]).float()
target = torch.tensor([[0, 0, 0, 0, 1]]).float()
x = loss(predict, target)

from sklearn import model_selection

# project code
import dcm_utils
import read_metadata

abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = os.path.join(abs_proj_path, data_path)
dcm_files = dcm_utils.dcm_dir_list(abs_data_path)
n_dcm_files = len(dcm_files)

dcm_files_data = []
for file in dcm_files:
    dcm_files_data.append(file.split('\\'))

dcm_study_dict = {}
for i in range(n_dcm_files):
    key = os.path.join(*dcm_files_data[i][:-1])
    input_list = dcm_study_dict.get(key)
    if input_list is None:
        input_list = []
    input_list.append(i)
    dcm_study_dict[key] = input_list

median_indices = []
for key in dcm_study_dict:
    input_list = dcm_study_dict[key]
    # med_index = len(input_list) // 2
    # num_per_study = 40
    # get the median 50 elements
    i1 = 0  # int(med_index - num_per_study // 2)
    i2 = len(input_list)  # int(med_index + num_per_study // 2)
    sep_len = 8
    median_indices += input_list[i1:i2:sep_len]  # space out b/c we don't want too-similar images

# truncate
tn_dcm_files = len(median_indices)
# t_dcm_files = []
t_dcm_files_data = []
for i in median_indices:
    # t_dcm_files.append(dcm_files[i])
    t_dcm_files_data.append(dcm_files_data[i])


# open metadata
global_ids = []
for i in range(tn_dcm_files):
    global_ids.append(t_dcm_files_data[i][0])
labels, groups = read_metadata.get_labels_groups(global_ids)
abs_dcm_files = dcm_utils.prepend_abs(abs_data_path, dcm_files)
n_classes = len(set(labels))

for i in range(tn_dcm_files):
    if labels[i] < 0:
        labels[i] = 0

# partition into train / test / split
unique_ids = list(set(global_ids))
train_ids, test_ids = model_selection.train_test_split(
    unique_ids, test_size=0.2) # random_state?
train_ids, valid_ids = model_selection.train_test_split(
    train_ids, test_size=0.25)

X_train_files = []
X_valid_files = []
X_test_files = []
y_train = []
y_valid = []
y_test = []
z_train = []
z_valid = []
z_test = []

for i in range(tn_dcm_files):
    gid = global_ids[i]
    if gid in train_ids:
        X_train_files.append(abs_dcm_files[i])
        y_train.append(labels[i])
        z_train.append(groups[i])
    if gid in valid_ids:
        X_valid_files.append(abs_dcm_files[i])
        y_valid.append(labels[i])
        z_valid.append(groups[i])
    if gid in test_ids:
        X_test_files.append(abs_dcm_files[i])
        y_test.append(labels[i])
        z_test.append(groups[i])


def to_one_hot(my_int):
    a = torch.tensor([my_int])
    return torch.nn.functional.one_hot(a, num_classes=5)

class MyDataset(Dataset):

    def __init__(self, t_abs_dcm_files, t_labels):
        self.abs_dcm_files = t_abs_dcm_files
        self.labels = t_labels
        super().__init__()

    def __len__(self):
        return len(self.abs_dcm_files)

    def __getitem__(self, idx):
        x_file = self.abs_dcm_files[idx]
        x_image = dcm_utils.open_dcm_image(x_file)
        result = np.empty((1, 1, 256, 256)) # num_channels, z, x, y
        result[0, 0, :128, :128] = x_image
        label_hot_np = [0, 0, 0, 0, 0]
        label_hot_np[self.labels[idx]] = 1
        label_hot = torch.tensor(label_hot_np)
        print(label_hot)
        return idx, create_tensor(result), label_hot, label_hot


train_dataset = MyDataset(X_train_files, y_train)
validation_dataset = MyDataset(X_valid_files, y_valid)
test_dataset = MyDataset(X_valid_files, y_test)
# lol = DataLoader(train_dataset, batch_size = 4)
# for batch in lol:
#     print(len(batch))


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_scheduler(parameters, optimizer, train_loader):
    if parameters.scheduler == 'plateau':
        scheduler_main = ReduceLROnPlateau(
            optimizer,
            patience=parameters.scheduler_patience,
            verbose=True,
            factor=parameters.scheduler_factor,
            min_lr=0
        )
    elif parameters.scheduler == 'cosineannealing':
        cosine_max_epochs = parameters.cosannealing_epochs
        if parameters.minimum_lr is not None:
            cosine_min_lr = parameters.minimum_lr
        else:
            if parameters.lr <= 1e-5:
                cosine_min_lr = parameters.lr * 0.1
            else:
                cosine_min_lr = 1e-6
        scheduler_main = CosineAnnealingLR(
            optimizer,
            T_max=(cosine_max_epochs * len(train_loader)),
            eta_min=cosine_min_lr
        )
    elif parameters.scheduler == 'step':
        scheduler_main = StepLR(optimizer, step_size=parameters.step_size, gamma=0.1)
    elif parameters.scheduler == 'stepinf':
        scheduler_main = StepLR(optimizer, step_size=999, gamma=0.1)
    elif parameters.scheduler == 'singlestep':
        scheduler_main = MultiStepLR(
            optimizer,
            milestones=[np.random.randint(30,36)],
            gamma=0.1
        )
    elif parameters.scheduler == 'multistep':
        step_size_z = random.randint((-(parameters.step_size // 2)), (parameters.step_size // 2))
        scheduler_main = MultiStepLR(
            optimizer,
            milestones=[
                parameters.step_size,
                2*parameters.step_size+step_size_z,
                3*parameters.step_size+step_size_z,
            ],
            gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler {parameters.scheduler}")
    if not parameters.warmup:
        scheduler = scheduler_main
    else:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=parameters.stop_warmup_at_epoch,
            after_scheduler=scheduler_main
        )
    
    return scheduler, scheduler_main


def train(parameters: dict, callbacks: list = None):
    device = torch.device(parameters.device)
    print(device)
    
    # Reproducibility & benchmarking
    torch.backends.cudnn.benchmark = parameters.cudnn_benchmarking
    torch.backends.cudnn.deterministic = parameters.cudnn_deterministic
    torch.manual_seed(parameters.seed)
    if torch.cuda.is_available():
        print("!!!")
        torch.cuda.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)

    neptune_experiment = None
    
    # Load data for subgroup statistics
    
    # Prepare datasets
    # train_dataset = np.empty((100, 100, 100))
    # validation_dataset = np.empty((100, 100, 100))

    # DataLoaders
    train_sampler = None
    validation_sampler = None
    train_shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=parameters.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=parameters.num_workers,
        pin_memory=parameters.pin_memory,
        drop_last=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        sampler=validation_sampler,
        num_workers=parameters.num_workers,
        pin_memory=parameters.pin_memory,
        drop_last=True
    )
    # validation_labels = validation_dataset.get_labels()

    model = create_model()

    # Loss function and optimizer
    if parameters.architecture in ['3d_resnet18_fc', '2d_resnet50']:
        if parameters.label_type == 'cancer':
            loss_train = loss_eval = nn.BCEWithLogitsLoss()
        else:
            # for BI-RADS and BPE pretraining use softmax
            loss_train = loss_eval = nn.CrossEntropyLoss()
    else:
        loss_train = losses.BCELossWithSmoothing(smoothing=parameters.label_smoothing)
        loss_eval = losses.bce
    
    if parameters.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=parameters.lr,
            weight_decay=parameters.weight_decay
        )
    elif parameters.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), parameters.lr)
    elif parameters.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), parameters.lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer {parameters.optimizer}")

    # Scheduler
    scheduler, scheduler_main = get_scheduler(parameters, optimizer, train_loader)

    model.to(device)
    
    # Training/validation loop
    global_step = 0
    global_step_val = 0
    best_epoch = 0
    best_metric = 0
    resolve_callbacks('on_train_start', callbacks)
    try:
        for epoch_number in tqdm(range(1, (parameters.num_epochs + 1))):
            resolve_callbacks('on_epoch_start', callbacks, epoch_number)
            last_epoch = global_step // len(train_loader)

            logger.info(f'Starting *training* epoch number {epoch_number}')

            epoch_data = {
                "epoch_number": epoch_number
            }
            
            training_labels = {}

            # Training phase
            epoch_loss = []
            model.train()

            resolve_callbacks('on_train_start', callbacks)

            minibatch_number = 0
            number_of_used_training_examples = 0

            training_losses = []
            training_predictions = dict()

            for i_batch, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                resolve_callbacks('on_batch_start', callbacks)
                try:
                    indices, raw_data_batch, label_batch, mixed_label = batch
                    hmm = raw_data_batch.to(device)
                    output = model(hmm.float())

                    for ind_n, ind in enumerate(indices):
                        training_labels[ind] = label_batch[ind_n]
                    label_batch = label_batch.to(device)
                    print("!!!")
                    print(label_batch)
                    print(output)

                    minibatch_loss = 0

                    if len(label_batch) > 0:
                        number_of_used_training_examples = number_of_used_training_examples + len(label_batch)
                        subtraction = raw_data_batch  # (b_s, z, x, y)

                        for param in model.parameters():
                            param.grad = None
                        if parameters.mixup:
                            mixed_label = mixed_label.to(device)
                            mixup_loss1 = loss_train(output, mixed_label[:,0,:])
                            mixup_loss2 = loss_train(output, mixed_label[:,1,:])
                            minibatch_loss = (0.5 * mixup_loss1) + (0.5 * mixup_loss2)
                        else:
                            minibatch_loss = loss_train(output, label_batch)
                            print("Loss:", minibatch_loss)
                            # if parameters.architecture in ['3d_resnet18_fc', '2d_resnet50']:
                            #     if parameters.label_type == 'cancer':
                            #         minibatch_loss = loss_train(output, label_batch.type_as(output))
                            #     else:
                            #         minibatch_loss = loss_train(output, torch.max(label_batch, 1)[1])  # target converted from one-hot to (batch_size, C)
                            # elif parameters.architecture == '3d_gmic':
                            #     is_malignant = label_batch[0][1] or label_batch[0][3]
                            #     is_benign = label_batch[0][0] or label_batch[0][2]
                            #     target = torch.tensor([[is_malignant, is_benign]]).cuda()
                            #     minibatch_loss = loss_train(output, target)
                            # else:
                            #     # THIS IS THE DEFAULT LOSS
                            #     minibatch_loss = loss_train(output, label_batch)
                            #     print("Loss:", minibatch_loss)
                            logger.info(f"Minibatch loss: {minibatch_loss}")
                        epoch_loss.append(float(minibatch_loss))

                        # Backprop
                        if parameters.half:
                            with amp.scale_loss(minibatch_loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            minibatch_loss.backward()

                        # Optimizer
                        optimizer.step()

                        for i in range(0, len(label_batch)):
                            training_predictions[indices[i]] = output[i].cpu().detach().numpy()

                        minibatch_number += 1
                        global_step += 1

                        # Log learning rate
                        current_lr = optimizer.param_groups[0]['lr']

                        # Resolve schedulers at step
                        if type(scheduler) == CosineAnnealingLR:
                            scheduler.step()

                        # Warmup scheduler step update
                        if type(scheduler) == GradualWarmupScheduler:
                            if parameters.warmup and epoch_number < parameters.stop_warmup_at_epoch:
                                scheduler.step(epoch_number + ((global_step - last_epoch * len(train_loader)) / len(train_loader)))
                            else:
                                if type(scheduler_main) == CosineAnnealingLR:
                                    scheduler.step()

                    else:
                        logger.warn('No examples in this training minibatch were correctly loaded.')
                except Exception as e:
                    logger.error('[Error in train loop', traceback.format_exc())
                    logger.error(e)
                    continue

                resolve_callbacks('on_batch_end', callbacks)

            # Resolve schedulers at epoch
            if type(scheduler) == ReduceLROnPlateau:
                scheduler.step(np.mean(epoch_loss))
            elif type(scheduler) == GradualWarmupScheduler:
                if type(scheduler_main) != CosineAnnealingLR:
                    # Don't step for cosine; cosine is resolved at iter
                    scheduler.step(epoch=(epoch_number+1), metrics=np.mean(epoch_loss))
            elif type(scheduler) in [StepLR, MultiStepLR]:
                scheduler.step()

            # AUROC
            epoch_data['training_losses'] = training_losses
            epoch_data['training_predictions'] = training_predictions
            epoch_data['training_labels'] = training_labels
            resolve_callbacks('on_train_end', callbacks, epoch=epoch_number, logs=epoch_data, neptune_experiment=neptune_experiment)

            torch.cuda.empty_cache()
            
            resolve_callbacks('on_val_start', callbacks)
            model.eval()
            logger.info(f'Starting *validation* epoch number {epoch_number}')
            with torch.no_grad():
                minibatch_number = 0
                number_of_used_validation_examples = 0

                validation_losses = []
                validation_predictions = dict()
                
                for i_batch, batch in tqdm(enumerate(validation_loader), total=len(validation_loader)):
                    indices, raw_data_batch, label_batch, _ = batch
                    global_step_val += 1

                    label_batch = label_batch.to(device)

                    if len(label_batch) > 0:
                        number_of_used_validation_examples = number_of_used_validation_examples + len(label_batch)
                        
                        subtraction = raw_data_batch
                        
                        if parameters.architecture in ['r2plus1d_18', 'mc3_18'] and parameters.input_type != 'three_channels':
                            subtraction = subtraction.unsqueeze(1).contiguous()
                        
                        if parameters.input_type == 'random':
                            modality_losses = []
                            for x_modality in range(subtraction.shape[1]):
                                x = subtraction[:, x_modality, ...]
                                output = model(x.to(device))
                                modality_loss = loss_eval(output, label_batch)
                                modality_losses.append(modality_loss.item())
                            minibatch_loss = sum(modality_losses) / len(modality_losses)
                            validation_losses.append(minibatch_loss)
                        else:
                            if parameters.architecture == '3d_gmic':
                                output, _ = model(subtraction.to(device))
                            else:
                                # default
                                output = model(subtraction.to(device))
                            print(output)
                            if parameters.architecture in ['3d_resnet18_fc', '2d_resnet50']:
                                if parameters.label_type == 'cancer':
                                    minibatch_loss = loss_eval(output, label_batch.type_as(output))
                                else:
                                    minibatch_loss = loss_eval(output, torch.max(label_batch, 1)[1])  # target converted from one-hot to (batch_size, C)
                            elif parameters.architecture == '3d_gmic':
                                is_malignant = label_batch[0][1] or label_batch[0][3]
                                is_benign = label_batch[0][0] or label_batch[0][2]
                                target = torch.tensor([[is_malignant, is_benign]]).cuda()
                                minibatch_loss = loss_eval(output, target)
                            else:
                                # DEFAULT LOSS IN VAL
                                minibatch_loss = loss_eval(output, label_batch)                        
                            validation_losses.append(minibatch_loss.item())
                        logger.info(f"Minibatch loss: {minibatch_loss}")

                        for i in range(0, len(label_batch)):
                            validation_predictions[indices[i]] = output[i].cpu().numpy()

                    minibatch_number = minibatch_number + 1

                epoch_data['validation_losses'] = validation_losses
                epoch_data['validation_predictions'] = validation_predictions
                # epoch_data['validation_labels'] = validation_labels
                validation_losses = []

                torch.cuda.empty_cache()

            val_res = resolve_callbacks('on_val_end', callbacks, epoch=epoch_number, logs=epoch_data, neptune_experiment=neptune_experiment)
            
            # Checkpointing
            if not parameters.save_checkpoints:
                # Do not save checkpoints if distributed slave process or user specifies an arg
                pass
            else:
                if parameters.save_best_only:
                    if parameters.label_type == 'birads' or parameters.label_type == 'bpe':
                        birads_AUC = val_res['EvaluateEpoch']['auc']
                        if birads_AUC > best_metric:
                            best_metric = birads_AUC
                            best_epoch = epoch_number
                            model_file_name = os.path.join(parameters.model_saves_directory, f"model_best_auroc")
                            save_checkpoint(model, model_file_name, optimizer, is_amp=parameters.half, epoch=epoch_number)
                    else:
                        malignant_AUC = val_res['EvaluateEpoch']['auc_malignant']
                        if malignant_AUC > best_metric:
                            best_metric = malignant_AUC
                            best_epoch = epoch_number
                            model_file_name = os.path.join(parameters.model_saves_directory, f"model_best_auroc")
                            save_checkpoint(model, model_file_name, optimizer, is_amp=parameters.half, epoch=epoch_number)
                else:
                    model_file_name = os.path.join(parameters.model_saves_directory, f"model-epoch{epoch_number}")
                    save_checkpoint(model, model_file_name, optimizer, step=global_step, is_amp=parameters.half, epoch=epoch_number)

            resolve_callbacks('on_epoch_end', callbacks, epoch=epoch_number, logs=epoch_data)
    except KeyboardInterrupt:
        pass

    return


def get_args():
    parser = argparse.ArgumentParser("MRI Training pipeline")

    # File paths
    parser.add_argument("--metadata", type=str, default="/PATH/TO/PICKLE/FILE/WITH/METADATA.pkl", help="Pickled metadata file path")
    parser.add_argument("--datalist", type=str, default="/PATH/TO/PICKLE/FILE/WITH/DATALIST.pkl", help="Pickled data list file path")
    parser.add_argument("--subgroup_data", type=str, default='/PATH/TO/PICKLE/FILE/WITH/SUBGROUP/DATA.pkl', help='Pickled data with subgroup information')
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--weights_policy", type=str, help='Custom loaded weights surgery')

    # Input   
    parser.add_argument("--input_type", type=str, default='sub_t1c2', choices={'sub_t1c1', 'sub_t1c2', 't1c1', 't1c2', 't1pre', 'mip_t1c2', 'three_channel', 't2', 'random', 'multi', 'MIL'})
    parser.add_argument("--input_size", type=str, default='normal', choices={'normal', 'small'})
    # default was cancer for --label_type
    parser.add_argument("--label_type", type=str, default='birads', choices={'cancer', 'birads', 'bpe'}, help='What labels should be used, e.g. pretraining on BIRADS and second stage on cancer.')
    parser.add_argument("--subtraction_clipping", type=boolean_string, default=False, help='When performing subtraction, clip lower range values to 0')
    parser.add_argument("--preprocessing_policy", type=str, default='none', choices={'none', 'clahe'})
    parser.add_argument("--age_as_channel", type=boolean_string, default=False, help='Use age as additional channel')
    parser.add_argument("--isotropic", type=boolean_string, default=False, help='Use isotropic spacing (default is False-anisotropic)')

    # Model & augmentation
    parser.add_argument("--architecture", type=str, default="3d_resnet18", choices={'3d_resnet18', '3d_gmic', '3d_resnet18_fc', '3d_resnet34', '3d_resnet50', '3d_resnet101', 'r2plus1d_18', 'mc3_18', '2d_resnet50', 'multi_channel'})
    parser.add_argument("--resnet_width", type=float, default=1, help='Multiplier of ResNet width')
    parser.add_argument("--topk", type=int, default=10, help='Used only in our modified 3D resnet')
    parser.add_argument("--resnet_groups", type=int, default=16, help='Set 0 for batch norm; otherwise value will be number of groups in group norm')
    parser.add_argument("--aug_policy", type=str, default='affine', choices={'affine', 'none', 'strong_affine', 'rare_affine', 'weak_affine', 'policy1', '5deg_10scale', '10deg_5scale', '10deg_10scale', '10deg_10scale_p75', 'motion', 'ghosting', 'spike'})
    parser.add_argument("--affine_scale", type=float, default=0.10)
    parser.add_argument("--affine_rotation_deg", type=int, default=10)
    parser.add_argument("--affine_translation", type=int, default=0)
    parser.add_argument("--mixup", type=boolean_string, default=False, help='Use mixup augmentation')
    parser.add_argument("--loss", type=str, default="bce", choices={'bce'}, help='Which loss function to use')
    parser.add_argument("--network_modification", type=str, default=None, choices={'resnet18_bottleneck', 'resnet_36'})
    parser.add_argument("--cutout", type=boolean_string, default=False, help='Apply 3D cutout at training')
    parser.add_argument("--cutout_percentage", type=float, default=0.4)
    parser.add_argument("--label_smoothing", type=float, default=0.0, help='Label smoothing ratio')
    parser.add_argument("--dropout", type=boolean_string, default=False, 
    help='Adds Dropout layer before FC layer with p=0.25.')
    parser.add_argument("--stochastic_depth_rate", type=float, default=0.0,
    help='Uses stochastic depth in training')
    parser.add_argument("--use_se_layer", type=boolean_string, default=False,
    help='Use squeeze-and-excitation module')
    parser.add_argument("--inplanes", type=int, default=64)

    # Parallel computation
    parser.add_argument("--gpus", type=int, default=1, help='Needs to be specified when using DDP')
    parser.add_argument("--local_rank", type=int, default=-1, metavar='N', help='Local process rank')

    # Optimizers, schedulers
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=75)
    parser.add_argument("--optimizer", type=str, default='adam', choices={'adam', 'adamw', 'sgd'})
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=boolean_string, default=True)
    parser.add_argument("--stop_warmup_at_epoch", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default='cosineannealing', choices={'plateau', 'cosineannealing', 'step', 'stepinf', 'multistep', 'singlestep'})
    parser.add_argument("--step_size", type=int, default=15, help='If using StepLR, this is a step_size value')
    parser.add_argument("--scheduler_patience", type=int, default=7, help='Patience for ReduceLROnPlateau')
    parser.add_argument("--scheduler_factor", type=float, default=0.1, help='Rescaling factor for scheduler')
    parser.add_argument("--cosannealing_epochs", type=int, default=60, help='Length of a cosine annealing schedule')
    parser.add_argument("--minimum_lr", type=float, default=None, help='Minimum learning rate for the scheduler')

    # Efficiency 
    parser.add_argument("--num_workers", type=int, default=1) #19
    parser.add_argument("--pin_memory", type=boolean_string, default=True)
    parser.add_argument("--cudnn_benchmarking", type=boolean_string, default=True)
    parser.add_argument("--cudnn_deterministic", type=boolean_string, default=False)
    parser.add_argument("--half", type=boolean_string, default=False, help="Use half precision (fp16)") # true
    parser.add_argument("--half_level", type=str, default='O2', choices={'O1', 'O2'})
    parser.add_argument("--training_fraction", type=float, default=1.00)
    parser.add_argument("--number_of_training_samples", type=int, default=None, help='If this value is not None, it will overrule `training_fraction` parameter')
    parser.add_argument("--validation_fraction", type=float, default=1.00)
    parser.add_argument("--number_of_validation_samples", type=int, default=None, help='If this value is not None, it will overrule `validation_fraction` parameter')

    # Logging & debugging
    parser.add_argument("--logdir", type=str, default="/DIR/TO/LOGS/", help="Directory where logs are saved")
    parser.add_argument("--experiment", type=str, default="mri_training", help="Name of the experiment that will be used in logging")
    parser.add_argument("--log_every_n_steps", type=int, default=30)
    parser.add_argument("--save_checkpoints", type=boolean_string, default=True, help='Set to False if you dont want to save checkpoints')
    parser.add_argument("--save_best_only", type=boolean_string, default=False, help='Save checkpoints after every epoch if True; only after metric improvement if False')
    parser.add_argument("--seed", type=int, default=420)
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training fractions
    assert (0 < args.training_fraction <= 1.00), "training_fraction not in (0,1] range."
    assert (0 < args.validation_fraction <= 1.00), "validation_fraction not in (0,1] range."

    # Logging directories
    args.experiment_dirname = args.experiment + time.strftime('%Y%m%d%H%M%S')
    args.model_saves_directory = os.path.join(args.logdir, args.experiment_dirname)
    if os.path.exists(args.model_saves_directory):
        print("Warning: This model directory already exists")
    os.makedirs(args.model_saves_directory, exist_ok=True)

    # Save all arguments to the separate file
    parameters_path = os.path.join(args.model_saves_directory, "parameters.pkl")
    with open(parameters_path, "wb") as f:
        pickle.dump(vars(args), f)

    return args


def set_up_logger(args, log_args=True):
    log_file_name = 'output_log.log'
    log_file_path = os.path.join(args.model_saves_directory, log_file_name)
    
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logger.info(f"model_dir = {args.model_saves_directory}")
    if log_args:
        args_pprint = pprint.pformat(args)
        logger.info(f"parameters:\n{args_pprint}")

    return


if __name__ == "__main__":
    args = get_args()
    # set_up_logger(args)
    callbacks = [
        History(
            save_path=args.model_saves_directory,
            distributed=False,
            local_rank=args.local_rank
        ),
        RedundantCallback(),
        EvaluateEpoch(
            save_path=args.model_saves_directory,
            distributed=False,
            local_rank=args.local_rank,
            world_size=args.gpus,
            label_type=args.label_type
        )
    ]
    train(args, callbacks)
