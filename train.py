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

from utilities.callbacks import History, RedundantCallback, resolve_callbacks, EvaluateEpoch
from utilities.warmup import GradualWarmupScheduler
from utilities.utils import boolean_string, save_checkpoint
from networks import losses
from architecture_investigation import *
from labels_presentation import *


# loss = nn.CrossEntropyLoss()
# predict = torch.tensor([[100, 100, 100, 100, 1000]]).float()
# target = torch.tensor([[0, 0, 0, 0, 1]]).float()
# x = loss(predict, target)

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
train_ids, test_ids = model_selection.train_test_split(unique_ids, test_size=0.2)

X_train_files = []
X_test_files = []
y_train = []
y_test = []
z_train = []
z_test = []

for i in range(tn_dcm_files):
    gid = global_ids[i]
    if gid in train_ids:
        X_train_files.append(abs_dcm_files[i])
        y_train.append(labels[i])
        z_train.append(groups[i])
    if gid in test_ids:
        X_test_files.append(abs_dcm_files[i])
        y_test.append(labels[i])
        z_test.append(groups[i])


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
        result = np.empty((1, 1, 128, 128))  # num_channels, z, x, y
        result[0, 0] = x_image
        label_hot_np = [0, 0, 0, 0, 0]
        label_hot_np[self.labels[idx]] = 1
        label_hot = torch.tensor(label_hot_np).float()
        return idx, create_tensor(result), label_hot, label_hot


train_dataset = MyDataset(X_train_files, y_train)
test_dataset = MyDataset(X_test_files, y_test)


def get_argmax_2x5(output):
    result = []
    for j in range(2):
        max_output_index = 0
        max_output_val = output[j][0]
        for k in range(1, 5):
            if output[j][k] > max_output_val:
                max_output_index = k
                max_output_val = output[j][k]
        result.append(max_output_index)
    return result


def get_correct_2x5(output, target):
    correct = 0
    for j in range(2):
        max_output_index = 0
        max_output_val = output[j][0]
        max_target_index = 0
        max_target_val = target[j][0]
        for k in range(1, 5):
            if output[j][k] > max_output_val:
                max_output_index = k
                max_output_val = output[j][k]
            if target[j][k] > max_target_val:
                max_target_index = k
                max_target_val = target[j][k]
        if max_target_index == max_output_index:
            correct += 1
    return correct


examp1 = torch.tensor([[-3.6933, -0.9389,  0.3308, -2.0394, -2.4012],
        [-3.4493, -0.8140, -0.5046, -1.4021, -3.3808]])

examp2 = torch.tensor([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0]])

get_correct_2x5(examp1, examp2)






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
    
    # Reproducibility & benchmarking
    torch.backends.cudnn.benchmark = parameters.cudnn_benchmarking
    torch.backends.cudnn.deterministic = parameters.cudnn_deterministic
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)

    neptune_experiment = None
    
    # Load data for subgroup statistics

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=parameters.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=parameters.batch_size,
        shuffle=True
    )

    model = create_model()
    loss_train = loss_eval = nn.CrossEntropyLoss()
    
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
    
    # Training loop
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

            total_epoch = 0
            correct_epoch = 0
            for i_batch, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                resolve_callbacks('on_batch_start', callbacks)
                try:
                    indices, raw_data_batch, label_batch, mixed_label = batch
                    hmm = raw_data_batch.to(device)
                    output = model(hmm.float())

                    total_epoch += 2
                    correct_epoch += get_correct_2x5(output, label_batch)
                    print()
                    print("Correct this Epoch: {}".format(correct_epoch))
                    print("Total this Epoch: {}".format(total_epoch))
                    print("Accuracy Rate: {}".format(correct_epoch / total_epoch))

                    for ind_n, ind in enumerate(indices):
                        training_labels[ind] = label_batch[ind_n]
                    label_batch = label_batch.to(device)

                    minibatch_loss = 0

                    if len(label_batch) > 0:
                        number_of_used_training_examples = number_of_used_training_examples + len(label_batch)
                        subtraction = raw_data_batch  # (b_s, z, x, y)

                        for param in model.parameters():
                            param.grad = None

                        minibatch_loss = loss_train(output, label_batch)
                        print("Loss:", minibatch_loss)

                        # Backprop
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

            print("TESTING -----------")
            test_preds = []
            test_trues = []
            test_correct = 0
            test_total = 0
            for i_batch, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                indices, raw_data_batch, label_batch, mixed_label = batch
                hmm = raw_data_batch.to(device)
                output = model(hmm.float())
                test_correct += get_correct_2x5(output, label_batch)
                test_total += 2
                test_preds += get_argmax_2x5(output)
                test_trues += get_argmax_2x5(label_batch)
                print()
                print("Correct this Test: {}".format(test_correct))
                print("Total this Test: {}".format(test_total))
                print("Accuracy Rate: {}".format(correct_epoch / total_epoch))
            ea_dict, er_dict = separate_by_group(test_preds, test_trues, z_test)
            print(ea_dict, er_dict)
            summarize_ar_dict(ea_dict, er_dict)

            # Resolve schedulers at epoch
            if type(scheduler) == ReduceLROnPlateau:
                scheduler.step(np.mean(epoch_loss))
            elif type(scheduler) == GradualWarmupScheduler:
                if type(scheduler_main) != CosineAnnealingLR:
                    # Don't step for cosine; cosine is resolved at iter
                    scheduler.step(epoch=(epoch_number+1), metrics=np.mean(epoch_loss))
            elif type(scheduler) in [StepLR, MultiStepLR]:
                scheduler.step()

            torch.cuda.empty_cache()
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
