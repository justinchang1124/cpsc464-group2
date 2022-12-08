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
from sklearn import model_selection

# project code
from architecture import create_model
import dcm_utils
import read_metadata
from datetime import datetime

abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = os.path.join(abs_proj_path, data_path)
dcm_files = dcm_utils.dcm_dir_list(abs_data_path)
n_dcm_files = len(dcm_files)

dcm_files_data = []
for file in dcm_files:
    dcm_files_data.append(file.split('\\'))

# open metadata
global_ids = []
for i in range(n_dcm_files):
    global_ids.append(dcm_files_data[i][0])
labels = [read_metadata.id_to_label(gid) for gid in global_ids]
groups = [read_metadata.id_to_group(gid) for gid in global_ids]

abs_dcm_files = dcm_utils.prepend_abs(abs_data_path, dcm_files)
n_classes = len(set(labels))

X_train_files = []
X_test_files = []
y_train = []
y_test = []
z_train = []
z_test = []

augment_groups = True
use_distance = False

iter_order = list(range(n_dcm_files))
random.shuffle(iter_order)
iter_order = iter_order[:1000]

group_total = read_metadata.empty_group_counter()
for i in iter_order:
    group_total[groups[i]] += 1
group_upper = max(group_total.values())

grpmap_train = read_metadata.empty_group_counter()
grpmap_test = read_metadata.empty_group_counter()

for i in iter_order:
    train_grp = grpmap_train[groups[i]]
    test_grp = grpmap_test[groups[i]]
    total_grp = group_total[groups[i]]
    aug_factor = 1
    if augment_groups:
        aug_factor = int(group_upper / total_grp)
    if train_grp <= total_grp * 0.8:
        for j in range(aug_factor):
            X_train_files.append(abs_dcm_files[i])
            y_train.append(labels[i])
            z_train.append(groups[i])
        grpmap_train[groups[i]] += 1
    elif test_grp <= total_grp * 0.2:
        for j in range(aug_factor):
            X_test_files.append(abs_dcm_files[i])
            y_test.append(labels[i])
            z_test.append(groups[i])
        grpmap_test[groups[i]] += 1

class MyDataset(Dataset):

    def __init__(self, t_abs_dcm_files, t_labels, t_groups):
        self.abs_dcm_files = t_abs_dcm_files
        self.labels = t_labels
        self.groups = t_groups
        super().__init__()

    def __len__(self):
        return len(self.abs_dcm_files)

    def __getitem__(self, idx):
        dcm_file = self.abs_dcm_files[idx]
        dcm_data = dcm_utils.open_dcm_with_image(dcm_file)
        img_clamp = dcm_utils.perc_clamp_dcm_image(dcm_data.pixel_array, 1, 99)
        img_norm = dcm_utils.normalize_dcm_image(img_clamp)
        img_tensor = dcm_utils.dcm_image_to_tensor4d(img_norm)
        img_aug = dcm_utils.augment_tensor4d(img_tensor)
        return idx, img_aug, dcm_utils.label_to_one_hot(self.labels[idx]), self.groups[idx]


train_dataset = MyDataset(X_train_files, y_train, z_train)
test_dataset = MyDataset(X_test_files, y_test, z_test)


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
                    indices, raw_data_batch, label_batch, groups = batch
                    output = model(raw_data_batch)

                    output_labels = dcm_utils.get_argmax_batch(output)
                    target_labels = dcm_utils.get_argmax_batch(label_batch)
                    diff_labels = dcm_utils.diff_lists(output_labels, target_labels)

                    correct_epoch += diff_labels.count(0)
                    total_epoch += len(output_labels)

                    print()
                    print("Correct this Epoch: {}".format(correct_epoch))
                    print("Total this Epoch: {}".format(total_epoch))
                    print("Accuracy Rate: {}".format(correct_epoch / total_epoch))

                    for ind_n, ind in enumerate(indices):
                        training_labels[ind] = label_batch[ind_n]
                    label_batch = label_batch.to(device)

                    if len(label_batch) > 0:
                        number_of_used_training_examples = number_of_used_training_examples + len(label_batch)

                        for param in model.parameters():
                            param.grad = None

                        diff_multiplier = 1
                        if use_distance:
                            diff_multiplier += sum(map(abs, diff_labels))
                        minibatch_loss = loss_train(output, label_batch) * diff_multiplier
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
            test_group = []
            for i_batch, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
                indices, raw_data_batch, label_batch, groups = batch
                test_group += groups
                output = model(raw_data_batch)

                test_preds += dcm_utils.get_argmax_batch(output)
                test_trues += dcm_utils.get_argmax_batch(label_batch)

            time_str = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
            dcm_utils.write_labels(test_preds, "{}/preds_{}_{}.txt".format(abs_proj_path, epoch_number, time_str))
            dcm_utils.write_labels(test_trues, "{}/trues_{}_{}.txt".format(abs_proj_path, epoch_number, time_str))
            dcm_utils.write_labels(z_test, "{}/ztest_{}_{}.txt".format(abs_proj_path, epoch_number, time_str))
            ea_dict, er_dict = dcm_utils.separate_by_group(test_preds, test_trues, test_group)
            dcm_utils.summarize_ar_dict(ea_dict, er_dict)

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
