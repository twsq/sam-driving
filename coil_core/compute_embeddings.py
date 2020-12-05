import os
import time
import sys
import random


import torch
import traceback
import dlib

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel
from input import CoILDataset, Augmenter
from logger import coil_logger
from coilutils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximun_checkpoint_reach, get_next_checkpoint, validation_stale_point
import numpy as np

# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, dataset_name, validation_set=False):
    latest = None
    # We set the visible cuda devices
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    g_conf.immutable(False)
    # At this point the log file with the correct naming is created.
    merge_with_yaml(os.path.join('configs', exp_batch, exp_alias+'.yaml'))
    # If using validation dataset, fix a very high number of hours
    if validation_set:
        g_conf.NUMBER_OF_HOURS = 10000
    g_conf.immutable(True)


    # Define the dataset.
    full_dataset = [os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name)]
    augmenter = Augmenter(None)
    if validation_set:
        # Definition of the dataset to be used. Preload name is just the validation data name
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_names=[dataset_name])
    else:
        dataset = CoILDataset(full_dataset, transform=augmenter, 
                              preload_names=[str(g_conf.NUMBER_OF_HOURS) + 'hours_' + dataset_name], train_dataset=True)

    # The data loader is the multi threaded module from pytorch that release a number of
    # workers to get all the data.
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=g_conf.NUMBER_OF_LOADING_WORKERS,
                                              pin_memory=True)

    # Define model
    model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
    
    """ 
        ######
        Run a single driving benchmark specified by the checkpoint were validation is stale
        ######
    """

    if g_conf.FINISH_ON_VALIDATION_STALE is not None:

        while validation_stale_point(g_conf.FINISH_ON_VALIDATION_STALE) is None:
            time.sleep(0.1)

        validation_state_iteration = validation_stale_point(g_conf.FINISH_ON_VALIDATION_STALE)
        
        checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias
                                         , 'checkpoints', str(validation_state_iteration) + '.pth'))
        print("Validation loaded ", validation_state_iteration)
    else:
        """
        #####
        Main Loop , Run a benchmark for each specified checkpoint on the "Test Configuration"
        #####
        """
        while not maximun_checkpoint_reach(latest, g_conf.TEST_SCHEDULE):
            # Get the correct checkpoint
            # We check it for some task name, all of then are ready at the same time
            if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE,
                                        control_filename + '_' + task_list[0]):

                latest = get_next_checkpoint(g_conf.TEST_SCHEDULE,
                                             control_filename + '_' + task_list[0])
                                             
                checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias
                                         , 'checkpoints', str(latest) + '.pth'))
                print("Validation loaded ", latest)
            else:
                time.sleep(0.1)
    
    # Load the model and prepare set it for evaluation
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    first_iter = True
    for data in data_loader:

        # Compute the forward pass on a batch from the dataset and get the intermediate 
        # representations of the squeeze network
        if "seg" in g_conf.SENSORS.keys():
            perception_rep, speed_rep, intentions_rep = \
                model.get_intermediate_representations(data,
                                                       dataset.extract_inputs(data).cuda(),
                                                       dataset.extract_intentions(data).cuda())
            perception_rep = perception_rep.data.cpu()
            speed_rep = speed_rep.data.cpu()
            intentions_rep = intentions_rep.data.cpu()
        if first_iter:
            perception_rep_all = perception_rep
            speed_rep_all = speed_rep
            intentions_rep_all = intentions_rep
        else:
            perception_rep_all = torch.cat([perception_rep_all, perception_rep], 0)
            speed_rep_all = torch.cat([speed_rep_all, speed_rep], 0)
            intentions_rep_all = torch.cat([intentions_rep_all, intentions_rep], 0)
        first_iter = False

    # Save intermediate representations
    perception_rep_all = perception_rep_all.tolist()
    speed_rep_all = speed_rep_all.tolist()
    intentions_rep_all = intentions_rep_all.tolist()
    np.save(os.path.join('_preloads', exp_batch + '_' + exp_alias + '_' + dataset_name + '_representations'), [perception_rep_all, speed_rep_all, intentions_rep_all])
    
