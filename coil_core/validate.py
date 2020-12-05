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
    maximun_checkpoint_reach, get_next_checkpoint


def write_regular_output(iteration, output):
    for i in range(len(output)):
        coil_logger.write_on_csv(iteration, [output[i][0],
                                            output[i][1],
                                            output[i][2]])


# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, dataset_name, suppress_output):
    latest = None
    try:
        # We set the visible cuda devices
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        # At this point the log file with the correct naming is created.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias+'.yaml'))
        # The validation dataset is always fully loaded, so we fix a very high number of hours
        g_conf.NUMBER_OF_HOURS = 10000
        set_type_of_process('validation', dataset_name)

        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        if suppress_output:
            sys.stdout = open(os.path.join('_output_logs',
                                           exp_alias + '_' + g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                              exp_alias + '_err_' + g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)


        # Define the dataset.
        full_dataset = [os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name)]
        augmenter = Augmenter(None)
        # Definition of the dataset to be used. Preload name is just the validation data name
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_names=[dataset_name])

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=g_conf.NUMBER_OF_LOADING_WORKERS,
                                                  pin_memory=True)

        # Create model.
        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        # The window used to keep track of the validation loss
        l1_window = []
        # If we have evaluated a checkpoint, get the validation losses of all the previously 
        # evaluated checkpoints (validation loss is used for early stopping)
        latest = get_latest_evaluated_checkpoint()
        if latest is not None:  # When latest is noe
            l1_window = coil_logger.recover_loss_window(dataset_name, None)

        model.cuda()

        best_mse = 1000
        best_error = 1000
        best_mse_iter = 0
        best_error_iter = 0

        # Loop to validate all checkpoints as they are saved during training
        while not maximun_checkpoint_reach(latest, g_conf.TEST_SCHEDULE):
            if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE):
                with torch.no_grad():
                    # Get and load latest checkpoint
                    latest = get_next_checkpoint(g_conf.TEST_SCHEDULE)
    
                    checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias
                                            , 'checkpoints', str(latest) + '.pth'))
                    checkpoint_iteration = checkpoint['iteration']
                    print("Validation loaded ", checkpoint_iteration)
    
                    model.load_state_dict(checkpoint['state_dict'])
                    model.eval()
                    
                    accumulated_mse = 0
                    accumulated_error = 0
                    iteration_on_checkpoint = 0
                    if g_conf.USE_REPRESENTATION_LOSS:
                        accumulated_perception_rep_mse = 0
                        accumulated_speed_rep_mse = 0
                        accumulated_intentions_rep_mse = 0
                        accumulated_rep_mse = 0
                        accumulated_perception_rep_error = 0
                        accumulated_speed_rep_error = 0
                        accumulated_intentions_rep_error = 0
                        accumulated_rep_error = 0
                    
                    # Validation loop
                    for data in data_loader:
    
                        # Compute the forward pass on a batch from  the validation dataset
                        controls = data['directions']
                            
                        # Run model forward and get outputs
                        # First case corresponds to squeeze network, second case corresponds to driving model without 
                        # mimicking losses, last case corresponds to mimic network
                        if "seg" in g_conf.SENSORS.keys():
                            output = model.forward_branch(data,
                                                          dataset.extract_inputs(data).cuda(),
                                                          controls, 
                                                          dataset.extract_intentions(data).cuda())
                        elif not g_conf.USE_REPRESENTATION_LOSS:
                            output = model.forward_branch(data,
                                                          dataset.extract_inputs(data).cuda(),
                                                          controls)
                        else:
                            output, intermediate_reps = model.forward_branch(data,
                                                                             dataset.extract_inputs(data).cuda(),
                                                                             controls)

                        write_regular_output(checkpoint_iteration, output)
                            
                        # Compute control loss on current validation batch and accumulate it
                        targets_to_use = dataset.extract_targets(data)

                        mse = torch.mean((output -
                                            targets_to_use.cuda())**2).data.tolist()
                        mean_error = torch.mean(
                                        torch.abs(output -
                                                    targets_to_use.cuda())).data.tolist()
    
                        accumulated_error += mean_error
                        accumulated_mse += mse

                        error = torch.abs(output - targets_to_use.cuda())
                        
                        # Compute mimicking losses on current validation batch and accumulate it
                        if g_conf.USE_REPRESENTATION_LOSS:
                            expert_reps = dataset.extract_representations(data)
                            # First L1 losses (seg mask, speed, intention mimicking losses)
                            if g_conf.USE_PERCEPTION_REP_LOSS:
                                perception_rep_loss = torch.sum(torch.abs(intermediate_reps[0] - expert_reps[0].cuda())).data.tolist() / (3 * output.shape[0])
                            else:
                                perception_rep_loss = 0
                            if g_conf.USE_SPEED_REP_LOSS:
                                speed_rep_loss = torch.sum(torch.abs(intermediate_reps[1] - expert_reps[1].cuda())).data.tolist() / (3 * output.shape[0])
                            else:
                                speed_rep_loss = 0
                            if g_conf.USE_INTENTION_REP_LOSS:
                                intentions_rep_loss = torch.sum(torch.abs(intermediate_reps[2] - expert_reps[2].cuda())).data.tolist() / (3 * output.shape[0])
                            else:
                                intentions_rep_loss = 0
                            rep_error = g_conf.REP_LOSS_WEIGHT * (perception_rep_loss + speed_rep_loss + intentions_rep_loss)
                            accumulated_perception_rep_error += perception_rep_loss
                            accumulated_speed_rep_error += speed_rep_loss
                            accumulated_intentions_rep_error += intentions_rep_loss
                            accumulated_rep_error += rep_error
                            
                            # L2 losses now
                            if g_conf.USE_PERCEPTION_REP_LOSS:
                                perception_rep_loss = torch.sum((intermediate_reps[0] - expert_reps[0].cuda()) ** 2).data.tolist() / (3 * output.shape[0])
                            else:
                                perception_rep_loss = 0
                            if g_conf.USE_SPEED_REP_LOSS:
                                speed_rep_loss = torch.sum((intermediate_reps[1] - expert_reps[1].cuda()) ** 2).data.tolist() / (3 * output.shape[0])
                            else:
                                speed_rep_loss = 0
                            if g_conf.USE_INTENTION_REP_LOSS:
                                intentions_rep_loss = torch.sum((intermediate_reps[2] - expert_reps[2].cuda()) ** 2).data.tolist() / (3 * output.shape[0])
                            else:
                                intentions_rep_loss = 0
                            rep_mse = g_conf.REP_LOSS_WEIGHT * (perception_rep_loss + speed_rep_loss + intentions_rep_loss)
                            accumulated_perception_rep_mse += perception_rep_loss
                            accumulated_speed_rep_mse += speed_rep_loss
                            accumulated_intentions_rep_mse += intentions_rep_loss
                            accumulated_rep_mse += rep_mse
    
                        # Log a random position
                        position = random.randint(0, len(output.data.tolist())-1)
                        
                        # Logging
                        if g_conf.USE_REPRESENTATION_LOSS:
                            total_mse = mse + rep_mse
                            total_error = mean_error + rep_error
                            coil_logger.add_message('Iterating',
                                 {'Checkpoint': latest,
                                  'Iteration': (str(iteration_on_checkpoint*120)+'/'+str(len(dataset))),
                                  'MeanError': mean_error,
                                  'MSE': mse,
                                  'RepMeanError': rep_error, 
                                  'RepMSE': rep_mse,
                                  'MeanTotalError': total_error,
                                  'TotalMSE': total_mse, 
                                  'Output': output[position].data.tolist(),
                                  'GroundTruth': targets_to_use[position].data.tolist(),
                                  'Error': error[position].data.tolist(),
                                  'Inputs': dataset.extract_inputs(data)[position].data.tolist()},
                                  latest)
                        else:
                            coil_logger.add_message('Iterating',
                                 {'Checkpoint': latest,
                                  'Iteration': (str(iteration_on_checkpoint*120)+'/'+str(len(dataset))),
                                  'MeanError': mean_error,
                                  'MSE': mse,
                                  'Output': output[position].data.tolist(),
                                  'GroundTruth': targets_to_use[position].data.tolist(),
                                  'Error': error[position].data.tolist(),
                                  'Inputs': dataset.extract_inputs(data)[position].data.tolist()},
                                  latest)
                        iteration_on_checkpoint += 1
                        
                        if g_conf.USE_REPRESENTATION_LOSS:
                                print("Iteration %d  on Checkpoint %d : Error %f" % (iteration_on_checkpoint,
                                                                            checkpoint_iteration, total_error))
                        else:
                            print("Iteration %d  on Checkpoint %d : Error %f" % (iteration_on_checkpoint,
                                                                        checkpoint_iteration, mean_error))
    
                    """
                        ########
                        Finish a round of validation, write results, wait for the next
                        ########
                    """
                    # Compute average L1 and L2 losses over whole round of validation and log them
                    checkpoint_average_mse = accumulated_mse/(len(data_loader))
                    checkpoint_average_error = accumulated_error/(len(data_loader))
                    coil_logger.add_scalar('L2 Loss', checkpoint_average_mse, latest, True)
                    coil_logger.add_scalar('Loss', checkpoint_average_error, latest, True)
                    
                    if g_conf.USE_REPRESENTATION_LOSS:
                        checkpoint_average_perception_rep_mse = accumulated_perception_rep_mse/(len(data_loader))
                        checkpoint_average_speed_rep_mse = accumulated_speed_rep_mse/(len(data_loader))
                        checkpoint_average_intentions_rep_mse = accumulated_intentions_rep_mse/(len(data_loader))
                        checkpoint_average_rep_mse = accumulated_rep_mse/(len(data_loader))
                        checkpoint_average_total_mse = checkpoint_average_mse + checkpoint_average_rep_mse

                        checkpoint_average_perception_rep_error = accumulated_perception_rep_error/(len(data_loader))
                        checkpoint_average_speed_rep_error = accumulated_speed_rep_error/(len(data_loader))
                        checkpoint_average_intentions_rep_error = accumulated_intentions_rep_error/(len(data_loader))
                        checkpoint_average_rep_error = accumulated_rep_error/(len(data_loader))
                        checkpoint_average_total_error = checkpoint_average_error + checkpoint_average_rep_mse
                        
                        # Log L1/L2 loss terms
                        coil_logger.add_scalar('Perception Rep Loss', checkpoint_average_perception_rep_mse, latest, True)
                        coil_logger.add_scalar('Speed Rep Loss', checkpoint_average_speed_rep_mse, latest, True)
                        coil_logger.add_scalar('Intentions Rep Loss', checkpoint_average_intentions_rep_mse, latest, True)
                        coil_logger.add_scalar('Overall Rep Loss', checkpoint_average_rep_mse, latest, True)
                        coil_logger.add_scalar('Total L2 Loss', checkpoint_average_total_mse, latest, True)

                        coil_logger.add_scalar('Perception Rep Error', checkpoint_average_perception_rep_error, latest, True)
                        coil_logger.add_scalar('Speed Rep Error', checkpoint_average_speed_rep_error, latest, True)
                        coil_logger.add_scalar('Intentions Rep Error', checkpoint_average_intentions_rep_error, latest, True)
                        coil_logger.add_scalar('Total Rep Error', checkpoint_average_rep_error, latest, True)
                        coil_logger.add_scalar('Total Loss', checkpoint_average_total_error, latest, True)
                    else:
                        checkpoint_average_total_mse = checkpoint_average_mse
                        checkpoint_average_total_error = checkpoint_average_error
                    
                    if checkpoint_average_total_mse < best_mse:
                        best_mse = checkpoint_average_total_mse
                        best_mse_iter = latest
    
                    if checkpoint_average_total_error < best_error:
                        best_error = checkpoint_average_total_error
                        best_error_iter = latest
    
                    # Print for logging / to terminal validation results
                    if g_conf.USE_REPRESENTATION_LOSS:
                        coil_logger.add_message('Iterating',
                             {'Summary':
                                 {
                                  'Control Error': checkpoint_average_error,
                                  'Control Loss': checkpoint_average_mse,
                                  'Rep Error': checkpoint_average_rep_error, 
                                  'Rep Loss': checkpoint_average_rep_mse, 
                                  'Error': checkpoint_average_total_error, 
                                  'Loss': checkpoint_average_total_mse, 
                                  'BestError': best_error,
                                  'BestMSE': best_mse,
                                  'BestMSECheckpoint': best_mse_iter,
                                  'BestErrorCheckpoint': best_error_iter
                                 },
        
                              'Checkpoint': latest},
                                                latest)
                    else:
                        coil_logger.add_message('Iterating',
                             {'Summary':
                                 {
                                  'Error': checkpoint_average_error,
                                  'Loss': checkpoint_average_mse,
                                  'BestError': best_error,
                                  'BestMSE': best_mse,
                                  'BestMSECheckpoint': best_mse_iter,
                                  'BestErrorCheckpoint': best_error_iter
                                 },
        
                              'Checkpoint': latest},
                                                latest)
                    
                    # Save validation loss history (validation loss is used for early stopping)
                    l1_window.append(checkpoint_average_total_error)
                    coil_logger.write_on_error_csv(dataset_name, checkpoint_average_total_error)
    
                    # Early stopping
                    if g_conf.FINISH_ON_VALIDATION_STALE is not None:
                        if dlib.count_steps_without_decrease(l1_window) > 3 and \
                                dlib.count_steps_without_decrease_robust(l1_window) > 3:
                            coil_logger.write_stop(dataset_name, latest)
                            break

            else:

                latest = get_latest_evaluated_checkpoint()
                time.sleep(1)

                coil_logger.add_message('Loading', {'Message': 'Waiting Checkpoint'})
                print("Waiting for the next Validation")

        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)

    except RuntimeError as e:
        if latest is not None:
            coil_logger.erase_csv(latest)
        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)
