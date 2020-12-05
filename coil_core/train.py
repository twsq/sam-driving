import os
import sys
import random
import time
import traceback
import torch
import torch.optim as optim

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss, adjust_learning_rate_auto
from input import CoILDataset, Augmenter, select_balancing_strategy
from logger import coil_logger
from coilutils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint, \
                                    check_loss_validation_stopped


# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, suppress_output=True, number_of_workers=12):
    """
        The main training function. This functions loads the latest checkpoint
        for a given, exp_batch (folder) and exp_alias (experiment configuration).
        With this checkpoint it starts from the beginning or continue some training.
    Args:
        gpu: The GPU number
        exp_batch: the folder with the experiments
        exp_alias: the alias, experiment name
        suppress_output: if the output are going to be saved on a file
        number_of_workers: the number of threads used for data loading

    Returns:
        None

    """
    try:
        # We set the visible cuda devices to select the GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        g_conf.VARIABLE_WEIGHT = {}
        # At this point the log file with the correct naming is created.
        # You merge the yaml file with the global configuration structure.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))
        set_type_of_process('train')
        # Set the process into loading status.
        coil_logger.add_message('Loading', {'GPU': gpu})
        
        # Seed RNGs
        torch.manual_seed(g_conf.MAGICAL_SEED)
        random.seed(g_conf.MAGICAL_SEED)

        # Put the output to a separate file if it is the case

        if suppress_output:
            if not os.path.exists('_output_logs'):
                os.mkdir('_output_logs')
            sys.stdout = open(os.path.join('_output_logs', exp_alias + '_' +
                              g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"), "a",
                              buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                              exp_alias + '_err_'+g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)

        if coil_logger.check_finish('train'):
            coil_logger.add_message('Finished', {})
            return

        # Preload option
        if g_conf.PRELOAD_MODEL_ALIAS is not None:
            checkpoint = torch.load(os.path.join('_logs', g_conf.PRELOAD_MODEL_BATCH,
                                                  g_conf.PRELOAD_MODEL_ALIAS,
                                                 'checkpoints',
                                                 str(g_conf.PRELOAD_MODEL_CHECKPOINT)+'.pth'))


        # Get the latest checkpoint to be loaded
        # returns none if there are no checkpoints saved for this model
        checkpoint_file = get_latest_saved_checkpoint()
        if checkpoint_file is not None:
            checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias,
                                    'checkpoints', str(get_latest_saved_checkpoint())))
            iteration = checkpoint['iteration']
            best_loss = checkpoint['best_loss']
            best_loss_iter = checkpoint['best_loss_iter']
        else:
            iteration = 0
            best_loss = 10000.0
            best_loss_iter = 0

        # Define the dataset.
        # Can specify a list of training datasets or just a single training dataset
        if len(g_conf.TRAIN_DATASET_NAMES) == 0:
            train_dataset_list = [g_conf.TRAIN_DATASET_NAME]
        else:
            train_dataset_list = g_conf.TRAIN_DATASET_NAMES
        full_dataset = [os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name) for dataset_name in train_dataset_list]

        # By instantiating the augmenter we get a callable that augment images and transform them
        # into tensors.
        augmenter = Augmenter(g_conf.AUGMENTATION)

        # Instantiate the class used to read a dataset. The coil dataset generator
        # can be found
        dataset = CoILDataset(full_dataset, transform=augmenter,
                                preload_names=[str(g_conf.NUMBER_OF_HOURS)
                                                    + 'hours_' + dataset_name for dataset_name in train_dataset_list], train_dataset=True)
        print ("Loaded dataset")

        # Create dataloader, model, and optimizer
        data_loader = select_balancing_strategy(dataset, iteration, number_of_workers)
        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=g_conf.LEARNING_RATE)

        # If we have a previous checkpoint, load model, optimizer, and record of previous 
        # train loss values (used for the learning rate schedule)
        if checkpoint_file is not None or g_conf.PRELOAD_MODEL_ALIAS is not None:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            accumulated_time = checkpoint['total_time']
            loss_window = coil_logger.recover_loss_window('train', iteration)
        else:  # We accumulate iteration time and keep the average speed
            accumulated_time = 0
            loss_window = []

        print ("Before the loss")
      
        # Define control loss function
        criterion = Loss(g_conf.LOSS_FUNCTION)
        
        if iteration == 0 and is_ready_to_save(iteration):

            state = {
                'iteration': iteration,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'total_time': accumulated_time,
                'optimizer': optimizer.state_dict(),
                'best_loss_iter': best_loss_iter
            }
            torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                           , 'checkpoints', str(iteration) + '.pth'))
        # Training loop
        for data in data_loader:

            # Basically in this mode of execution, we validate every X Steps, if it goes up 3 times,
            # add a stop on the _logs folder that is going to be read by this process
            if g_conf.FINISH_ON_VALIDATION_STALE is not None and \
                    check_loss_validation_stopped(iteration, g_conf.FINISH_ON_VALIDATION_STALE):
                break
            """
                ####################################
                    Main optimization loop
                ####################################
            """

            iteration += 1
            
            # Adjust learning rate based on training loss
            if iteration % 1000 == 0:
                adjust_learning_rate_auto(optimizer, loss_window)

            capture_time = time.time()
            model.zero_grad()
            
            controls = data['directions']
                
            # Run model forward and get outputs
            # First case corresponds to training squeeze network, second case corresponds to training driving model without 
            # mimicking losses, last case corresponds to training mimic network
            if "seg" in g_conf.SENSORS.keys():
                branches = model(data, dataset.extract_inputs(data).cuda(), dataset.extract_intentions(data).cuda())
            elif not g_conf.USE_REPRESENTATION_LOSS:
                branches = model(data, dataset.extract_inputs(data).cuda())
            else:
                branches, intermediate_reps = model(data, dataset.extract_inputs(data).cuda())

            # Compute control loss
            targets_to_use = dataset.extract_targets(data)
            loss_function_params = {
                'branches': branches,
                'targets': targets_to_use.cuda(),
                'controls': controls.cuda(),
                'inputs': dataset.extract_inputs(data).cuda(),
                'branch_weights': g_conf.BRANCH_LOSS_WEIGHT,
                'variable_weights': g_conf.VARIABLE_WEIGHT
            }
            loss, _ = criterion(loss_function_params)
            
            # Compute mimicking loss
            if g_conf.USE_REPRESENTATION_LOSS:
                expert_reps = dataset.extract_representations(data)
                # Seg mask mimicking loss
                if g_conf.USE_PERCEPTION_REP_LOSS:
                    perception_rep_loss_elementwise = (intermediate_reps[0] - expert_reps[0].cuda()) ** 2
                    perception_rep_loss = g_conf.PERCEPTION_REP_WEIGHT * torch.sum(perception_rep_loss_elementwise) / branches[0].shape[0]
                else:
                    perception_rep_loss = torch.tensor(0.).cuda()
                # Speed mimicking loss
                if g_conf.USE_SPEED_REP_LOSS:
                    speed_rep_loss_elementwise = (intermediate_reps[1] - expert_reps[1].cuda()) ** 2
                    speed_rep_loss = g_conf.SPEED_REP_WEIGHT * torch.sum(speed_rep_loss_elementwise) / branches[0].shape[0]
                else:
                    speed_rep_loss = torch.tensor(0.).cuda()
                # Stop intentions mimicking loss
                if g_conf.USE_INTENTION_REP_LOSS:
                    intentions_rep_loss_elementwise = (intermediate_reps[2] - expert_reps[2].cuda()) ** 2
                    intentions_rep_loss = g_conf.INTENTIONS_REP_WEIGHT * torch.sum(intentions_rep_loss_elementwise) / branches[0].shape[0]
                else:
                    intentions_rep_loss = torch.tensor(0.).cuda()
                rep_loss = g_conf.REP_LOSS_WEIGHT * (perception_rep_loss + speed_rep_loss + intentions_rep_loss)
                overall_loss = loss + rep_loss
            else:
                overall_loss = loss
            overall_loss.backward()
            optimizer.step()
            """
                ####################################
                    Saving the model if necessary
                ####################################
            """

            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'total_time': accumulated_time,
                    'optimizer': optimizer.state_dict(),
                    'best_loss_iter': best_loss_iter
                }
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                               , 'checkpoints', str(iteration) + '.pth'))

            """
                ################################################
                    Adding tensorboard logs.
                    Making calculations for logging purposes.
                    These logs are monitored by the printer module.
                #################################################
            """
            coil_logger.add_scalar('Loss', loss.data, iteration)
            if g_conf.USE_REPRESENTATION_LOSS:
                coil_logger.add_scalar('Perception Rep Loss', perception_rep_loss.data, iteration)
                coil_logger.add_scalar('Speed Rep Loss', speed_rep_loss.data, iteration)
                coil_logger.add_scalar('Intentions Rep Loss', intentions_rep_loss.data, iteration)
                coil_logger.add_scalar('Overall Rep Loss', rep_loss.data, iteration)
                coil_logger.add_scalar('Total Loss', overall_loss.data, iteration)
            if 'rgb' in data:
                coil_logger.add_image('Image', torch.squeeze(data['rgb']), iteration)
            if overall_loss.data < best_loss:
                best_loss = overall_loss.data.tolist()
                best_loss_iter = iteration

            # Log a random position
            position = random.randint(0, len(data) - 1)

            output = model.extract_branch(torch.stack(branches[0:4]), controls)
            error = torch.abs(output - targets_to_use.cuda())

            accumulated_time += time.time() - capture_time

            # Log to terminal and log file
            if g_conf.USE_REPRESENTATION_LOSS:
                coil_logger.add_message('Iterating',
                                        {'Iteration': iteration,
                                         'Loss': overall_loss.data.tolist(),
                                         'Control Loss': loss.data.tolist(), 
                                         'Rep Loss': rep_loss.data.tolist(), 
                                         'Images/s': (iteration * g_conf.BATCH_SIZE) / accumulated_time,
                                         'BestLoss': best_loss, 'BestLossIteration': best_loss_iter,
                                         'Output': output[position].data.tolist(),
                                         'GroundTruth': targets_to_use[
                                             position].data.tolist(),
                                         'Error': error[position].data.tolist(),
                                         'Inputs': dataset.extract_inputs(data)[
                                             position].data.tolist()},
                                        iteration)
            else:
                coil_logger.add_message('Iterating',
                                        {'Iteration': iteration,
                                         'Loss': loss.data.tolist(), 
                                         'Images/s': (iteration * g_conf.BATCH_SIZE) / accumulated_time,
                                         'BestLoss': best_loss, 'BestLossIteration': best_loss_iter,
                                         'Output': output[position].data.tolist(),
                                         'GroundTruth': targets_to_use[
                                             position].data.tolist(),
                                         'Error': error[position].data.tolist(),
                                         'Inputs': dataset.extract_inputs(data)[
                                             position].data.tolist()},
                                        iteration)
            # Save training loss history (useful for restoring training runs since learning rate is adjusted 
            # based on training loss)
            loss_window.append(overall_loss.data.tolist())
            coil_logger.write_on_error_csv('train', overall_loss.data)
            print("Iteration: %d  Loss: %f" % (iteration, overall_loss.data))

        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except RuntimeError as e:

        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
