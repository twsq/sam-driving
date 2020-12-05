from logger import coil_logger
import torch.nn as nn
import torch
import importlib

from configs import g_conf
from coilutils.general import command_number_to_index

from .building_blocks import Conv
from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join

class CoILICRA(nn.Module):
    def __init__(self, params):
        # TODO: Improve the model autonaming function

        super(CoILICRA, self).__init__()
        self.params = params

        number_first_layer_channels = 0
        for _, sizes in g_conf.SENSORS.items():
            number_first_layer_channels += sizes[0] * g_conf.NUMBER_FRAMES_FUSION

        # Get one item from the dict
        sensor_input_shape = next(iter(g_conf.SENSORS.values()))
        sensor_input_shape = [number_first_layer_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]

        # For this case we check if the perception layer is of the type "conv"
        if 'conv' in params['perception']:
            perception_convs = Conv(params={'channels': [number_first_layer_channels] +
                                                          params['perception']['conv']['channels'],
                                            'kernels': params['perception']['conv']['kernels'],
                                            'strides': params['perception']['conv']['strides'],
                                            'dropouts': params['perception']['conv']['dropouts'],
                                            'end_layer': True})

            perception_fc = FC(params={'neurons': [perception_convs.get_conv_output(sensor_input_shape)]
                                                  + params['perception']['fc']['neurons'],
                                       'dropouts': params['perception']['fc']['dropouts'],
                                       'end_layer': False})

            self.perception = nn.Sequential(*[perception_convs, perception_fc])

            number_output_neurons = params['perception']['fc']['neurons'][-1]

        elif 'res' in params['perception']:  # pre defined residual networks
            resnet_module = importlib.import_module('network.models.building_blocks.resnet')
            resnet_module = getattr(resnet_module, params['perception']['res']['name'])

            self.perception = resnet_module(pretrained=g_conf.PRE_TRAINED,
                                             num_classes=params['perception']['res']['num_classes'], 
                                             input_channels=number_first_layer_channels)
            number_output_neurons = params['perception']['res']['num_classes']
                
            if g_conf.SEPARATE_PERCEPTION_INTENTION_REPS:
                self.intention_rep = resnet_module(pretrained=g_conf.PRE_TRAINED,
                                                    num_classes=params['intention_rep']['res']['num_classes'], 
                                                    input_channels=number_first_layer_channels)
                number_output_neurons += params['intention_rep']['res']['num_classes']                  

        else:

            raise ValueError("invalid convolution layer type")
        
        self.measurements = FC(params={'neurons': [len(g_conf.INPUTS)] +
                                                   params['measurements']['fc']['neurons'],
                                       'dropouts': params['measurements']['fc']['dropouts'],
                                       'end_layer': False})
        
        if "seg" in g_conf.SENSORS.keys():
            intentions_size = len(g_conf.INTENTIONS)
            self.intention_branch = FC(params={'neurons': [intentions_size] +
                                                 params['intentions']['fc']['neurons'],
                                               'dropouts': params['intentions']['fc']['dropouts'],
                                               'end_layer': False})
            join_neurons = [params['measurements']['fc']['neurons'][-1] + 
                            params['intentions']['fc']['neurons'][-1] + 
                            number_output_neurons] + params['join']['fc']['neurons']
        else:
            join_neurons = [params['measurements']['fc']['neurons'][-1] + 
                            number_output_neurons] + params['join']['fc']['neurons']
        
        self.join = Join(
            params={'after_process':
                            FC(params={'neurons': join_neurons,
                                    'dropouts': params['join']['fc']['dropouts'],
                                    'end_layer': False}),
                        'mode': 'cat'
                    }
            )
        
        if "seg" in g_conf.SENSORS.keys():
            self.join2 = Join(
                params={'after_process':
                                FC(params={'neurons': [number_output_neurons + 
                                                       params['intentions']['fc']['neurons'][-1]] + 
                                                       params['join2']['fc']['neurons'],
                                           'dropouts': params['join2']['fc']['dropouts'],
                                           'end_layer': False}),
                         'mode': 'cat'
                        }
                )
        elif (g_conf.USE_REPRESENTATION_LOSS and g_conf.USE_INTENTION_REP_LOSS) or g_conf.SEPARATE_PERCEPTION_INTENTION_REPS:
            self.join2 = Join(
                params={'after_process':
                                FC(params={'neurons': [number_output_neurons] + 
                                                      params['join2']['fc']['neurons'],
                                           'dropouts': params['join2']['fc']['dropouts'],
                                           'end_layer': False}),
                         'mode': 'cat'
                        }
                )
        
        self.speed_branch = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                   params['speed_branch']['fc']['neurons'] + [1],
                                       'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                       'end_layer': True})

        # Create the fc vector separatedely
        branch_fc_vector = []
        for i in range(params['branches']['number_of_branches']):
            branch_fc_vector.append(FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                            params['branches']['fc']['neurons'] +
                                                            [len(g_conf.TARGETS)],
                                                'dropouts': params['branches']['fc']['dropouts'] + [0.0],
                                                'end_layer': True}))

        self.branches = Branching(branch_fc_vector)  # Here we set branching automatically

        if 'conv' in params['perception']:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
    
    def fuse_inputs(self, data, benchmark=False):
        # fuse sensor inputs together
        sensor_num = 0
        for name in ['rgb', 'seg']:
            if name in g_conf.SENSORS.keys():
                if sensor_num == 0:
                    if benchmark:
                        fused_input = data[name].cuda()
                    else:
                        fused_input = torch.squeeze(data[name].cuda())
                else:
                    if benchmark:
                        fused_input = torch.cat((fused_input, data[name].cuda()), 1)
                    else:
                        fused_input = torch.cat((fused_input, torch.squeeze(data[name].cuda())), 1)
                sensor_num += 1
        return fused_input

    def forward(self, x, a, intentions=None, benchmark=False):
        intentions_rep = None
        x_input = self.fuse_inputs(x, benchmark)
        """ ###### APPLY THE PERCEPTION MODULE """
        x, inter = self.perception(x_input)
        if g_conf.SEPARATE_PERCEPTION_INTENTION_REPS:
            intentions_rep, intentions_inter = self.intention_rep(x_input)
        ## Not a variable, just to store intermediate layers for future vizualization
        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(a)

        # Join measurements and intentions rep
        if "seg" in g_conf.SENSORS.keys():
            intentions_rep = self.intention_branch(intentions)
            m = torch.cat((m, intentions_rep), 1)
        if (g_conf.USE_REPRESENTATION_LOSS and g_conf.USE_INTENTION_REP_LOSS) or g_conf.SEPARATE_PERCEPTION_INTENTION_REPS:
            if not g_conf.SEPARATE_PERCEPTION_INTENTION_REPS:
                intentions_rep = x[:, g_conf.PERCEPTION_REP_SIZE:]
                x = x[:, 0:g_conf.PERCEPTION_REP_SIZE]
            speed_rep = m
            m = torch.cat((m, intentions_rep), 1)
        elif g_conf.USE_REPRESENTATION_LOSS:
            speed_rep = m
        
        """ Join measurements, intentions rep, and perception"""
        j = self.join(x, m)
        branch_outputs = self.branches(j)

        if "seg" in g_conf.SENSORS.keys() or (g_conf.USE_REPRESENTATION_LOSS and g_conf.USE_INTENTION_REP_LOSS) or g_conf.SEPARATE_PERCEPTION_INTENTION_REPS:
            j2 = self.join2(x, intentions_rep)
            speed_branch_output = self.speed_branch(j2)
        else:
            speed_branch_output = self.speed_branch(x)
        # We concatenate speed with the rest.
        if not g_conf.USE_REPRESENTATION_LOSS:
            return branch_outputs + [speed_branch_output]
        return branch_outputs + [speed_branch_output], [x, speed_rep, intentions_rep]

    def forward_branch(self, x, a, branch_number, intentions=None, benchmark=False):
        """
        DO a forward operation and return a single branch.

        Args:
            x: the image input
            a: speed measurement
            branch_number: the branch number to be returned

        Returns:
            the forward operation on the selected branch

        """
        # Convert to integer just in case .
        # TODO: take four branches, this is hardcoded
        if not g_conf.USE_REPRESENTATION_LOSS:
            output_vec = torch.stack(self.forward(x, a, intentions, benchmark)[0:4])
        else:
            branches_with_speed, intermediate_reps = self.forward(x, a, intentions, benchmark)
            output_vec = torch.stack(branches_with_speed[0:4])
        if benchmark or (not g_conf.USE_REPRESENTATION_LOSS):
            return self.extract_branch(output_vec, branch_number)
        return self.extract_branch(output_vec, branch_number), intermediate_reps

    def get_perception_layers(self, x):
        return self.perception.get_layers_features(x)

    def extract_branch(self, output_vec, branch_number):

        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]
        
    # Note that this function should be used only for getting intermediate representations 
    # of squeeze network to be saved for training
    def get_intermediate_representations(self, x, a, intentions=None, benchmark=False):
        intentions_rep = None
        x_input = self.fuse_inputs(x, benchmark)
        """ ###### APPLY THE PERCEPTION MODULE """
        x, inter = self.perception(x_input)
        if g_conf.SEPARATE_PERCEPTION_INTENTION_REPS:
            intentions_rep, intentions_inter = self.intention_rep(x_input)

        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(a)

        # Get intentions rep for squeeze network
        if "seg" in g_conf.SENSORS.keys():
            intentions_rep = self.intention_branch(intentions)
            
        return x, m, intentions_rep
        
