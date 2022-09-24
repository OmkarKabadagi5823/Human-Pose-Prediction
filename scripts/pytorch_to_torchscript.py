import os
import sys

sys.path.append(os.path.dirname(os.environ['HPE_HOME']))

from pose_prediction.models.SeSGCNStudent import Model as SeSGCNModel
from pose_prediction.utils.datasets import CHICO

import torch
from torch.utils.data import DataLoader
import configparser
import argparse

def main():
    args = get_args()
    model_config = load_config(args.config)
    model_name, model_params = model_config.values()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SeSGCNModel(
        model_params['input_channels'],
        model_params['input_frames'],
        model_params['output_frames'],
        model_params['st_gcnn_dropout'],
        model_params['joints_to_consider'],
        model_params['tcnn_layers'],
        model_params['tcnn_kernel_size'],
        model_params['tcnn_dropout']
    ).float().to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    example_inputs = get_example_inputs(args.chico_poses, model_params)
    torchscript = torch.jit.script(model, example_inputs)

    save_path = args.output if args.output is not None else 'torchscipt_{}'.format(model_name)
    torchscript.save(save_path)

def get_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TorchScript model")

    parser.add_argument(
        '--config',
        metavar='model_config_file',
        type=str,
        required=True,
        help='path to model config file'
    )

    parser.add_argument(
        '--checkpoint',
        metavar='model_checkpoint_file',
        type=str,
        required=True,
        help='path to model checkpoint file'
    )

    parser.add_argument(
        '--chico-poses',
        metavar='chico_poses_path',
        type=str,
        required=True,
        help='path to CHICO pose dataset root directory'
    )

    parser.add_argument(
        '--output',
        metavar='out_filename',
        type=str,
        required=False,
        help='name of the output file to write torchscript model'
    )

    return parser.parse_args()

def load_config(config_path):
    config = configparser.ConfigParser()

    config.read(config_path)
   
    model_name = config['model'].get('name')
    model_parameters = config['model.parameters']
   
    model_config = {
        'mode_name': model_name,
        'model_params': {
            'input_channels': model_parameters.getint('input_channels'),
            'input_frames': model_parameters.getint('input_frames'),
            'output_frames': model_parameters.getint('output_frames'),
            'joints_to_consider': model_parameters.getint('joints_to_consider'),
            'st_gcnn_dropout': model_parameters.getfloat('st_gcnn_dropout'),
            'tcnn_layers': model_parameters.getint('tcnn_layers'),
            'tcnn_kernel_size': list(map(int, model_parameters.get('tcnn_kernel_size').split(','))),
            'tcnn_dropout': model_parameters.getfloat('tcnn_dropout')
        }
    }

    return model_config

def get_example_inputs(chico_poses_path, model_params):
    chico_dataset = CHICO.PoseDataset(
        chico_poses_path,
        'train',
        model_params['input_frames'],
        model_params['output_frames'],
        actions=CHICO.normal_actions,
        win_stride=1
    )

    chico_dataset = DataLoader(chico_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

    for i, batch in enumerate(chico_dataset):
        input_sequence = batch[:, 0:model_params['input_frames'], :, :].permute(0, 3, 1, 2).float()

        if i > 1:
            break

    return input_sequence


if __name__ == '__main__':
    main()