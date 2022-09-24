import os
import sys

sys.path.append(os.path.dirname(os.environ['HPE_HOME']))

from pose_prediction.pose_predictor import SeSGCNPosePredictor
from pose_prediction.utils.datasets import CHICO
from torch.utils.data import DataLoader

import argparse
import configparser
import logging

def main():
    args = get_args()

    pose_predictor = SeSGCNPosePredictor()
    pose_predictor.load_config(args.config)
    pose_predictor.create_model(args.checkpoint)
    pose_predictor.load_masks(args.maskA, args.maskT)
    pose_predictor.eval()

    model_config = load_config(args.config)
    _, model_params = model_config.values()

    example_inputs = get_example_inputs(args.chico_poses, model_params)

    predicted_sequences = pose_predictor.predict(example_inputs)

    (batch_size, n_output_frames, n_joints, n_channels) = predicted_sequences.shape

    assert (n_output_frames, n_joints, n_channels) == (25, 15, 3)
    
    logging.info("predicted sequence format:\n\tn_output_frames: {}\n\tn_joints: {}\n\tn_channels: {}".format(
        n_output_frames, n_joints, n_channels)) 

    logging.info("Test Successful")

def get_args():
    parser = argparse.ArgumentParser(description="Test the student model on CHICO dataset")

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
        '--maskA',
        metavar='maskA_file',
        type=str,
        required=True,
        help='path to maskA file'
    )

    parser.add_argument(
        '--maskT',
        metavar='maskT_file',
        type=str,
        required=True,
        help='path to maskT file'
    )

    parser.add_argument(
        '--chico-poses',
        metavar='chico_poses_path',
        type=str,
        required=True,
        help='path to CHICO pose dataset root directory'
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

    chico_dataset = DataLoader(chico_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    for i, batch in enumerate(chico_dataset):
        input_sequence = batch[:, 0:model_params['input_frames'], :, :].permute(0, 3, 1, 2).float()

        if i > 1:
            break

    return input_sequence

if __name__ == "__main__":
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
    main()