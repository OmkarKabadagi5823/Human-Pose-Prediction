import torch
import numpy as np
from models.SeSGCNStudent import Model
from pose_prediction.error import PredictorStateError

import configparser
import time
from typing import Optional, List

class SeSGCNPosePredictor(object):
    def __init__(self):
        self._init_params()
        self._device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    
    def _init_params(self):
        self._model_parameters = None
        self._model = None
        self._maskA = None
        self._maskT = None

    def load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)

        model_parameters = config['model.parameters']

        self._model_parameters = {
            'input_channels': model_parameters.getint('input_channels'),
            'input_frames': model_parameters.getint('input_frames'),
            'output_frames': model_parameters.getint('output_frames'),
            'joints_to_consider': model_parameters.getint('joints_to_consider'),
            'st_gcnn_dropout': model_parameters.getfloat('st_gcnn_dropout'),
            'tcnn_layers': model_parameters.getint('tcnn_layers'),
            'tcnn_kernel_size': list(map(int, model_parameters.get('tcnn_kernel_size').split(','))),
            'tcnn_dropout': model_parameters.getfloat('tcnn_dropout')
        }

    def load_model(self, model_path: str, checkpoint_path: Optional[str] = None):
        if self._model_parameters is None:
            raise PredictorStateError('Model parameters must be loaded before loading the model.')

        self._model = Model(
            self._model_parameters['input_channels'],
            self._model_parameters['input_frames'],
            self._model_parameters['output_frames'],
            self._model_parameters['st_gcnn_dropout'],
            self._model_parameters['joints_to_consider'],
            self._model_parameters['tcnn_layers'],
            self._model_parameters['tcnn_kernel_size'],
            self._model_parameters['tcnn_dropout']
        ).float().to(self._device)

        if checkpoint_path:
            self.load_weights(checkpoint_path)

    def load_weights(self, checkpoint_path: str):
        if self._model is None:
            raise PredictorStateError('Model needs to be loaded before loading the weights')

        self._model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
    
    def load_masks(self, maskA_path: str, maskT_path: str):
        maskA = np.load(maskA_path)
        maskT = np.load(maskT_path)

        self._maskA = torch.tensor(maskA)
        self._maskT = torch.tensor(maskT)
        
        self._maskA = self._maskA.to(self._device)
        self._maskT = self._maskT.to(self._device)

    def eval(self):
        if (
            self._model is None or self._maskA is None or self._maskT is None
        ):
            raise PredictorStateError('Invalid state of the model. Check if model and masks are loaded.')
        
        self._model.eval()

    def predict(self, input_sequence: torch.Tensor):
        predicted_sequence = None
        
        with torch.no_grad():
            predicted_sequence = self._model(input_sequence, self._maskA, self._maskT)
            predicted_sequence = predicted_sequence.permute(0,1,3,2).contiguous()
            
        return predicted_sequence