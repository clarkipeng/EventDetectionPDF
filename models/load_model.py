from SleepEventDetection.models.BiRNN import MultiBiRNN
from SleepEventDetection.models.PrecTime import PrecTime
from SleepEventDetection.models.UNet1D import UNet1D

def get_model(
    model_name : str,
    objective : str,
    sequence_length : int,
    agg_feats : bool = True,
    use_time_cat : bool = False,
):
    outsize = 2
    if objective[:3] == 'seg':
        outsize = 1
        
    inputsize = 2
    if agg_feats:
        inputsize = 8
    
    if model_name == 'rnn':
        return MultiBiRNN(input_channels = inputsize,
                          n_layers = 2,
                          use_time_cat = use_time_cat,
                          num_classes = outsize,
                         )
    elif model_name == 'unet':
        return UNet1D(input_channels = inputsize,
                      sequence_length = sequence_length, 
                      use_time_cat = use_time_cat,
                      num_classes = outsize,
                      ks = 7,
                     )
    elif model_name == 'unet_t':
        return UNet1D(input_channels = inputsize,
                      sequence_length = sequence_length, 
                      use_time_cat = use_time_cat,
                      num_classes = outsize, 
                      ks = 7,
                      use_attention = True,
                     )
    elif model_name == 'prectime':
        return PrecTime(input_channels = inputsize,
                        sequence_length = sequence_length, 
                        use_time_cat = use_time_cat,
                        num_classes = outsize,
                       )
    raise ValueError('model not listed')