import librosa as lib
import numpy as np
import network_architectures as netark
import torch.nn.functional as Fx
import torch
from torch.autograd import Variable
import sys,os
from collections import OrderedDict
import extractor as exm


usegpu = True


n_fft = 1024
hop_length = 512
n_mels = 128
trainType = 'weak_mxh64_1024'
pre_model_path = 'mx-h64-1024_0d3-1.17.pkl'
featType = 'layer18' # or layer 19 -  layer19 might not work well
globalpoolfn = Fx.avg_pool2d # can use max also
netwrkgpl = Fx.avg_pool2d # keep it fixed



def load_model(netx,modpath):
    #load through cpu -- safest
    state_dict = torch.load(modpath,map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    netx.load_state_dict(new_state_dict)

def get_features(extractor,inpt):
    # return pytorch tensor 
    extractor.eval()
    indata = torch.Tensor(inpt)
    if usegpu:
        indata = indata.cuda()
    with torch.no_grad():
        pred = extractor(indata)
        if len(pred.size()) > 2:
            gpred = globalpoolfn(pred,kernel_size=pred.size()[2:])
            gpred = gpred.view(gpred.size(0),-1)

    return gpred

def file_to_input(filename,srate=44100):

    try:
        y, sr = lib.load(filename,sr=None)
    except:
        raise IOError('Give me an audio  file which I can read!!')
    
    if len(y.shape) > 1:
        print ('Mono Conversion') 
        y = lib.to_mono(y)

    if sr != srate:
        print ('Resampling to {}'.format(srate))
        y = lib.resample(y,sr,srate)

        
    mel_feat = lib.feature.melspectrogram(y=y,sr=srate,n_fft=n_fft,hop_length=hop_length,n_mels=128)
    inpt = lib.power_to_db(mel_feat).T  

    # input needs to be 4D, batch_size X 1 X inpt_size[0] X inpt_size[1]
    inpt = np.reshape(inpt,(1,1,inpt.shape[0],inpt.shape[1]))
    return inpt

def get_extractor(network_architecture=netark, trainType=trainType, global_average_pooling=netwrkgpl, pre_model_path=pre_model_path, use_gpu=usegpu, feature_type=featType):
    netType = getattr(network_architecture,trainType)
    net = netType(527,global_average_pooling)
    load_model(net,pre_model_path)
    if use_gpu:
        net.cuda()
    feat_extractor = exm.featExtractor(net,feature_type)
        
    return feat_extractor

def extract_features(_input,**kwargs):
    extractor = get_extractor(**kwargs)   
    features = get_features(extractor,_input)
    return features

#keep sample rate 44100, no need 
def main(filename,srate=44100):

    _input = file_to_input(filename,srate)
    feature = extract_features(_input)
    extractor = get_extractor()   
    features = get_features(extractor,_input)
    features = features.cpu().numpy()

    return features


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError(' You need to give filename as first argument..Duhhh!!')
    if not os.path.isfile(sys.argv[1]):
        raise ValueError('give me a audio file which exist!!!')
    
    main(sys.argv[1])
