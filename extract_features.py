import torchvision
import torch
import numpy as np
import os
import pandas
import tqdm
import sound_feature_extractor
import sklearn
import sklearn.preprocessing
import scipy

ESC_PATH = "ESC-50-master"
datasets = ['STL',"flowers102",'ESC-50','cora',"toronto"]
dataset_default = "STL"
home = os.path.expanduser("~")

data_path_default = os.path.join(home,"data")
refined_path_default = "refined_datasets/"

def extract_features(dataset=dataset_default, data_path=data_path_default, refined_path=refined_path_default, save_raw=False):

    if dataset == "STL" or dataset=="flowers102":

        transform_data = torchvision.transforms.Compose([
#            torchvision.transforms.Resize(342),
            torchvision.transforms.Resize(299),
            torchvision.transforms.CenterCrop(299),
            torchvision.transforms.ToTensor(),
            ])

        model=torchvision.models.inception_v3(pretrained=True,transform_input=True)
        ## Remove the last layer
        model.fc = torch.nn.Sequential()
        model.eval()
        model.cuda()
        if dataset == "STL":
            fold_set = torchvision.datasets.STL10(data_path,folds=0,split="train",download=True,transform=transform_data)
            name = "stl"
        elif dataset == "flowers102":
            fold_set = torchvision.datasets.ImageFolder(os.path.join(data_path,"102flowers","training"),transform=transform_data)
            name = "flowers102"
            
        dataloader = torch.utils.data.DataLoader(
                fold_set,
                batch_size=50,
                shuffle=False,
                num_workers=8)

        labels = list()
        features = list()
        if save_raw:
            images = list()
        with torch.no_grad():
            for (x, y) in dataloader:
                if save_raw:
                    images.append(x.cpu().numpy())
                new_x = model(x.cuda()) 
                features.append(new_x.cpu().numpy())
                labels.append(y.numpy())        
        if save_raw:
            images = np.concatenate(images)
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        print(features.shape,labels.shape,np.bincount(labels))
        np.savez(os.path.join(refined_path,"features",name+".npz"), x=features.reshape(features.shape[0],-1), y=labels)

        #Prepare for matlab
        matlab_dict = dict(x=features.reshape(features.shape[0],-1).T,y=labels)
        scipy.io.savemat(os.path.join(refined_path,"features",name+".mat"),matlab_dict)

        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        features = scaler.fit_transform(features.reshape(features.shape[0],-1))
        matlab_dict = dict(x=features.reshape(features.shape[0],-1).T,y=labels)
        scipy.io.savemat(os.path.join(refined_path,"features",name+"_minmaxscaled.mat"),matlab_dict)

        if save_raw:
            np.savez(os.path.join(refined_path,"raw",name), x=images.reshape(images.shape[0],-1), y=labels)
    elif dataset == "cora":
        cora_path = os.path.join(data_path,"cora")
        cora_content = os.path.join(cora_path,"cora.content")
        name = "cora"
        feature_columns = ["f_{}".format(i) for i in range(1433)]
        class_column = "class"
        columns =  feature_columns + [class_column]
        nodes = pandas.read_csv(cora_content, sep='\t', names=columns, header=None)
        features = nodes[feature_columns].to_numpy().astype(np.float32)
        labels = sklearn.preprocessing.LabelEncoder().fit_transform(nodes[class_column].to_numpy()).astype(np.int32)
        np.savez(os.path.join(refined_path,"features",name+".npz"), x=features.reshape(features.shape[0],-1), y=labels)
        print(features.shape,labels.shape,np.bincount(labels))

        #Prepare for matlab
        matlab_dict = dict(x=features.reshape(features.shape[0],-1).T,y=labels)
        scipy.io.savemat(os.path.join(refined_path,"features",name+".mat"),matlab_dict)

        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        features = scaler.fit_transform(features.reshape(features.shape[0],-1))
        matlab_dict = dict(x=features.reshape(features.shape[0],-1).T,y=labels)
        scipy.io.savemat(os.path.join(refined_path,"features",name+"_minmaxscaled.mat"),matlab_dict)


        pass
    elif dataset == "toronto":
        toronto_path = os.path.join(data_path,"toronto")
        clean_path = os.path.join(toronto_path,"Toronto.mat")
        noisy_path = os.path.join(toronto_path,"Toronto_SNR7.mat")
        matlab_clean = scipy.io.loadmat(clean_path)
        real_signal = np.array(matlab_clean["G"][0][0][2])
        matlab_noisy = scipy.io.loadmat(noisy_path)
        features = np.array(matlab_noisy["G"][0][0][2])
        name = "toronto"
        np.savez(os.path.join(refined_path,"features",name+".npz"), x=features, y=real_signal)
        #Prepare matlab
        matlab_dict = dict(x=features.T,y=real_signal.T)
        scipy.io.savemat(os.path.join(refined_path,"features",name+".mat"),matlab_dict)
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        features = scaler.fit_transform(features.reshape(features.shape[0],-1))
        matlab_dict = dict(x=features.reshape(features.shape[0],-1).T,y=real_signal)
        scipy.io.savemat(os.path.join(refined_path,"features",name+"_minmaxscaled.mat"),matlab_dict)
        print(name+"_minmaxscaled.mat")
        print(features.shape)

    elif dataset == "ESC-50":
        csv_path = os.path.join(ESC_PATH,"meta","esc50.csv")
        audio_path = os.path.join(data_path,ESC_PATH,"audio")
        df = pandas.read_csv(os.path.join(data_path,csv_path))
        name = "esc-50"
        labels = list()
        all_features = list()
        if save_raw:
            sounds = list()

        for idx, line in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
            filename = line["filename"]
            target = line["target"]
            _input = sound_feature_extractor.file_to_input(os.path.join(audio_path,filename))
            if save_raw:
                sounds.append(_input)
            extractor = sound_feature_extractor.get_extractor(pre_model_path="sound_feature_extractor/pretrained_model.pkl")   
            features = sound_feature_extractor.get_features(extractor,_input)
            features = features.cpu().numpy()
            all_features.append(features)
            labels.append([target])
        if save_raw:
            sounds = np.concatenate(sounds)    
        features = np.concatenate(all_features)
        labels = np.concatenate(labels)
        print(np.bincount(labels),features.shape,labels.shape)
        np.savez(os.path.join(refined_path,"features",name+".npz"), x=features.reshape(features.shape[0],-1), y=labels)
        #Prepare for matlab
        matlab_dict = dict(x=features.reshape(features.shape[0],-1).T,y=labels)
        scipy.io.savemat(os.path.join(refined_path,"features",name+".mat"),matlab_dict)

        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        features = scaler.fit_transform(features.reshape(features.shape[0],-1))
        matlab_dict = dict(x=features.reshape(features.shape[0],-1).T,y=labels)
        scipy.io.savemat(os.path.join(refined_path,"features",name+"_minmaxscaled.mat"),matlab_dict)


        if save_raw:
            np.savez(os.path.join(refined_path,"raw","esc-50.npz"), x=images.reshape(images.shape[0],-1), y=labels)
        
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract features from dataset.')
    parser.add_argument('--dataset',
                          choices=datasets, default=dataset_default,
                          help='Dataset to extract features')
    parser.add_argument('--data_path',
                          type=str, default=data_path_default,
                          help='Dataset to extract features')
    parser.add_argument('--refined_path',
                          type=str, default=refined_path_default,
                          help='Refined dataset path')

    parser.add_argument('--save_raw',
                          action="store_true", default=False,
                          help='Save raw dataset (images/sound/text)')
    
    args = parser.parse_args()
    extract_features(args.dataset,args.data_path,args.refined_path,args.save_raw)
