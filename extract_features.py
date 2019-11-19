import torchvision
import torch
import numpy as np
import os
import pandas
import tqdm
import sound_feature_extractor

datasets = ['STL',"flowers102",'ESC-50','IMDB']
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
        train_sets = list()
        dataloaders = list()
        if dataset == "STL":
            fold_set = torchvision.datasets.STL10(data_path,folds=0,split="train",download=True,transform=transform_data)
            name = "stl.npz"
        elif dataset == "flowers102":
            fold_set = torchvision.datasets.ImageFolder(os.path.join(data_path,"102flowers","training"),transform=transform_data)
            name = "flowers102.npz"
            
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
            for batch_id, (x, y) in enumerate(dataloader):
                print(x.shape,y.shape)
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
        np.savez(os.path.join(refined_path,"features",name), x=features.reshape(features.shape[0],-1), y=labels)
        if save_raw:
            np.savez(os.path.join(refined_path,"raw",name), x=images.reshape(images.shape[0],-1), y=labels)
    elif dataset == "ESC-50":
        data_path = os.path.join(home,"data")
        esc_path = "ESC-50-master"
        csv_path = os.path.join(esc_path,"meta","esc50.csv")
        audio_path = os.path.join(data_path,esc_path,"audio")
        df = pandas.read_csv(os.path.join(data_path,csv_path))

        labels = list()
        all_features = list()
        if save_raw:
            sound = list()

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
        np.savez(os.path.join(refined_path,"features","esc-50.npz"), x=features.reshape(features.shape[0],-1), y=labels)
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
