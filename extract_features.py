import torchvision
import torch
import numpy as np
import os

datasets = ['STL','AudioSet','IMDB']
dataset_default = "STL"
data_path_default = os.path.join("~","data")
refined_path_default = "refined_datasets/"

def extract_features(dataset=dataset_default, data_path=data_path_default, refined_path=refined_path_default, save_raw=False):

    if dataset == "STL":

        transform_data = torchvision.transforms.Compose([
            torchvision.transforms.Resize(299),
            torchvision.transforms.ToTensor(),
            ])

        model=torchvision.models.inception_v3(pretrained=True)
        ## Remove the last layer
        model.fc = torch.nn.Sequential()
        model.eval()
        model.cuda()
        train_sets = list()
        dataloaders = list()
        fold_set = torchvision.datasets.STL10(data_path,folds=0,split="train",download=True,transform=transform_data)
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
                if save_raw:
                    images.append(x.cpu().numpy())
                new_x = model(x.cuda()) 
                features.append(new_x.cpu().numpy())
                labels.append(y.numpy())        
        if save_raw:
            images = np.concatenate(images)
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        print(np.bincount(labels))
        np.savez(os.path.join(refined_path,"features","stl.npz"), x=features.reshape(features.shape[0],-1), y=labels)
        if save_raw:
            np.savez(os.path.join(refined_path,"raw","stl.npz"), x=images.reshape(images.shape[0],-1), y=labels)
    
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
