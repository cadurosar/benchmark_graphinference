import scipy
import scipy.io
import os
import shutil
import tqdm

home = os.path.expanduser("~")
data_path_default = os.path.join(home,"data")

def prepare_flowers(data_path=data_path_default):
    flowers_path =  os.path.join(data_path,"102flowers")
    img_path = os.path.join(flowers_path,"jpg")
    
    labels = scipy.io.loadmat(os.path.join(flowers_path,"imagelabels.mat"))["labels"][0]
    setid = scipy.io.loadmat(os.path.join(flowers_path,"setid.mat"))["trnid"][0]

    for a in range(max(labels)):
        os.makedirs(os.path.join(flowers_path,"training",str(a)))

    all_images = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    len(all_images),all_images[-1]
    for filename in sorted(tqdm.tqdm(all_images)):
        _id = int(filename.split(".")[0].split("_")[1])
        if _id in setid:
            target = labels[_id]-1
            source = os.path.join(img_path,filename)
            destination = os.path.join(flowers_path,"training",str(target),filename)
            shutil.copyfile(source, destination)    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare Flowers102 dataset.')
    parser.add_argument('--data_path',
                          type=str, default=data_path_default,
                          help='Dataset to extract features')
    
    args = parser.parse_args()
    prepare_flowers(args.data_path)
