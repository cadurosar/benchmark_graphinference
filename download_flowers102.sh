wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -P $1/data -c
mkdir $1/data/102flowers/
tar -xzvf $1/data/102flowers.tgz -C $1/data/102flowers/
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat -P $1/data/102flowers/
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat -P $1/data/102flowers/
