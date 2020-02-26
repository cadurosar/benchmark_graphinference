mkdir $1/data/cora
wget https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.cites -P $1/data/cora/ -c
wget https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.content -P $1/data/cora/ -c