# Makefile for Ladder Loss :: Code
# Copyright (C) 2018-2020 Mo Zhou
# License: Proprietary+Confidential. Redistribution is prohibited.
#
# sshfs remote:~/ ~/mnt

SRC := *.py Makefile

RSYNC_FLAGS= \
		--exclude=pretrained/ \
		--exclude=trained \
		--exclude=__pycache__ \
		--exclude=curves/defensive \
		--exclude=curves/vanilla \
		--exclude=curves/mintmaxe \
		--exclude=curves/defense-compare \
		--exclude=curves/logs \
		--exclude=curves/logs-ruthless \
		--exclude=curves/quick-logs \
		--exclude=visualization \
		--exclude=curves/*.pdf \
		--exclude=curves/*.svg

# step 0: data can be downloaded using cocofetch.py
cocofetch:
	wget -c http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
	unzip -d . captions_train-val2014.zip
	mkdir ./pool
	python3 cocofetch.py fetch -j8
	python3 cocofetch.py fetch -j8 -Jannotations/captions_train2014.json
	python3 cocofetch verify
	python3 coocfetch scan

# step 1: preprocess the language part
lingual:
	python3 lingual.py Prepare --save coco.all.toks

# step 2: preprocess the visual part
visual:
	# pre-calcuclate the ResNet152 representations
	python3 visual.py Prepare -D cuda:0 --crop TenCrop --cnn resnet152 \
		--featname avgpool/squeeze+mean0+norm0 --save coco.all.res152
	# pre-calculate the ResNet18 representations
	python3 visual.py Prepare -D cuda:0 --crop TenCrop --cnn resnet18 \
		--featname avgpool/squeeze+mean0+norm0 --save coco.all.res18

# step 3-1: train the VSE model (baseline)
train-vse:
	# ResNet152 version
	python3 thCapRank.py Train -D cuda:0 -L runs/coco-vsepp-res152
	# ResNet18 version
	python3 thCapRank.py Train -D cuda:0 -L runs/coco-vsepp-res18 --cnnpkl coco.all.res18 --cnndim 512

# step 3-2: train the CVSE model (ladder loss)
train-cvse:
	# download en_core_web_lg spacy model if you have not done so
	python3 -m spacy download en_core_web_lg
	# build relevance degree matrix cache (greatly speed up calculation)
	python3 config.ladder.py
	# ResNet152 version
	python3 thCapRank.py Train -D cuda:0 -L runs/coco-ladder-res152 \
		-S runs/coco-vsepp-res152/split_info.json -C config.ladder.py
	# ResNet18 version
	python3 thCapRank.py Train -D cuda:0 -L runs/coco-ladder-res18 \
		-S runs/coco-vsepp-res18/split_info.json -C config.ladder.py \
		--cnnpkl coco.all.res18 --cnndim 512

# step 4-1: evaluate the recall performance of the trained model
eval-recall:
	python3 ladderloss.py doRecall -R runs/coco-vsepp-res152/feat_best.pth
	python3 ladderloss.py doRecall -R runs/coco-ladder-res152/feat_best.pth

# step 4-2: evaluate the kendall tau performance of the trained model
eval-tau:
	python3 ladderloss.py doTau -R runs/coco-vsepp-res152/feat_best.pth \
		-S runs/coco-vsepp-res152/split_info.json -M cbow
	python3 ladderloss.py doTau -R runs/coco-ladder-res152/feat_best.pth \
		-S runs/coco-ladder-res152/split_info.json -M cbow
