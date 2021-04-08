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
	python3 cocofetch.py verify
	python3 coocfetch.py scan

# step 1: preprocess the language part
lingual:
	python3 lingual.py Prepare --save coco.all.toks
	python3 lingual.py Prepare --save f30k.all.toks --DATASET F30K \
		--srcanno /dev/null --srcannoval /dev/null

# step 2: preprocess the visual part
visual:
	# pre-calcuclate the ResNet152 representations
	python3 visual.py Prepare -D cuda:0 --crop TenCrop --cnn resnet152 \
		--featname avgpool/squeeze+mean0+norm0 --save coco.all.res152
	# pre-calculate the ResNet18 representations
	python3 visual.py Prepare -D cuda:0 --crop TenCrop --cnn resnet18 \
		--featname avgpool/squeeze+mean0+norm0 --save coco.all.res18
	# pre-calculate the VGG19 representations
	python3 visual.py Prepare -D cuda:0 --crop TenCrop --cnn vgg19 \
		--featname classifier.4/squeeze+mean0+norm0 --save coco.all.vgg19
	#
	python3 visual.py Prepare -D cuda --pool /dev/shm/flickr30k-images/ \
		--crop TenCrop --cnn vgg19 --featname classifier.4/squeeze+mean0+norm0 \
		--save f30k.all.vgg19 --dataset F30K
	python3 visual.py Prepare -D cuda --pool /dev/shm/flickr30k-images/ \
		--crop TenCrop --cnn resnet152 --featname avgpool/squeeze+mean0+norm0 \
		--save f30k.all.res152--dataset F30K

# step 3-1: train the VSE model (baseline)
train-vse:
	# ResNet152 version
	python3 thCapRank.py Train -D cuda:0 -L runs/coco-vsepp-res152
	# ResNet18 version
	python3 thCapRank.py Train -D cuda:0 -L runs/coco-vsepp-res18 \
		-S runs/coco-vsepp-res152/split_info.json \
		--cnnpkl coco.all.res18 --cnndim 512
	# VGG19 version
	python3 thCapRank.py Train -D cuda:0 -L runs/coco-vsepp-vgg19 \
		-S runs/coco-vsepp-res152/split_info.json \
		--cnnpkl coco.all.vgg19 --cnndim 4096

# step 3-2: train the CVSE model (ladder loss)
train-cvse:
	# download en_core_web_lg spacy model if you have not done so
	python3 -m spacy download en_core_web_lg
	# build relevance degree matrix cache (greatly speed up calculation)
	python3 config.ladder.py
	python3 config.ladderf30k.py
	# ResNet152 version
	python3 thCapRank.py Train -D cuda:0 -L runs/coco-ladder-res152 \
		-S runs/coco-vsepp-res152/split_info.json -C config.ladder.py
	# ResNet18 version
	python3 thCapRank.py Train -D cuda:0 -L runs/coco-ladder-res18 \
		-S runs/coco-vsepp-res152/split_info.json -C config.ladder.py \
		--cnnpkl coco.all.res18 --cnndim 512
	# VGG19 version
	python3 thCapRank.py Train -D cuda:0 -L runs/coco-ladder-vgg19 \
		-S runs/coco-vsepp-res152/split_info.json -C config.ladder.py \
		--cnnpkl coco.all.vgg19 --cnndim 4096

train-ft:
	python3 thCapRank.py Train -D cuda:0 -L runs/vsecoft-r152 \
		--cnndim 2048 --finetune Resnet152 --cocopool ~/COCO \
		-S runs/coco-vsepp-res152/split_info.json \
		--snapshot runs/coco-vsepp-res152/snapshot_latest.pth
	python3 thCapRank.py Train -D cuda:0 -L runs/ladcoft-r152 \
		--cnndim 2048 --finetune Resnet152 --cocopool ~/COCO \
		-S runs/coco-ladder-res152/split_info.json \
		--snapshot runs/coco-ladder-res152/snapshot_latest.pth

# step 4-1: evaluate the recall performance of the trained model
eval-recall:
	python3 ladderloss.py doRecall -R runs/coco-vsepp-res152/feat_best.pth
	python3 ladderloss.py doRecall -R runs/coco-ladder-res152/feat_best.pth
	python3 ladderloss.py doRecall -R runs/coco-vsepp-vgg19/feat_best.pth
	python3 ladderloss.py doRecall -R runs/coco-ladder-vgg19/feat_best.pth

# step 4-2: evaluate the kendall tau performance of the trained model
eval-tau:
	python3 ladderloss.py doTau -R runs/coco-vsepp-res152/feat_best.pth \
		-S runs/coco-vsepp-res152/split_info.json -M cbow
	python3 ladderloss.py doTau -R runs/coco-ladder-res152/feat_best.pth \
		-S runs/coco-ladder-res152/split_info.json -M cbow
	python3 ladderloss.py doTau -R runs/coco-vsepp-vgg19/feat_best.pth \
		-S runs/coco-vsepp-res152/split_info.json -M cbow
	python3 ladderloss.py doTau -R runs/coco-ladder-vgg19/feat_best.pth \
		-S runs/coco-ladder-res152/split_info.json -M cbow

# step 5-1: Train with SBERT and evaluate
train-sbert:
	python3 thCapRank.py Train -D cuda -L runs/coco-sbert-res152 \
		-S runs/coco-vsepp-res152/split_info.json -C config.sbert.py
TOKEN ?= sbert
eval-sbert:
	python3 ladderloss.py doRecall -R runs/coco-$(TOKEN)-res152/feat_best.pth
	python3 ladderloss.py doTau -R runs/coco-$(TOKEN)-res152/feat_best.pth \
		-S runs/coco-vsepp-res152/split_info.json -M sbert

# step5-2: Train with adaptive ladder loss
train-akl: 
	python3 thCapRank.py Train -D cuda -L runs/coco-akl-res152 \
		-S runs/coco-vsepp-res152/split_info.json -C config.akl.py
eval-akl:
	$(MAKE) eval-sbert TOKEN=akl


submit-train-all:
	# The first one should be started manually
	tmux new-window "CUDA_VISIBLE_DEVICES=4 python3 thCapRank.py Train -D cuda -L trained/co-vse-r152"
	tmux new-window "CUDA_VISIBLE_DEVICES=5 python3 thCapRank.py Train -D cuda -L trained/co-vse-v19  -S trained/co-vse-r152/split_info.json --cnnpkl coco.all.vgg19 --cnndim 4096"
	tmux new-window "CUDA_VISIBLE_DEVICES=6 python3 thCapRank.py Train -D cuda -L trained/co-lad-r152 -S trained/co-vse-r152/split_info.json -C config.ladder.py"
	tmux new-window "CUDA_VISIBLE_DEVICES=7 python3 thCapRank.py Train -D cuda -L trained/co-lad-v19  -S trained/co-vse-r152/split_info.json -C config.ladder.py --cnnpkl coco.all.vgg19 --cnndim 4096"

batch-coco:
	# Models with no fine-tune
	python3 thCapRank.py Train -D cuda -L batch/coco-vsepp-res152
	python3 ladderloss.py doRecall     -R batch/coco-vsepp-res152/feat_best.pth
	python3 ladderloss.py doTau        -R batch/coco-vsepp-res152/feat_best.pth  -S batch/coco-vsepp-res152/split_info.json -M sbert
	python3 thCapRank.py Train -D cuda -L batch/test                             -S batch/coco-vsepp-res152/split_info.json --snapshot batch/coco-vsepp-res152/model_best.pth --eval5k
	python3 ladderloss.py doTau        -R batch/test/feat_eval5k.pth             -S batch/coco-vsepp-res152/split_info.json -M sbert --valsize -1
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-vsepp-vgg19   -S batch/coco-vsepp-res152/split_info.json --cnnpkl coco.all.vgg19 --cnndim 4096
	python3 ladderloss.py doRecall     -R batch/coco-vsepp-vgg19/feat_best.pth
	python3 ladderloss.py doTau        -R batch/coco-vsepp-vgg19/feat_best.pth   -S batch/coco-vsepp-res152/split_info.json -M sbert
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-ladder-res152 -S batch/coco-vsepp-res152/split_info.json                                       -C config.ladder.py
	python3 ladderloss.py doRecall     -R batch/coco-ladder-res152/feat_best.pth
	python3 ladderloss.py doTau        -R batch/coco-ladder-res152/feat_best.pth -S batch/coco-vsepp-res152/split_info.json -M sbert
	python3 thCapRank.py Train -D cuda -L batch/test                             -S batch/coco-vsepp-res152/split_info.json --snapshot batch/coco-ladder-res152/model_best.pth --eval5k
	python3 ladderloss.py doTau        -R batch/test/feat_eval5k.pth             -S batch/coco-vsepp-res152/split_info.json -M sbert --valsize -1
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-ladder-vgg19  -S batch/coco-vsepp-res152/split_info.json --cnnpkl coco.all.vgg19 --cnndim 4096 -C config.ladder.py 
	python3 ladderloss.py doRecall     -R batch/coco-ladder-vgg19/feat_best.pth
	python3 ladderloss.py doTau        -R batch/coco-ladder-vgg19/feat_best.pth  -S batch/coco-vsepp-res152/split_info.json -M sbert
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-sbert-res152  -S batch/coco-vsepp-res152/split_info.json                                       -C config.sbert.py
	python3 ladderloss.py doRecall     -R batch/coco-sbert-res152/feat_best.pth
	python3 ladderloss.py doTau        -R batch/coco-sbert-res152/feat_best.pth  -S batch/coco-vsepp-res152/split_info.json -M sbert
	python3 thCapRank.py Train -D cuda -L batch/test                             -S batch/coco-vsepp-res152/split_info.json --snapshot batch/coco-sbert-res152/model_best.pth --eval5k
	python3 ladderloss.py doTau        -R batch/test/feat_eval5k.pth             -S batch/coco-vsepp-res152/split_info.json -M sbert --valsize -1
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-sbert-vgg19   -S batch/coco-vsepp-res152/split_info.json --cnnpkl coco.all.vgg19 --cnndim 4096 -C config.sbert.py
	python3 ladderloss.py doRecall     -R batch/coco-sbert-vgg19/feat_best.pth
	python3 ladderloss.py doTau        -R batch/coco-sbert-vgg19/feat_best.pth   -S batch/coco-vsepp-res152/split_info.json -M sbert
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-akl-res152  -S batch/coco-vsepp-res152/split_info.json                                       -C config.akl.py
	python3 thCapRank.py Train -D cuda -L batch/test                             -S batch/coco-vsepp-res152/split_info.json --snapshot batch/coco-akl-res152/model_best.pth --eval5k
	python3 ladderloss.py doTau        -R batch/test/feat_eval5k.pth             -S batch/coco-vsepp-res152/split_info.json -M sbert --valsize -1
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-akl-vgg19   -S batch/coco-vsepp-res152/split_info.json --cnnpkl coco.all.vgg19 --cnndim 4096 -C config.akl.py
	# Models with fine-tune
	python3 thCapRank.py Train -D cuda -L batch/coco-vsepp-res152ft  -S batch/coco-vsepp-res152/split_info.json --finetune Resnet152 --cnndim 2048 --maxepoch 15 --cocopool ~/COCO --snapshot batch/coco-vsepp-res152/snapshot_latest.pth
	python3 ladderloss.py doRecall     -R batch/coco-vsepp-res152ft/feat_best.pth
	python3 ladderloss.py doTau        -R batch/coco-vsepp-res152ft/feat_best.pth -S batch/coco-vsepp-res152/split_info.json -M sbert
	python3 thCapRank.py Train -D cuda -L batch/test                              -S batch/coco-vsepp-res152/split_info.json --snapshot batch/coco-vsepp-res152ft/model_best.pth --eval5k --finetune Resnet152 --cocopool ~/COCO --finetune_snapshot batch/coco-vsepp-res152ft/finetune_best.pth
	python3 ladderloss.py doTau        -R batch/test/feat_eval5k.pth              -S batch/coco-vsepp-res152/split_info.json -M sbert --valsize -1
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-vsepp-vgg19ft   -S batch/coco-vsepp-res152/split_info.json --finetune VGG19     --cnndim 4096 --maxepoch 15 --cocopool ~/COCO --snapshot batch/coco-vsepp-vgg19/snapshot_latest.pth
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-ladder-res152ft -S batch/coco-vsepp-res152/split_info.json --finetune Resnet152 --cnndim 2048 --maxepoch 15 --cocopool ~/COCO --snapshot batch/coco-ladder-res152/snapshot_latest.pth -C config.ladder.py
	python3 ladderloss.py doRecall     -R batch/coco-ladder-res152ft/feat_best.pth
	python3 ladderloss.py doTau        -R batch/coco-ladder-res152ft/feat_best.pth -S batch/coco-vsepp-res152/split_info.json -M sbert
	python3 thCapRank.py Train -D cuda -L batch/test                              -S batch/coco-vsepp-res152/split_info.json --snapshot batch/coco-ladder-res152ft/model_best.pth --eval5k --finetune Resnet152 --cocopool ~/COCO --finetune_snapshot batch/coco-ladder-res152ft/finetune_best.pth
	python3 ladderloss.py doTau        -R batch/test/feat_eval5k.pth              -S batch/coco-vsepp-res152/split_info.json -M sbert --valsize -1
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-ladder-vgg19ft  -S batch/coco-vsepp-res152/split_info.json --finetune VGG19     --cnndim 4096 --maxepoch 15 --cocopool ~/COCO --snapshot batch/coco-ladder-vgg19/snapshot_latest.pth  -C config.ladder.py
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-sbert-res152ft  -S batch/coco-vsepp-res152/split_info.json --finetune Resnet152 --cnndim 2048 --maxepoch 15 --cocopool ~/COCO --snapshot batch/coco-sbert-res152/snapshot_latest.pth  -C config.sbert.py
	python3 ladderloss.py doRecall     -R batch/coco-sbert-res152ft/feat_best.pth
	python3 ladderloss.py doTau        -R batch/coco-sbert-res152ft/feat_best.pth -S batch/coco-vsepp-res152/split_info.json -M sbert
	python3 thCapRank.py Train -D cuda -L batch/test                              -S batch/coco-vsepp-res152/split_info.json --snapshot batch/coco-sbert-res152ft/model_best.pth --eval5k --finetune Resnet152 --cocopool ~/COCO --finetune_snapshot batch/coco-sbert-res152ft/finetune_best.pth
	python3 ladderloss.py doTau        -R batch/test/feat_eval5k.pth              -S batch/coco-vsepp-res152/split_info.json -M sbert --valsize -1
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-sbert-vgg19ft   -S batch/coco-vsepp-res152/split_info.json --finetune VGG19     --cnndim 4096 --maxepoch 15 --cocopool ~/COCO --snapshot batch/coco-sbert-vgg19/snapshot_latest.pth   -C config.sbert.py
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-akl-res152ft  -S batch/coco-vsepp-res152/split_info.json --finetune Resnet152 --cnndim 2048 --maxepoch 15 --cocopool ~/COCO --snapshot batch/coco-akl-res152/snapshot_latest.pth  -C config.sbert.py
	python3 ladderloss.py doRecall     -R batch/coco-akl-res152ft/feat_best.pth
	python3 ladderloss.py doTau        -R batch/coco-akl-res152ft/feat_best.pth -S batch/coco-vsepp-res152/split_info.json -M sbert
	python3 thCapRank.py Train -D cuda -L batch/test                            -S batch/coco-vsepp-res152/split_info.json --snapshot batch/coco-akl-res152ft/model_best.pth --eval5k --finetune Resnet152 --cocopool ~/COCO --finetune_snapshot batch/coco-akl-res152ft/finetune_best.pth
	python3 ladderloss.py doTau        -R batch/test/feat_eval5k.pth              -S batch/coco-vsepp-res152/split_info.json -M sbert --valsize -1
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-akl-vgg19ft  -S batch/coco-vsepp-res152/split_info.json --finetune VGG19 --cnndim 4096 --maxepoch 15 --cocopool ~/COCO --snapshot batch/coco-akl-vgg19/snapshot_latest.pth  -C config.akl.py
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-sbert-vgg19ft  -S batch/coco-vsepp-res152/split_info.json --finetune VGG19 --cnndim 4096 --maxepoch 15 --cocopool ~/COCO --snapshot batch/coco-sbert-vgg19/snapshot_latest.pth  -C config.sbert.py
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-ladder-vgg19ft  -S batch/coco-vsepp-res152/split_info.json --finetune VGG19 --cnndim 4096 --maxepoch 15 --cocopool ~/COCO --snapshot batch/coco-sbert-vgg19/snapshot_latest.pth  -C config.ladder.py
	#
	python3 thCapRank.py Train -D cuda -L batch/coco-vsepp-vgg19ft  -S batch/coco-vsepp-res152/split_info.json --finetune VGG19 --cnndim 4096 --maxepoch 15 --cocopool ~/COCO --snapshot batch/coco-sbert-vgg19/snapshot_latest.pth

batch-f30k:
	# models with no fine-tune
	python3 thCapRank.py Train -D cuda --cnnpkl f30k.all.res152 --tokspkl f30k.all.toks -L batch/f30k-vsepp-res152
	python3 ladderloss.py doRecall     -R batch/f30k-vsepp-res152/feat_best.pth;  python3 ladderloss.py doTau        -R batch/f30k-vsepp-res152/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	python3 thCapRank.py Train -D cuda --cnnpkl f30k.all.res152 --tokspkl f30k.all.toks -L batch/f30k-ladder-res152 -S batch/f30k-vsepp-res152/split_info.json -C config.ladderf30k.py
	python3 ladderloss.py doRecall     -R batch/f30k-ladder-res152/feat_best.pth; python3 ladderloss.py doTau        -R batch/f30k-ladder-res152/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	python3 thCapRank.py Train -D cuda --cnnpkl f30k.all.res152 --tokspkl f30k.all.toks -L batch/f30k-sbert-res152  -S batch/f30k-vsepp-res152/split_info.json -C config.sbertf30k.py
	python3 ladderloss.py doRecall     -R batch/f30k-sbert-res152/feat_best.pth;  python3 ladderloss.py doTau        -R batch/f30k-sbert-res152/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	python3 thCapRank.py Train -D cuda --cnnpkl f30k.all.res152 --tokspkl f30k.all.toks -L batch/f30k-akl-res152  -S batch/f30k-vsepp-res152/split_info.json -C config.aklf30k.py
	python3 ladderloss.py doRecall     -R batch/f30k-akl-res152/feat_best.pth;  python3 ladderloss.py doTau        -R batch/f30k-akl-res152/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	python3 thCapRank.py Train -D cuda --cnnpkl f30k.all.vgg19  --tokspkl f30k.all.toks -L batch/f30k-vsepp-vgg19   -S batch/f30k-vsepp-res152/split_info.json --cnndim 4096
	python3 ladderloss.py doRecall     -R batch/f30k-vsepp-vgg19/feat_best.pth;   python3 ladderloss.py doTau        -R batch/f30k-vsepp-vgg19/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	python3 thCapRank.py Train -D cuda --cnnpkl f30k.all.vgg19  --tokspkl f30k.all.toks -L batch/f30k-ladder-vgg19  -S batch/f30k-vsepp-res152/split_info.json --cnndim 4096 -C config.ladderf30k.py
	python3 ladderloss.py doRecall     -R batch/f30k-ladder-vgg19/feat_best.pth;   python3 ladderloss.py doTau       -R batch/f30k-ladder-vgg19/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	python3 thCapRank.py Train -D cuda --cnnpkl f30k.all.vgg19  --tokspkl f30k.all.toks -L batch/f30k-sbert-vgg19   -S batch/f30k-vsepp-res152/split_info.json --cnndim 4096 -C config.sbertf30k.py
	python3 ladderloss.py doRecall     -R batch/f30k-sbert-vgg19/feat_best.pth;   python3 ladderloss.py doTau        -R batch/f30k-sbert-vgg19/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	python3 thCapRank.py Train -D cuda --cnnpkl f30k.all.vgg19  --tokspkl f30k.all.toks -L batch/f30k-akl-vgg19   -S batch/f30k-vsepp-res152/split_info.json --cnndim 4096 -C config.aklf30k.py
	python3 ladderloss.py doRecall     -R batch/f30k-akl-vgg19/feat_best.pth;   python3 ladderloss.py doTau        -R batch/f30k-akl-vgg19/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	# models with fine-tune
	F30K=1 python3 thCapRank.py Train -D cuda --tokspkl f30k.all.toks -L batch/f30k-vsepp-res152ft -S batch/f30k-vsepp-res152/split_info.json --finetune Resnet152 --cnndim 2048 --maxepoch 15 --snapshot batch/f30k-vsepp-res152/snapshot_latest.pth --cocopool /dev/shm/flickr30k-images
	python3 ladderloss.py doRecall -R batch/f30k-vsepp-res152ft/feat_best.pth; python3 ladderloss.py doTau -R batch/f30k-vsepp-res152ft/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	# FIXME: BUG -C config.ladderf30k.py
	F30K=1 python3 thCapRank.py Train -D cuda --tokspkl f30k.all.toks -L batch/f30k-ladder-res152ft -S batch/f30k-vsepp-res152/split_info.json --finetune Resnet152 --cnndim 2048 --maxepoch 15 --snapshot batch/f30k-ladder-res152/snapshot_latest.pth --cocopool /dev/shm/flickr30k-images
	python3 ladderloss.py doRecall -R batch/f30k-ladder-res152ft/feat_best.pth; python3 ladderloss.py doTau -R batch/f30k-ladder-res152ft/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	# FIXME: BUG -C config.sbertf30k.py
	F30K=1 python3 thCapRank.py Train -D cuda --tokspkl f30k.all.toks -L batch/f30k-sbert-res152ft -S batch/f30k-vsepp-res152/split_info.json --finetune Resnet152 --cnndim 2048 --maxepoch 15 --snapshot batch/f30k-sbert-res152/snapshot_latest.pth --cocopool /dev/shm/flickr30k-images
	python3 ladderloss.py doRecall -R batch/f30k-sbert-res152ft/feat_best.pth; python3 ladderloss.py doTau -R batch/f30k-sbert-res152ft/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	F30K=1 python3 thCapRank.py Train -D cuda --tokspkl f30k.all.toks -L batch/f30k-akl-res152ft -S batch/f30k-vsepp-res152/split_info.json --finetune Resnet152 --cnndim 2048 --maxepoch 15 --snapshot batch/f30k-akl-res152/snapshot_latest.pth --cocopool /dev/shm/flickr30k-images -C config.aklf30k.py
	python3 ladderloss.py doRecall -R batch/f30k-akl-res152ft/feat_best.pth; python3 ladderloss.py doTau -R batch/f30k-akl-res152ft/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	F30K=1 python3 thCapRank.py Train -D cuda --tokspkl f30k.all.toks -L batch/f30k-akl-vgg19ft -S batch/f30k-vsepp-res152/split_info.json --finetune VGG19 --cnndim 4096 --maxepoch 15 --snapshot batch/f30k-akl-vgg19/snapshot_latest.pth --cocopool /dev/shm/flickr30k-images -C config.aklf30k.py
	python3 ladderloss.py doRecall -R batch/f30k-akl-vgg19ft/feat_best.pth; python3 ladderloss.py doTau -R batch/f30k-akl-vgg19ft/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	F30K=1 python3 thCapRank.py Train -D cuda --tokspkl f30k.all.toks -L batch/f30k-sbert-vgg19ft -S batch/f30k-vsepp-res152/split_info.json --finetune VGG19 --cnndim 4096 --maxepoch 15 --snapshot batch/f30k-sbert-vgg19/snapshot_latest.pth --cocopool /dev/shm/flickr30k-images -C config.sbertf30k.py
	python3 ladderloss.py doRecall -R batch/f30k-sbert-vgg19ft/feat_best.pth; python3 ladderloss.py doTau -R batch/f30k-sbert-vgg19ft/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	F30K=1 python3 thCapRank.py Train -D cuda --tokspkl f30k.all.toks -L batch/f30k-ladder-vgg19ft -S batch/f30k-vsepp-res152/split_info.json --finetune VGG19 --cnndim 4096 --maxepoch 15 --snapshot batch/f30k-ladder-vgg19/snapshot_latest.pth --cocopool /dev/shm/flickr30k-images -C config.ladderf30k.py
	python3 ladderloss.py doRecall -R batch/f30k-ladder-vgg19ft/feat_best.pth; python3 ladderloss.py doTau -R batch/f30k-ladder-vgg19ft/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
	#
	F30K=1 python3 thCapRank.py Train -D cuda --tokspkl f30k.all.toks -L batch/f30k-vsepp-vgg19ft -S batch/f30k-vsepp-res152/split_info.json --finetune VGG19 --cnndim 4096 --maxepoch 15 --snapshot batch/f30k-vsepp-vgg19/snapshot_latest.pth --cocopool /dev/shm/flickr30k-images
	python3 ladderloss.py doRecall -R batch/f30k-vsepp-vgg19ft/feat_best.pth; python3 ladderloss.py doTau -R batch/f30k-vsepp-vgg19ft/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K

batch-akl:
	python3 thCapRank.py Train -D cuda -L batch/coco-akl22-res152 -S batch/coco-vsepp-res152/split_info.json -C config.akl22.py
	python3 thCapRank.py Train -D cuda -L batch/coco-akl33-res152 -S batch/coco-vsepp-res152/split_info.json -C config.akl33.py
	python3 thCapRank.py Train -D cuda -L batch/coco-akl44-res152 -S batch/coco-vsepp-res152/split_info.json -C config.akl44.py
	python3 thCapRank.py Train -D cuda -L batch/coco-akl55-res152 -S batch/coco-vsepp-res152/split_info.json -C config.akl55.py
	python3 thCapRank.py Train -D cuda -L batch/coco-akl23-res152 -S batch/coco-vsepp-res152/split_info.json -C config.akl23.py
	python3 thCapRank.py Train -D cuda -L batch/coco-akl24-res152 -S batch/coco-vsepp-res152/split_info.json -C config.akl24.py
	python3 thCapRank.py Train -D cuda -L batch/coco-akl25-res152 -S batch/coco-vsepp-res152/split_info.json -C config.akl25.py
	#
	python3 ladderloss.py doRecall -R batch/coco-akl22-res152/feat_best.pth; python3 ladderloss.py doTau -R batch/coco-akl22-res152/feat_best.pth -S batch/coco-vsepp-res152/split_info.json -M sbert

batch-aklf30k:
	python3 thCapRank.py Train -D cuda -L batch/f30k-akl22-res152 -S batch/f30k-vsepp-res152/split_info.json --cnnpkl f30k.all.res152 --tokspkl f30k.all.toks -C config.akl22f.py
	python3 thCapRank.py Train -D cuda -L batch/f30k-akl33-res152 -S batch/f30k-vsepp-res152/split_info.json --cnnpkl f30k.all.res152 --tokspkl f30k.all.toks -C config.akl33f.py
	python3 thCapRank.py Train -D cuda -L batch/f30k-akl44-res152 -S batch/f30k-vsepp-res152/split_info.json --cnnpkl f30k.all.res152 --tokspkl f30k.all.toks -C config.akl44f.py
	python3 thCapRank.py Train -D cuda -L batch/f30k-akl55-res152 -S batch/f30k-vsepp-res152/split_info.json --cnnpkl f30k.all.res152 --tokspkl f30k.all.toks -C config.akl55f.py
	python3 thCapRank.py Train -D cuda -L batch/f30k-akl23-res152 -S batch/f30k-vsepp-res152/split_info.json --cnnpkl f30k.all.res152 --tokspkl f30k.all.toks -C config.akl23f.py
	python3 thCapRank.py Train -D cuda -L batch/f30k-akl24-res152 -S batch/f30k-vsepp-res152/split_info.json --cnnpkl f30k.all.res152 --tokspkl f30k.all.toks -C config.akl24f.py
	python3 thCapRank.py Train -D cuda -L batch/f30k-akl25-res152 -S batch/f30k-vsepp-res152/split_info.json --cnnpkl f30k.all.res152 --tokspkl f30k.all.toks -C config.akl25f.py
	#
	python3 ladderloss.py doRecall     -R batch/f30k-akl22-res152/feat_best.pth;  python3 ladderloss.py doTau        -R batch/f30k-akl22-res152/feat_best.pth -S batch/f30k-vsepp-res152/split_info.json -M sbert --dataset F30K
