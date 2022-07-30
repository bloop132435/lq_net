if [ -e .env ]; then
  source .env
  if [ "$3" == "load_data" ]; then
    copy_imagenet_to_ddr
  fi
fi

if [ "$FASTDIR" == "" ]; then
  FASTDIR=/workspace
fi

if [ -d $FASTDIR/git/aim-uofa-model-quantization ]; then
  cd $FASTDIR/git/aim-uofa-model-quantization
elif [ -d /workspace/git/aim-uofa-model-quantization ]; then
  cd /workspace/git/aim-uofa-model-quantization
else
  FASTDIR=../..
  cd .
fi


# script=main.py
# train_batch=20
# val_batch=20
# dataset='imagenet'
# root=$FASTDIR/data/imagenet
# case='fake'
# keybase=''
# keyword=','
# model='unknow'
# base=1
# epochs=0
# options=''
# pretrain='none'

# config=config.bin
# if [ "$1" != "" ]; then config=$1; fi
# if [ -e $config ];
# then
  # echo "Loading config from $config"
  # source $config
# fi

# if [ "$DELAY" != "" ]; then
  # delay=$DELAY
# else
  # delay=0
# fi

# if [ "$2" != "" ]; then script=$2; fi

options=''


python main.py --dataset cifar10 --root $FASTDIR/data/cifar10 \
  --model resnet20 --base 1 \
  --epochs 1000 -b 100 -v 100 \
  --case official --keyword cifar10,bacs,lq \
  --delay 0 \
  --fm_bit 8 --wt_bit 7 --fm_enable --wt_enable\
  --workers 15 --save_freq 1\
  --bits 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3

# previous configs
# 1 --bits 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 8 DNF
# 2 --bits 5 4 4 4 3 4 3 4 3 3 3 3 3 3 3 3 8 (resnet18, 50 epochs) = 87.77%
