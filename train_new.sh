#!/bin/bash

set -e

usage()
{
    echo
    echo "Usage: ./trash_new.sh <model_name>"
    echo
    echo "where <model_name> could be one of the following:"
    echo "    xception, mobilenet_v2, resnet50, googlenet_bn, inception_v2,"
    echo "    efficientnet_b0, efficientnet_b1, efficientnet_b4"
    echo
}

if [ $# -ne 1 ]; then
    usage
    exit
fi

case $1 in
    xception )
        python3 train.py --dropout_rate 0.5 --weight_decay 1e-5 \
                         --optimizer adam --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 0.045 --final_lr 1e-5 \
                         --epochs 200
        ;;
    """
    xception )
        python3 train.py --dropout_rate 0.5 --weight_decay 1e-5 \
                         --optimizer adam --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 0.045 --final_lr 1e-5 \
                         --epochs 60
        ;;
    """
    mobilenet_v2 )
        python3 train.py --dropout_rate 0.2 --weight_decay 1e-6 \
                         --optimizer adam --batch_size 64 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 60 mobilenet_v2
        ;;
    resnet50 )
        python3 train.py --dropout_rate 0.5 --weight_decay 2e-4 \
                         --optimizer adam --batch_size 16 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 80 resnet50
        ;;
    googlenet_bn )
        python3 train.py --dropout_rate 0.4 --weight_decay 2e-4 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 60 googlenet_bn
        ;;
    inception_v2 )
        python3 train.py --dropout_rate 0.4 --weight_decay 2e-4 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 80 inception_v2
        ;;
    efficientnet_b0 )
        python3 train.py --dropout_rate 0.2 --weight_decay 1e-4 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-4 \
                         --epochs 60 efficientnet_b0
        ;;
    efficientnet_b1 )
        python3 train.py --dropout_rate 0.2 --weight_decay 1e-4 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-4 \
                         --epochs 60 efficientnet_b1
        ;;
    efficientnet_b4 )
        python3 train.py --dropout_rate 0.2 --weight_decay 1e-4 \
                         --optimizer adam --batch_size 16 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-4 \
                         --epochs 80 efficientnet_b4
        ;;
    osnet )
        python3 train.py --dropout_rate 0.2 \
                         --optimizer adam --batch_size 32 --iter_size 1 \
                         --lr_sched exp --initial_lr 1e-2 --final_lr 1e-5 \
                         --epochs 60 osnet
        ;;
    * )
        usage
        exit
esac
