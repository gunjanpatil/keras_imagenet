"""train.sh

This script is used to train the ImageNet models.
"""


import os
import time
import argparse
import datetime

import tensorflow as tf

from config import config
from utils.utils import config_keras_backend, clear_keras_session
from utils.dataset import get_dataset
from models.models import get_batch_size
from models.models import get_iter_size
from models.models import get_lr_func
from models.models import get_initial_lr
from models.models import get_final_lr
from models.models import get_weight_decay
from models.models import get_optimizer
from models.models import get_training_model


DESCRIPTION = """For example:
$ python3 train.py --dataset_dir  ${HOME}/data/ILSVRC2012/tfrecords \
                   --dropout_rate 0.4 \
                   --optimizer    adam \
                   --batch_size   32 \
                   --iter_size    1 \
                   --lr_sched     exp \
                   --initial_lr   1e-2 \
                   --final_lr     1e-5 \
                   --weight_decay 2e-4 \
                   --epochs       60 \
                   googlenet_bn
"""
SUPPORTED_MODELS = (
    '"mobilenet_v2", "resnet50", "googlenet_bn", "inception_v2", '
    '"efficientnet_b0", "efficientnet_b1", "efficientnet_b4", "xception"'
    '"osnet" or just specify a saved Keras model (.h5) file')


def train(main_dir, model_name, dropout_rate, optim_name,
          use_lookahead, batch_size, iter_size,
          lr_sched, initial_lr, final_lr,
          weight_decay, epochs, dataset_dir, log_dir,
          nb_train_samples, nb_val_samples,
          data_agumentation, *, model=None, model_save_dir=None):
    """Prepare data and train the model."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    batch_size   = get_batch_size(model_name, batch_size)
    iter_size    = get_iter_size(model_name, iter_size)
    initial_lr   = get_initial_lr(model_name, initial_lr)
    final_lr     = get_final_lr(model_name, final_lr)
    optimizer    = get_optimizer(model_name, optim_name, initial_lr)
    weight_decay = get_weight_decay(model_name, weight_decay)

    # get training and validation data
    ds_train = get_dataset(dataset_dir, 'train', batch_size, data_agumentation=data_agumentation)
    ds_valid = get_dataset(dataset_dir, 'validation', batch_size, data_agumentation=data_agumentation)

    # instantiate training callbacks
    lrate = get_lr_func(epochs, lr_sched, initial_lr, final_lr)

    #save_name = model_name if not model_name.endswith('.h5') else \
    #            os.path.split(model_name)[-1].split('.')[0].split('-')[0]
    tfrecord_folder = dataset_dir.split('/')[-1]
    if model:
        initial_epoch = model.split('/')[-1].split('-')[-1].split('.')[0]
        print("[INFO] initial_epoch: ",initial_epoch)
    else:
        initial_epoch = 1
    if model_save_dir is not None:
        print("model will be loaded from: ", model)
        assert('/'.join(model.split('/')[:-1])==model_save_dir), "model to be loaded is not in model_save_dir"
        _model_save_dir = model_save_dir
    else:
        print("creating model saving directory: ",tfrecord_folder+"_"+model_name+"_"+timestamp)
        _model_save_dir = os.path.join(main_dir, "models",\
                          tfrecord_folder+"_"+model_name+"_"+timestamp)
        os.makedirs(_model_save_dir,exist_ok=True)
    print("[INFO]model save directory: ",_model_save_dir)
    save_name = tfrecord_folder+"_"+model_name+"_"+timestamp
    _log_dir = os.path.join(main_dir,"logs",tfrecord_folder,"logs_"+save_name)
    print("[INFO]log directory: ",_log_dir)
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(_model_save_dir, save_name) + '_ckpt-{epoch}.h5',
        monitor='val_loss')
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir= _log_dir)

    # build model and do training
    model = get_training_model(
        model_name=model_name,
        dropout_rate=dropout_rate,
        optimizer=optimizer,
        use_lookahead=use_lookahead,
        iter_size=iter_size,
        weight_decay=weight_decay,
        model=model)
    model.fit(
        x=ds_train,
        steps_per_epoch =  nb_train_samples // batch_size,
        validation_data=ds_valid,
        validation_steps= nb_val_samples // batch_size,
        callbacks=[lrate, model_ckpt, tensorboard],
        # The following doesn't seem to help in terms of speed.
        use_multiprocessing=True, workers=8,
        epochs=epochs)

    # training finished
    print("Saving final model to ",_model_save_dir)
    model.save('{}/{}-model-final.h5'.format(_model_save_dir, save_name))


def main():

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--main_dir',type=str, help="path to main directory consisting of tfrecords,models,logs,etc")
    parser.add_argument('--dataset_dir', type=str, help="path to tfrecords folder containing train and val tfrecords")
    parser.add_argument('--model_save_dir', help="directory to save intermediate models", type=str)
    parser.add_argument('--log_dir', help="directory to save tensorboard logs", type=str)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--use_lookahead', action='store_true')
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--iter_size', type=int, default=-1)
    parser.add_argument('--lr_sched', type=str, default='linear',
                        choices=['linear', 'exp'])
    parser.add_argument('--initial_lr', type=float, default=-1.)
    parser.add_argument('--final_lr', type=float, default=-1.)
    parser.add_argument('--weight_decay', type=float, default=-1.)
    parser.add_argument('--epochs', type=int, default=1,
                        help='total number of epochs for training [1]')
    parser.add_argument('--nb_train_samples',type=int,help="number of training samples")
    parser.add_argument('--nb_val_samples',type=int,help="number of validation samples")
    parser.add_argument('--data-agumentation',help="whether to augment data or not",\
                        action="store_true")
    parser.add_argument('--model', type=str,help="path to model h5 file to be loaded")
    parser.add_argument('--model-name', type=str,help=SUPPORTED_MODELS, default="xception")
    args = parser.parse_args()

    if args.use_lookahead and args.iter_size > 1:
        raise ValueError('cannot set both use_lookahead and iter_size')

        config_keras_backend()
    train(args.main_dir,args.model_name, args.dropout_rate, args.optimizer,
          args.use_lookahead, args.batch_size, args.iter_size,
          args.lr_sched, args.initial_lr, args.final_lr,
          args.weight_decay, args.epochs, args.dataset_dir,
          args.log_dir, args.nb_train_samples, args.nb_val_samples,
          args.data_agumentation,
          model=args.model, model_save_dir=args.model_save_dir)
    clear_keras_session()


if __name__ == '__main__':
    main()
