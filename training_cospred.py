from prosit_model import metrics as prosit_metrics
from prosit_model import losses
from cospred_model.model.transformerEncoder import TransformerConfig, TransformerEncoder
from cospred_model.trainer import Trainer, TrainerConfig
import model as model_lib
import io_cospred
import params.constants_location as constants_location
import params.constants as constants
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb
from contextlib import redirect_stdout
import numpy as np
from matplotlib import pyplot as plt
import os
from argparse import ArgumentParser
import time
import keras
from keras.utils import plot_model
import matplotlib
matplotlib.use('Agg')


def train_transformer(ds_train, ds_val, flag_fullspectrum, model, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("loaded data")
    print("size of train dataset", len(ds_train))
    print("size of val dataset", len(ds_val))

    start_time = int(time.time())       # start time for the training
    # create log folder
    folder_time_format = time.strftime('%Y%m%d_%H%M%S')
    log_time_format = time.strftime('%Y-%m-%d_%H:%M:%S')

    if flag_fullspectrum is True:
        model_name = 'transformer_full_' + folder_time_format
        n_output = constants.SPECTRA_DIMENSION
    else:
        model_name = 'transformer_byion_' + folder_time_format
        n_output = 174

    if model is None:
        mconf = TransformerConfig(vocab_size=constants.MAX_ALPHABETSIZE, block_size=37,
                                  embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,
                                  n_layer=8, n_head=16, n_embd=256,
                                  n_output=n_output,
                                  max_charge=10, max_ce=100)
        model = TransformerEncoder(mconf)

    print(sum(p.numel() for p in model.parameters()
          if p.requires_grad), 'model parameters')
    print(model)

    checkpoint_dir = model_dir + model_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    tconf = TrainerConfig(max_epochs=constants.TRAIN_EPOCHS, batch_size=1024, learning_rate=0.001,
                          lr_decay=True, warmup_tokens=512 * 20,
                          final_tokens=2 * len(ds_train) * 250,
                          num_workers=4,
                          ckpt_path=checkpoint_dir,
                          patience=50,
                          model_name=model_name)
    mode = 'offline'
    wandb.init(project=f'CoSpred',
               name=model_name,
               reinit=True, mode=mode)

    trainer = Trainer(model, ds_train, ds_val, config=tconf)
    trainer.train()

    # record elapsed time for the training
    eplased_time = int(time.time() - start_time)
    f = open(model_dir + "exetime.txt", "a")
    f.write('{}'.format(model_name) + "\t" +
            log_time_format + '\t' + str(eplased_time) + '\n')
    f.close()

# functions for prosit train


class TrainingPlot(keras.callbacks.Callback):
    def __init__(self, result_dir):
        self.result_dir = result_dir

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('cosine_similarity'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_cosine_similarity'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            N = np.arange(0, len(self.losses))

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.acc, label="train_cosine_similarity")
            plt.plot(N, self.val_losses, label="val_loss")
            plt.plot(N, self.val_acc, label="val_cosine_similarity")
            plt.title(
                "Training Loss and CosineSimilarity [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Cosine_similarity")
            plt.legend()
            plt.savefig('{}/Epoch-{}.png'.format(self.result_dir, epoch))
            plt.close()


def get_callbacks(model_dir_path, result_dir, model_name):
    import keras

    loss_format = "{val_loss:.5f}"
    epoch_format = "{epoch:03d}"
    weights_file = "{}/{}_epoch{}_loss{}.hdf5".format(
        model_dir_path, model_name, epoch_format, loss_format
    )
    csvlog_file = "{}/training.log".format(result_dir)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='{}/tensorboardlogs'.format(result_dir), histogram_freq=1)
    save = keras.callbacks.ModelCheckpoint(weights_file, save_best_only=True)
    stop = keras.callbacks.EarlyStopping(patience=10)
    decay = keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.2)
    csv_logger = keras.callbacks.CSVLogger(csvlog_file, append=False)
    plot_losses = TrainingPlot(result_dir)
    wandb_metric_logger = WandbMetricsLogger(log_freq=5)
    wandb_model_ckpt = WandbModelCheckpoint("models")
    return [save, stop, decay, csv_logger, plot_losses,
            tensorboard, wandb_metric_logger, wandb_model_ckpt]


def trainer_prosit(tf_ds_train, tf_ds_val, model, model_config, callbacks):
    if isinstance(model_config["loss"], list):
        loss = [losses.get(loss_name) for loss_name in model_config["loss"]]
    else:
        loss = losses.get(model_config["loss"])
    optimizer = model_config["optimizer"]

    # Define metrics
    metrics_to_compute = [
        prosit_metrics.ComputeMetrics()
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics_to_compute)

    history = model.fit(
        tf_ds_train,
        epochs=constants.TRAIN_EPOCHS,
        validation_data=tf_ds_val,
        shuffle=True,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True,
    )

    keras.backend.get_session().close()
    return (history)


def train_prosit(tf_ds_train, tf_ds_val, flag_fullspectrum, model, model_config, model_dir):
    start_time = int(time.time())       # start time for the training
    # create log folder
    folder_time_format = time.strftime('%Y%m%d_%H%M%S')
    log_time_format = time.strftime('%Y-%m-%d_%H:%M:%S')
    if flag_fullspectrum is True:
        model_name = 'prosit_full_' + folder_time_format
    else:
        model_name = 'prosit_byion_' + folder_time_format

    checkpoint_dir = model_dir + model_name

    result_dir = checkpoint_dir+'/log_{}'.format(folder_time_format)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    weight_dir = checkpoint_dir+'/weight_{}'.format(folder_time_format)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # visualize model architecture
    plot_model(model, to_file='{}/model.png'.format(result_dir),
               show_shapes=True)
    with open('{}/modelsummary.txt'.format(result_dir), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # wandb to capture metrices
    mode = 'offline'
    wandb.init(project=f'CoSpred', name=model_name, reinit=True, mode=mode)

    # training loop
    callbacks = get_callbacks(weight_dir, result_dir, model_name)
    history = trainer_prosit(tf_ds_train, tf_ds_val,
                             model, model_config, callbacks)
    history.history

    # record elapsed time for the training
    eplased_time = int(time.time() - start_time)
    f = open(model_dir + "exetime.txt", "a")
    f.write('log_{}'.format(folder_time_format) + "\t" +
            log_time_format + '\t' + str(eplased_time) + '\n')
    f.close()

    # record training matrices
    record_prosit(history, result_dir)


def record_prosit(history, result_dir):
    # plot
    fig = plt.figure()
    plt.plot(history.history['cosine_similarity'])
    plt.plot(history.history['val_cosine_similarity'])
    plt.title('Model Performance')
    plt.ylabel('cosine_similarity')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig.savefig('{}/cosine_similarity.png'.format(result_dir))

    # Plot training & validation loss values
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig.savefig('{}/loss.png'.format(result_dir))


def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--trained', default=False, action='store_true',
                        help='turn on loading best existing model')
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help='full spectrum presentation')
    parser.add_argument('-c', '--chunk', default=False, action='store_true',
                        help='train model in chunk')
    parser.add_argument('-p', '--prosit', default=False, action='store_true',
                        help='train with prosit model')
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # turn off tf logging

    # input file choices
    if args.chunk is True:
        data_path = constants_location.TRAINDATASET_PATH
    else:
        data_path = constants_location.TRAINDATA_PATH
    model_dir = constants_location.MODEL_DIR
    print(data_path)

    # load dataset
    dataset = io_cospred.genDataset(data_path, args.chunk)

    # load model
    model, model_config, weights_path = model_lib.load(
        model_dir, args.full, args.prosit, args.trained)

    # training
    if args.prosit is True:
        ds_train, ds_val = io_cospred.train_val_split(
            dataset, model_config, tensorformat='tf')
        train_prosit(ds_train, ds_val, args.full,
                     model, model_config, model_dir)
    else:
        ds_train, ds_val = io_cospred.train_val_split(
            dataset, model_config, tensorformat='torch')
        train_transformer(ds_train, ds_val, args.full, model, model_dir)


if __name__ == "__main__":
    main()
