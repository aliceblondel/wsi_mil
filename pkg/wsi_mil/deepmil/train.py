from .arguments import get_arguments
from .dataloader import Dataset_handler
from .models import DeepMIL
import numpy as np
import torch
# For the sklearn warnings
import warnings
warnings.filterwarnings('always')

import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, class_names):
    """
    Plot the confusion matrix and return it as an image tensor for TensorBoard.
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.close(figure)

    return figure

def log_confusion_matrix(cm, writer, class_names, epoch):
    """
    Log a confusion matrix to TensorBoard.
    """
    figure = plot_confusion_matrix(cm, class_names)
    writer.add_figure("Confusion Matrix", figure, epoch)

def writes_metrics(writer, to_write, epoch, labels):
    """writes_metrics.
    Writes the validation metrics (and the train loss) in a Tensorboard Writer.

    :param writer: Tensorboard Writer
    :param to_write: dict, scalars to write.
    :param epoch: time step.
    """
    for key in to_write:
        if key=="conf_matrix":
            log_confusion_matrix(to_write[key], writer, labels, epoch)
        elif type(to_write[key]) == dict:
            writer.add_scalars(key, to_write[key], epoch)
        else:
            writer.add_scalar(key, to_write[key], epoch)

def train(model, dataloader):
    model.network.train()
    mean_loss = []
    epobatch = 1/len(dataloader) # How many epochs per batch ?
    for input_batch, target_batch in dataloader:
        model.counter["batch"] += 1
        model.counter['epoch'] += epobatch
        [scheduler.step(model.counter['epoch']) for scheduler in model.schedulers]
        loss = model.optimize_parameters(input_batch, target_batch)
        mean_loss.append(loss)
    model.mean_train_loss = np.mean(mean_loss)
    print('train_loss: {}'.format(np.mean(mean_loss)))

def val(model, dataloader, labels):
    model.network.eval()
    mean_loss = []
    for input_batch, target_batch in dataloader:
        target_batch = target_batch.to(model.device)
        loss = model.evaluate(input_batch, target_batch)
        mean_loss.append(loss)
    model.mean_val_loss = np.mean(mean_loss)
    to_write = model.flush_val_metrics()
    writes_metrics(model.writer, to_write, model.counter['epoch'], labels=labels) 
    state = model.make_state()
    print('mean val loss {}'.format(np.mean(mean_loss)))
    model.update_learning_rate(model.mean_val_loss)
    model.early_stopping(model.args.sgn_metric * to_write[model.args.ref_metric], state)

def main(raw_args=None):
    args = get_arguments(raw_args=raw_args, train=True)
    model = DeepMIL(args=args, with_data=True)
    model.get_summary_writer()
    while model.counter['epoch'] < args.epochs:
        print("Epochs {}".format(round(model.counter['epoch'])))
        train(model=model, dataloader=model.train_loader)
        if args.use_val:
            val(model=model, dataloader=model.val_loader, labels=args.class_names)
        if model.early_stopping.early_stop:
            break
        if not args.use_val:
            torch.save(model.make_state(), 'model_best.pt.tar')
    model.writer.close()

