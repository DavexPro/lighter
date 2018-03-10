import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from .utils.progbar_utils import Progbar
from torch.autograd import Variable


class Lighter:
    def __init__(self, model_ft, use_gpu=True, use_cudnn=False):
        self.model_ft = model_ft
        self.use_gpu = use_gpu
        self.use_cudnn = use_cudnn

    def fit(self, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=25, callbacks=[]):
        since = time.time()
        history = []

        best_model_wts = copy.deepcopy(self.model_ft.state_dict())
        best_loss = np.inf

        patient_epoch = 0
        train_size = len(train_loader.dataset)
        valid_size = len(valid_loader.dataset)
        print('Train on %d samples, validate on %d samples' % (train_size, valid_size))

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                tic = time.time()
                if phase == 'train':
                    data_size = train_size
                    data_loader = train_loader
                    scheduler.step()
                    self.model_ft.train(True)
                else:
                    data_size = valid_size
                    data_loader = valid_loader
                    self.model_ft.train(False)

                running_num = 0
                running_loss = 0.0
                running_corrects = 0
                bar = Progbar(len(data_loader), width=30, verbose=1, interval=0.05)

                # Iterate over data.
                for ix, data in enumerate(data_loader):
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    if self.use_gpu:
                        if phase == 'val':
                            inputs = Variable(inputs.cuda(), volatile=True)
                            labels = Variable(labels.cuda(), volatile=True)
                        elif phase == 'train':
                            inputs = Variable(inputs.cuda())
                            labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = self.model_ft(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_num += inputs.size(0)
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    cur_acc = float(running_corrects) / running_num
                    cur_loss = float(running_loss) / running_num

                    bar.update(ix + 1, values=[['loss', cur_loss], ['acc', cur_acc]])

                epoch_loss = running_loss / data_size
                epoch_acc = running_corrects / data_size
                toc = time.time()

                print(
                    '{} Time: {}s\tLoss: {:.4f}\tAcc: {:.4f}'.format(phase, int(toc - tic), epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val':
                    if epoch_loss < best_loss:
                        patient_epoch = 0
                        print('val_loss improved from {} to {}.'.format(best_loss, epoch_loss))
                        torch.save(self.model_ft, './models/densenet.{}.best.pth.tar'.format(cur_prefix))
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self.model_ft.state_dict())
                    else:
                        print('val_loss did not improve.')

            patient_epoch += 1
            if patient_epoch > 5:
                break

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

        # load best model weights
        self.model_ft.load_state_dict(best_model_wts)
        return history

    def predict(self):
        pass

    def evaluate(self):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass

    def load_weights(self):
        pass

    def save_weights(self):
        pass
