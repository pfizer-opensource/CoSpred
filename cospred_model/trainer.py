"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import logging
import wandb
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from cospred_model import metrics as cospred_metrics

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    patience = 50
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    warmup_tokens = 375e6
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader
    max_charge = 6
    max_ce = 100

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.epoch = None
        self.epochs_with_no_improvement = 0
        self.current_best_checkpoint = None
        self.best_loss = 10000          # initiate the loss capture
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        # DataParallel wrappers keep raw model object in .module attribute
        # Delete previous checkpoint
        if self.current_best_checkpoint:
            os.remove(self.current_best_checkpoint)

        raw_model = self.model.module if hasattr(
            self.model, "module") else self.model
        current_best_checkpoint = os.path.join(self.config.ckpt_path,
                                               self.config.model_name +
                                               '_epoch' + str("{:03d}".format(self.epoch)) +
                                               '_loss' + str(round(self.best_loss, 5)) + '.pt')
        print('Saving checkpoint after epoch',
              self.epoch, current_best_checkpoint)
        self.current_best_checkpoint = current_best_checkpoint
        torch.save(raw_model.state_dict(), self.current_best_checkpoint)
        # raw_model = self.model.module if hasattr(self.model, "module") else self.model
        # logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            # is_train = True
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            dataloader = DataLoader(data, shuffle=True, pin_memory=True,
                                    batch_size=config.batch_size,
                                    # batch_size=3,
                                    num_workers=config.num_workers)
            losses = []

            # # DEBUG
            # tmp = next(iter(dataloader))
            # tmp.keys()
            # #

            pbar = tqdm(enumerate(dataloader), total=len(dataloader)
                        ) if is_train else enumerate(dataloader)
            # for it, (x, x_precursor, x_ce, y) in pbar:        # Fragile since depending on the column order
            # for it, (x, y) in pbar:
            for it, batch in pbar:
                # print(batch)
                # place data on the correct device
                x_tr = torch.cat((batch['sequence_integer'],
                                  batch['precursor_charge_onehot'],
                                  batch['collision_energy_aligned_normed']), dim=1)
                # x_tr = torch.cat((x_precursor[:,None], x), dim=1)
                # print("size of input", x_tr.shape)
                x_tr = x_tr.to(self.device)
                # x_precursor.to(self.device)
                y = batch['intensities_raw']
                y = y.to(self.device)
                # print("yshape")
                # print(y.shape)
                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x_tr, y)
                    # n_logits = logits.shape
                    # print("predicted output")
                    # print(n_logits)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    # print(loss)
                    losses.append(loss.item())
                # y_true= y.detach().cpu().squeeze(0).numpy()
                # y_pred= logits.detach().cpu().squeeze(0).numpy()
                y_true = y
                y_pred = logits
                metrics = cospred_metrics.ComputeMetrics(
                    true=y_true, pred=y_pred).return_metrics_mean()
                # print(metrics['recall'])

                if is_train:
                    for key, value in metrics.items():
                        wandb.log({'Train_' + key: value})

                if not is_train:
                    for key, value in metrics.items():
                        wandb.log({'Validation_' + key: value})
                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # # decay the learning rate based on our progress
                    # if config.lr_decay:
                    #     self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                    #     if self.tokens < config.warmup_tokens:
                    #         # linear warmup
                    #         lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    #     else:
                    #         # cosine learning rate decay
                    #         progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                    #         lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    #     lr = config.learning_rate * lr_mult
                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr
                    # else:
                    lr = config.learning_rate

                    # report progress
                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                    wandb.log({"Epoch": epoch, "Train Loss": loss.item()})
                    # , "Train Acc": acc_train, "Valid Loss": loss_valid, "Valid Acc": acc_valid})
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("validation loss: %f", test_loss)
                wandb.log({"validation loss": test_loss})
                return test_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            self.epoch = epoch

            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            improved = test_loss < best_loss
            if self.config.ckpt_path is not None and improved:
                self.epochs_with_no_improvement = 0
                best_loss = test_loss
                self.best_loss = best_loss
                self.save_checkpoint()
            elif self.test_dataset is not None and not improved:
                self.epochs_with_no_improvement += 1
            if self.epochs_with_no_improvement >= self.config.patience:
                print(
                    f'Performance did not improve after {self.epochs_with_no_improvement} epochs')
                break
