import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loss import MultiBoxLoss
from .network import build_dssd, build_ssd
from .training_data import DefaultBox


class TrainModel:
    def __init__(self, train_dl, val_dl, cfg_path, hyp_path, model="dssd"):
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        with open(hyp_path, 'r') as f:
            hyp = yaml.safe_load(f)

        if model=="dssd":
            self.model = build_dssd(cfg=cfg["ssd"])
        else:
            self.model = build_ssd(cfg=cfg["ssd"])

        self.seed = cfg["seed"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self.train_dl = train_dl
        self.val_dl = val_dl

        self.dbox = DefaultBox(cfg=cfg["dbox"], device=self.device)
        self.loss_layer = MultiBoxLoss(neg_pos=hyp["loss"]["neg_pos"], device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyp["lr"])
        self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer, patience=hyp["scheduler"]["patience"],
                                           verbose=True, factor=hyp["scheduler"]["factor"])

        self.epochs = hyp["epochs"]
        self.es_patience = hyp["es_patience"]

        self.loss_log = {
            "train_loss": [], "val_loss": [],
            "train_loss_loc": [], "train_loss_conf": [],
            "val_loss_loc": [], "val_loss_conf": []
        }

    def seed_everything(self, seed):
        # 乱数のシードを設定
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = True

    def train(self):
        print("使用デバイス: ", self.device)

        self.model.to(self.device)

        best_loss = np.inf
        patience = self.es_patience
        self.seed_everything(self.seed)

        print("==========start==========")
        for epoch in range(self.epochs+1):
            start_time = time.time()

            # train
            train_loss, train_loc, train_conf = self._epoch_loop(
                dl=self.train_dl,
                train_flg=True
            )

            # eval
            val_loss, val_loc, val_conf = self._epoch_loop(
                dl=self.val_dl,
                train_flg=False
            )

            finish_time = time.time()

            # ログの追加
            self.loss_log["train_loss"].append(train_loss)
            self.loss_log["val_loss"].append(val_loss)
            self.loss_log["train_loss_loc"].append(train_loc)
            self.loss_log["val_loss_loc"].append(val_loc)
            self.loss_log["train_loss_conf"].append(train_conf)
            self.loss_log["val_loss_conf"].append(val_conf)

            print("Epochs: {:03} | Train Loss : {:.5f} | Val Loss: {:.5f} | Time: {:.3f}"
                  .format(epoch, train_loss, val_loss, finish_time - start_time))

            self.scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_train_loss = train_loss
                best_epoch = epoch
                patience = self.es_patience
                torch.save(self.model.state_dict(), './ssd-weight.pth')
            else:
                patience -= 1

            if patience == 0:
                print("Early Stopping | Epochs: {:03} | Train Loss: {:.5f} | Best Loss: {:.5f}"
                      .format(best_epoch, best_train_loss, best_loss))
                break

    def _epoch_loop(self, dl, train_flg):
        epoch_loss = 0.0
        epoch_loss_l = 0.0
        epoch_loss_c = 0.0

        if train_flg:
            self.model.train()
        else:
            self.model.eval()

        for images, targets in dl:
            images = images.to(self.device)
            targets = [ann.to(self.device) for ann in targets]

            self.optimizer.zero_grad()

            # 順伝播
            loc, conf = self.model.forward(images)

            # 教師データの作成
            loc_t, conf_t = self.dbox.get_targets(targets)

            # lossの計算
            loss_l, loss_c = self.loss_layer.forward(loc, conf, loc_t, conf_t)
            loss = loss_l + loss_c

            if train_flg:
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_l += loss_l.item()
            epoch_loss_c += loss_c.item()

        epoch_loss /= len(dl)
        epoch_loss_l /= len(dl)
        epoch_loss_c /= len(dl)

        return epoch_loss, epoch_loss_l, epoch_loss_c

    def plot_logs(self):
        epochs = range(len(self.loss_log["train_loss"]))
        plot_list = ["loss", "loss_loc", "loss_conf"]
        plot_title = ["Network", "Loc", "Conf"]

        for i in range(len(plot_list)):
            plt.figure()
            plt.plot(epochs, self.loss_log["train_{}".format(plot_list[i])], label='training {}'.format(plot_list[i]))
            plt.plot(epochs, self.loss_log["val_{}".format(plot_list[i])], label='valid {}'.format(plot_list[i]))
            plt.title("{} Loss".format(plot_title[i]))
            plt.legend()

        plt.show()
