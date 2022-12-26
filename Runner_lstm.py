import os
import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn import preprocessing

from utils.Options import param
from utils.custom_loader import Loader
from Modeling.CNN_LSTM.model import Conv1d_RNN


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Runner(param):
    def __init__(self):
        super(Runner, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        os.makedirs(self.OUTPUT_LOSS, exist_ok=True)
        os.makedirs(self.OUTPUT_CKP, exist_ok=True)
        # self.device='cpu'

    def _build_model(self):
        model = Conv1d_RNN(in_channel=16, out_channel=3)

        return model

    def train(self):
        print('-------- Traning Start --------')
        print(f'[DEVICE] : {self.device}')

        model = self._build_model().to(self.device)

        model_optim = optim.Adam(model.parameters(), lr=self.LR)
        criterion = nn.MSELoss()

        snp_le = preprocessing.LabelEncoder()
        tr_dataset = Loader(self.ROOT, snp_le, self.classes, training=True, valid=False)
        valid_dataset = Loader(self.ROOT, snp_le, self.classes, training=True, valid=True)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=self.BATCHSZ, shuffle=False)

        summary = SummaryWriter(self.OUTPUT_LOSS)
        score_list = []

        for ep in range(self.EPOCH):
            model.train()

            tr_dataloader = DataLoader(dataset=tr_dataset, batch_size=self.BATCHSZ, shuffle=True)
            loss_list = []
            acc = [0, 0]

            for idx, (x, y) in enumerate(tqdm.tqdm(tr_dataloader, desc=f'Epcoh : [{ep}/{self.EPOCH}]')):
                x = x.float().to(self.device)
                y = y.float().to(self.device)

                pred = model(x)
                loss = criterion(pred, y)

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                loss_list.append(loss.item())
                acc[0] += (pred.argmax() == y.detach()).type(torch.float).sum().item()

            model.eval()
            with torch.no_grad():
                for idx, (x, y) in enumerate(tqdm.tqdm(valid_dataloader, desc=f'Epcoh : [{ep}/{self.EPOCH}]')):
                    x = x.float().to(self.device)
                    y = y.float().to(self.device)

                    pred = model(x)
                    acc[1] += (pred.argmax() == y.detach()).type(torch.float).sum().item()

            loss_avg = sum(loss_list) / len(loss_list)
            tr_sum_acc = acc[0] / len(tr_dataloader)
            valid_sum_acc = acc[1] / len(valid_dataset)

            print(f"===> EPOCH[{ep}/{self.EPOCH}] || train acc : {tr_sum_acc}   |   train loss : {loss_avg}   |   validation acc : {valid_sum_acc}")

            summary.add_scalar("train/acc", tr_sum_acc, ep)
            summary.add_scalar("train/loss", loss_avg, ep)
            summary.add_scalar("validation/acc", valid_sum_acc, ep)

            score_list.append([ep, tr_sum_acc, loss_avg, valid_sum_acc])

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": ep,
                },
                os.path.join(f"{self.OUTPUT_CKP}", f"{ep}.pth"),
            )

        df = pd.DataFrame(columns=['epoch', 'tr_acc', 'tr_loss', 'valid_acc'], data=score_list)
        df.to_csv(f'{self.OUTPUT}/metric.csv', index=False)

    def test(self):
        print('-------- Traning Start --------')
        print(f'[DEVICE] : {self.device}')

        model = self._build_model().to(self.device)

        snp_le = preprocessing.LabelEncoder()

        te_dataset = Loader(self.ROOT, snp_le, self.classes, training=False, valid=False)
        te_dataloader = DataLoader(dataset=te_dataset, batch_size=self.BATCHSZ, shuffle=False)

        pred_list = []

        for ep in range(86, 87):
            checkpoint = torch.load(f'{self.OUTPUT_CKP}/{ep}.pth', map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])

            model.eval()

            for idx, (x, name) in enumerate(te_dataloader):
                x = x.float().to(self.device)
                pred = model(x)
                pred_cls = pred.detach().argmax()

                pred_list.append([name[0], pred_cls.item()])

            df = pd.DataFrame(columns=['id', 'class'], data=pred_list)
            reverse_cls = {0: 'A', 1: 'B', 2: 'C'}
            df = df.replace(reverse_cls)
            df.to_csv(f'{self.OUTPUT}/submission123.csv', index=False)


a = Runner()
a.test()