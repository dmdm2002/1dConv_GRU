import os

import pandas as pd
import numpy as np

import torch
import torch.utils.data as data


class Loader(data.DataLoader):
    def __init__(self, root, snp_le=None, cls=None, training=True, valid=False):
        self.root = root
        self.snp_le = snp_le
        self.cls = cls
        self.training = training

        if training:
            if valid:
                df = pd.read_csv(f'{self.root}/split/valid.csv')
                self.x, self.y = self.get_x_y(df)
            else:
                df = pd.read_csv(f'{self.root}/split/train.csv')
                self.x, self.y = self.get_x_y(df)
        else:
            df = pd.read_csv(f'{self.root}/test.csv')
            self.x, self.name = self.get_x_y(df)

    def get_x_y(self, df):
        if self.training:
            x = df.drop(columns=['id', 'class', 'father', 'mother', 'gender'])
            y = df['class']

            x = np.array(self.snp_encoding(x))
            y = np.array(self.class_encoding(y))

            return x, y

        else:
            x = df.drop(columns=['id', 'father', 'mother', 'gender'])
            name = df['id']
            x = np.array(self.snp_encoding(x))
            name = np.array(name)

            return x, name

    def class_encoding(self, df):
        return df.replace(self.cls)

    def snp_encoding(self, df):
        snp_col = [f'SNP_{str(x).zfill(2)}' for x in range(1, 16)]
        snp_data = []
        for col in snp_col:
            snp_data += list(df[col].values)

        self.snp_le.fit(snp_data)
        # snp_le을 이용해서 labeling 해준다
        for col in df.columns:
            if col in snp_col:
                df[col] = self.snp_le.transform(df[col])

        return df

    def __getitem__(self, index):
        if self.training:
          x = self.x[index]
          y = self.y[index]

          return [x, y]

        else:
          x = self.x[index]
          name = self.name[index]

          return [x, name]

    def __len__(self):
        return len(self.x)