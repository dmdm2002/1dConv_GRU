import torch
import torch.nn as nn


class Conv1d_RNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=3):
        super(Conv1d_RNN, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=in_channel,
                                  out_channels=in_channel*2,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  )
        # self.norm_1 = nn.BatchNorm1d(in_channel*2)
        self.activation_1 = nn.ReLU()
        # self.avgpool_1 = nn.AvgPool1d(2)

        self.conv1d_2 = nn.Conv1d(in_channels=in_channel*2,
                                  out_channels=in_channel*4,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  )
        # self.norm_2 = nn.BatchNorm1d(in_channel*4)
        self.activation_2 = nn.ReLU()
        # self.avgpool_2 = nn.AvgPool1d(2)

        self.gru = nn.GRU(input_size=in_channel*4,
                          hidden_size=in_channel*8,
                          num_layers=1,
                          bias=True,
                          bidirectional=False,
                          batch_first=True,
                          )

        self.dropout = nn.Dropout(0.01)

        self.dense1 = nn.Linear(in_channel*8, in_channel*4)
        self.dense2 = nn.Linear(in_channel*4, out_channel)

    def forward(self, x):
        x = x.permute(1, 0)
        # print(x.shape)
        x = self.conv1d_1(x)
        # x = self.norm_1(x)
        x = self.activation_1(x)

        x = self.conv1d_2(x)
        # x = self.norm_2(x)
        x = self.activation_2(x)

        x = x.permute(1, 0)
        self.gru.flatten_parameters()

        output, hidden = self.gru(x)
        x = hidden[-1]

        x = self.dropout(x)

        x = self.dense1(x)
        x = self.dense2(x)

        return x