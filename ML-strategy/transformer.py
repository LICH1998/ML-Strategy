#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-02-24 16:01
# @Author : lichangheng


import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import math
from matplotlib import pyplot
from tqdm import *

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--input_window', type=int, default=50, help='input window')
parser.add_argument('--output_window', type=int, default=5, help='output window')
parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
parser.add_argument('--epoch', type=int, default=50, help='epoch')
parser.add_argument('--lr', type=int, default=0.05, help='learning rate')

args = parser.parse_args()
input_window = args.input_window
output_window = args.output_window
batch_size = args.batch_size

# 对时间序列进行预测，将时间序列变为1*n的输入向量即可。
calculate_loss_over_all_values = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=128, num_layers=1, dropout=0.2):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=2, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = np.append(input_data[i:i + tw][:-output_window], output_window * [0])
        train_label = input_data[i:i + tw]
        # train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


def train(train_data, epoch):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            # print('| epoch {:3d} | {:5d}/{:5d} batches | '
            #       'lr {:02.6f} | {:5.2f} ms | '
            #       'loss {:5.5f} | ppl {:8.2f}'.format(
            #     epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
            #                   elapsed * 1000 / log_interval,
            #     cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def predict_future(eval_model, data_source, steps):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    _, data = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps, 1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    pyplot.plot(data, color="red")
    pyplot.plot(data[:input_window], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-future%d.png' % steps)
    pyplot.close()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)


model = TransAm().to(device)

criterion = nn.MSELoss()
lr = args.lr
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = float("inf")
epochs = args.epoch  # The number of epochs
best_model = None
data_his_ = None


def model_train(data_his):
    time.sleep(.01)
    data_his = np.array(data_his)
    print('开始训练transformer模型...')
    data_his = scaler.fit_transform(data_his.reshape(-1, 1)).reshape(-1)
    train_data = data_his
    train_sequence = create_inout_sequences(train_data, input_window)

    for epoch in tqdm(range(1, epochs + 1)):
        epoch_start_time = time.time()
        train(train_sequence, epoch)
        scheduler.step()

    print('模型训练完成')


def get_predict(data_his,test_data):
    with torch.no_grad():
        test_data = list(test_data)
        input = torch.FloatTensor(np.append(test_data[-args.input_window+args.output_window:], [0] * args.output_window)).view(-1, 1,
                                                                                                     1).to(device)
        # print(input.shape)
        output = model(input).cpu().reshape(50, 1)
        data_his = np.array(data_his)
        scaler.fit_transform(data_his.reshape(-1, 1)).reshape(-1)
        output = scaler.inverse_transform(output)
        p1, p2, p3,p4, p5 = output[-5], output[-4], output[-3], output[-2], output[-1]
        return p1, p2, p3,p4,p5

# data = np.array([2]*220)
# model_train(data)
# test_data = np.array([2]*45)
# print(get_predict(data,test_data))

