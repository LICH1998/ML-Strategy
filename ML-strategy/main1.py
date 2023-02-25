#!/usr/bin/env python 36
# -*- coding: utf-8 -*-
# @Time : 2023-02-24 16:38
# @Author : lichangheng
# 深度学习-时序预测策略

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy

import backtrader as bt
import pandas as pd
# import datetime
from datetime import datetime
import numpy as np
from transformer import *
import matplotlib


# 创建一个策略
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' 提供记录功能'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.order = None
        self.buyprice = None
        self.buycomm = None
        # 引用到输入数据的close价格
        self.dataclose = self.datas[0].close
        self.date = self.datas[0].datetime
        self.datavolume = self.datas[0].volume
        # 传入整个数据集用来训练
        self.pd_data = pd.read_csv('./data.csv', parse_dates=True)
        self.pd_data['date'] = self.pd_data['bob'].map(lambda x: datetime.strptime(str(x[:10]), "%Y-%m-%d").toordinal())
        # temp = self.pd_date.iloc[0]
        # self.pd_date = self.pd_date.apply(lambda i: i.shift(-1))
        # self.pd_date.iloc[-1] = temp
        # self.pd_data['date'] = self.pd_date.date.map(lambda x: round(x))

        # 训练Transformer时序预测模型
        model_train(self.pd_data['close'])
        # print(self.pd_data['date'])

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f ' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
                self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            self.order = None

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # 记录当前处理的close值
        self.log('Close, %.2f' % self.dataclose[0])
        # 得到当前部分数据起始数据对应的总数据中的index
        p_data_index = list(copy.deepcopy(self.pd_data['date'])).index(round(self.date[0]))
        p_data = self.pd_data[-49+p_data_index:p_data_index+1]['close']
        p1,p2,p3,p4,p5 = get_predict(self.pd_data['close'],p_data)
        p_l = [p1,p2,p3,p4,p5]
        high_cnt = [1 if i > p1 else 0 for i in p_l[1:]]
        low_cnt = [1 if i < p1 else 0 for i in p_l[1:]]

        # 订单是否
        if self.order:
            return
            # Check if we are in the market
        # 预测未来四天超过超过3天涨就买入
        if sum(high_cnt) > 3:
            self.log('BUY CREATE, %.2f' % self.dataclose[0])

            self.order = self.buy()
        # 预测未来四天超过超过3天跌就卖出
        if sum(low_cnt) > 3 and self.position:
            self.log('SELL CREATE, %.2f' % self.dataclose[0])

            self.order = self.sell()


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    # 增加一个策略
    cerebro.addstrategy(TestStrategy)
    # 获取数据
    stock_hfq_df = pd.read_csv("./data.csv", index_col='eob', parse_dates=True)
    start_date = datetime.strptime("2016-07-01", "%Y-%m-%d")  # 回测开始时间
    end_date = datetime.strptime("2017-07-01", "%Y-%m-%d")  # 回测结束时间
    data = bt.feeds.PandasData(dataname=stock_hfq_df, fromdate=start_date, todate=end_date)  # 加载数据
    cerebro.adddata(data)  # 将数据传入回测系统

    cerebro.broker.setcash(10000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=200)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='AnnualReturn')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.003, annualize=True, _name='SharpeRatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DrawDown')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.broker.setcommission(commission=0.001)
    results = cerebro.run()
    strat = results[0]
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print("--------------- AnnualReturn -----------------")
    print(strat.analyzers.AnnualReturn.get_analysis())
    print("--------------- SharpeRatio -----------------")
    print(strat.analyzers.SharpeRatio.get_analysis())
    print("--------------- DrawDown -----------------")
    print(strat.analyzers.DrawDown.get_analysis())

    cerebro.plot()