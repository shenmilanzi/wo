#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import numpy as np
import csv
import logging
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import roc_auc_score
# from sklearn.pipeline import Pipeline
import sys
reload(sys)
sys.setdefaultencoding('utf8')

logger = logging.getLogger('aucLog')
logger.setLevel(logging.INFO)

logfile = logging.FileHandler('auc.log')
logfile.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logfile.setFormatter(formatter)

logger.addHandler(logfile)


def get_field(raw_data):
    # 提取所需字段
    field = pd.DataFrame({
        '年龄段': raw_data['年龄段'],
        '手机品牌': raw_data['手机品牌']})
    return field


def get_result(predict_result):
    '''得到预测结果概率
    Params:
        predict_results: 原始预测结果
    Returns: 精确到小数点后四位的结果
    '''
    return round(predict_result[1], 7)


def rep_brand(dataframe):
    '''将手机品牌字段用对应数据的概率代替
    Params:
        dataframe: 原始数据, DataFrame
    Returns:
        dataframe: 手机品牌字段用对应数据的概率代替, DataFrame
    '''
    brand = dataframe['手机品牌']
    counts = brand.value_counts()
    proba = map(lambda x: float(counts[x]) / brand.count(), counts.keys())
    # freq 存储每个手机品牌的概率, dict
    freq = {key: proba[idx] for idx, key in enumerate(counts.keys())}

    # 用手机品牌概率替换手机品牌，方便分析
    dataframe['手机品牌'] = pd.DataFrame(
        map(lambda x: freq.get(x, 0), dataframe['手机品牌']))
    return dataframe


def compute_auc(predict_file, real_results):
    '''计算AUC评分, 等价于 sklearn.metrics.roc_auc_score
    Params:
        predict_file: 预测结果文件
        real_results: 真实结果
    Returns:
        auc: AUC评分
    '''
    # N: number of negative; M: number of positive
    N, M = real_results['是否去过迪士尼'].value_counts()
    yes = real_results[real_results['是否去过迪士尼'] == 1]
    no = real_results[real_results['是否去过迪士尼'] == 0]
    score = []
    pf = pd.read_csv(predict_file)
    for i in yes['用户标识']:
        p = pf[pf['IMEI'] == i]['SCORE']
        for j in no['用户标识']:
            n = pf[pf['IMEI'] == j]['SCORE']
            if p.values[0] > n.values[0]:
                score.append(1)
            elif p.values[0] == n.values[0]:
                score.append(0.5)
            else:
                score.append(0)
    auc = sum(score) / (M * N)
    return auc


if __name__ == '__main__':
    train_data = pd.read_csv('/data/第1题：算法题数据/数据集1_用户标签_本地_训练集.csv')
    y_train = pd.read_csv('/data/第1题：算法题数据/用户是否去过迪士尼_训练集.csv')
    test_data = pd.read_csv('/data/第1题：算法题数据/数据集1_用户标签_本地_测试集.csv')

    X_train, X_test = map(get_field, [train_data, test_data])

    # 使用手机品牌字段 counts = data['手机品牌'].value_counts()
    X_train, X_test = map(rep_brand, [X_train, X_test])
    logger.info('-----proba replace finished-----')

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train['是否去过迪士尼'])
    logger.info('-----model finished-----')

    results = classifier.predict_proba(X_test)
    predict_results = map(get_result, results)
    logger.info('-----predict finished-----')

    # 预测结果写入文件
    with open('result.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['IMEI', 'SCORE'])
        for idx in range(len(X_test)):
            writer.writerow(
                [test_data['用户标识'][idx], '%.6f' % predict_results[idx]])
    # 计算AUC评分
    # logger.info(str(compute_auc('result.csv', y_test)))
    # logger.info(str(roc_auc_score(y_test['是否去过迪士尼'], predict_results)))
