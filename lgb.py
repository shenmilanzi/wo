#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import lightgbm as lgb
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import roc_auc_score
# from sklearn.pipeline import Pipeline
import sys
reload(sys)
sys.setdefaultencoding('utf8')

logger = logging.getLogger('LightGBMLog')
logger.setLevel(logging.INFO)

logfile = logging.FileHandler('lgb.log')
logfile.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logfile.setFormatter(formatter)

logger.addHandler(logfile)


def get_field(raw_data):
    # 提取所需字段
    field = pd.DataFrame({
        '年龄段': raw_data['年龄段'],
        '大致消费水平': raw_data['大致消费水平'],
        '每月的大致刷卡消费次数': raw_data['每月的大致刷卡消费次数'],
        '手机品牌': raw_data['手机品牌'],
        '用户更换手机频次': raw_data['用户更换手机频次'],
        '视频': raw_data['视频'],
        '音乐': raw_data['音乐'],
        '图片': raw_data['图片'],
        '体育': raw_data['体育'],
        '健康': raw_data['健康'],
        '动漫': raw_data['动漫'],
        '搜索': raw_data['搜索'],
        '生活': raw_data['生活'],
        '购物': raw_data['购物'],
        '房产': raw_data['房产'],
        '地图': raw_data['地图'],
        '餐饮': raw_data['餐饮'],
        '汽车': raw_data['汽车'],
        '旅游': raw_data['旅游'],
        '综合': raw_data['综合'],
        'IT': raw_data['IT'],
        '聊天': raw_data['聊天'],
        '交友': raw_data['交友'],
        '社交': raw_data['社交'],
        '通话': raw_data['通话'],
        '论坛': raw_data['论坛'],
        '问答': raw_data['问答'],
        '阅读': raw_data['阅读'],
        '新闻': raw_data['新闻'],
        '教育': raw_data['教育'],
        '孕期': raw_data['孕期'],
        '育儿': raw_data['育儿'],
        '金融': raw_data['金融'],
        '股票': raw_data['股票'],
        '游戏': raw_data['游戏'],
        '固定联络圈规模': raw_data['固定联络圈规模'],
        '访问视频网站的次数': raw_data['访问视频网站的次数'],
        '访问音乐网站的次数': raw_data['访问音乐网站的次数'],
        '访问图片网站的次数': raw_data['访问图片网站的次数'],
        '访问体育网站的次数': raw_data['访问体育网站的次数'],
        '访问健康网站的次数': raw_data['访问健康网站的次数'],
        '访问动漫网站的次数': raw_data['访问动漫网站的次数'],
        '访问搜索网站的次数': raw_data['访问搜索网站的次数'],
        '访问生活网站的次数': raw_data['访问生活网站的次数'],
        '访问购物网站的次数': raw_data['访问购物网站的次数'],
        '访问房产网站的次数': raw_data['访问房产网站的次数'],
        '访问地图网站的次数': raw_data['访问地图网站的次数'],
        '访问餐饮网站的次数': raw_data['访问餐饮网站的次数'],
        '访问汽车网站的次数': raw_data['访问汽车网站的次数'],
        '访问旅游网站的次数': raw_data['访问旅游网站的次数'],
        '访问综合网站的次数': raw_data['访问综合网站的次数'],
        '访问IT网站的次数': raw_data['访问IT网站的次数'],
        '访问聊天网站的次数': raw_data['访问聊天网站的次数'],
        '访问交友网站的次数': raw_data['访问交友网站的次数'],
        '访问社交网站的次数': raw_data['访问社交网站的次数'],
        '访问通话网站的次数': raw_data['访问通话网站的次数'],
        '访问论坛网站的次数': raw_data['访问论坛网站的次数'],
        '访问问答网站的次数': raw_data['访问问答网站的次数'],
        '访问阅读网站的次数': raw_data['访问阅读网站的次数'],
        '访问新闻网站的次数': raw_data['访问新闻网站的次数'],
        '访问教育网站的次数': raw_data['访问教育网站的次数'],
        '访问孕期网站的次数': raw_data['访问孕期网站的次数'],
        '访问育儿网站的次数': raw_data['访问育儿网站的次数'],
        '访问金融网站的次数': raw_data['访问金融网站的次数'],
        '访问股票网站的次数': raw_data['访问股票网站的次数'],
        '访问游戏网站的次数': raw_data['访问游戏网站的次数']})
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


def get_category(cate):
    '''处理从app.csv读入的类别
    Params:
        cate: 每一行，类型：string
    Returns:
        具体的app，类型：list
    '''
    category = cate.replace('\n', ',').split(',')
    return [i for i in category if i != '']


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
    train = pd.read_csv('/data/第1题：算法题数据/数据集1_用户标签_本地_训练集.csv')
    label = pd.read_csv('/data/第1题：算法题数据/数据集2_用户是否去过迪士尼_训练集.csv')
    test = pd.read_csv('/data/第1题：算法题数据/数据集1_用户标签_本地_测试集.csv')
    with open('app.csv', 'r') as f:
        category = f.readlines()

    category = map(get_category, category)

    # 分别统计train和test中300APP分类情况
    for cat in category:
        train[cat[0]] = reduce(pd.Series.add, map(lambda x: train[x], cat[1:]))
        test[cat[0]] = reduce(pd.Series.add, map(lambda x: train[x], cat[1:]))

    # 生成所需的训练集和测试集
    train, test = map(get_field, [train, test])
    label = label['是否去过迪士尼']
    ID = test['用户标识']

    # 使用手机品牌字段 counts = data['手机品牌'].value_counts()
    train, test = map(rep_brand, [train, test])
    logger.info('-----proba replace finished-----')

    lgb_train = lgb.Dataset(train, label)

    params = {
        'max_depth': 6,
        'learning_rate': 0.03,
        'objective': 'binary',
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'metric': 'auc'
    }

    # train
    bst = lgb.cv(params,
                 train_set=lgb_train,
                 num_boost_round=1000,
                 nfold=10,
                 verbose_eval=1)

    estimator = lgb.train(params,
                          train_set=lgb_train,
                          num_boost_round=len(bst['auc-mean']))
    estimator.save_model('lgb_model.txt')
    logger.info('-----model finished-----')

    # predict
    y_pred = lgb.predict(test)
    logger.info('-----predict finished-----')

    # 预测结果写入文件
    submission = pd.DataFrame({'IMEI': ID, 'SCORE': y_pred})
    submission.to_csv('result.csv', index=False)

    # 计算AUC评分
    # logger.info(str(compute_auc('result.csv', y_test)))
    # logger.info(str(roc_auc_score(y_train[840000:], y_pred)))
