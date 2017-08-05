#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import numpy as np
import csv
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
# from sklearn.grid_search import GridSearchCV
# from sklearn.pipeline import Pipeline
import sys
reload(sys)
sys.setdefaultencoding('utf8')


def get_field(raw_data):
    # 提取所需字段
    field = pd.DataFrame({
        '年龄段': raw_data['年龄段'],
        '大致消费水平': raw_data['大致消费水平'],
        '每月的大致刷卡消费次数': raw_data['每月的大致刷卡消费次数'],
        '爱奇艺': raw_data['爱奇艺'],
        '腾讯视频': raw_data['腾讯视频'],
        '唱吧': raw_data['唱吧'],
        '女孩相机': raw_data['女孩相机'],
        '什么值得买': raw_data['什么值得买'],
        '星巴克中国': raw_data['星巴克中国'],
        '亚马逊': raw_data['亚马逊'],
        '蘑菇街': raw_data['蘑菇街'],
        '天猫': raw_data['天猫'],
        '手机天猫': raw_data['手机天猫'],
        '苹果地图': raw_data['苹果地图'],
        '嘀嘀打车': raw_data['嘀嘀打车'],
        '高铁管家': raw_data['高铁管家'],
        '航旅纵横': raw_data['航旅纵横'],
        '虎扑看球': raw_data['虎扑看球'],
        '新浪微博': raw_data['新浪微博'],
        '雪球': raw_data['雪球'],
        '百度魔图': raw_data['百度魔图'],
        '天天P图': raw_data['天天P图'],
        'Safari': raw_data['Safari'],
        'HUAWEI Browser': raw_data['HUAWEI Browser'],
        'XiaoMi Browser': raw_data['XiaoMi Browser'],
        'OPPO Browser': raw_data['OPPO Browser'],
        'SAMSUNG Browser': raw_data['SAMSUNG Browser'],
        'VIVO Browser': raw_data['VIVO Browser'],
        '华为应用市场': raw_data['华为应用市场'],
        '豆瓣': raw_data['豆瓣'],
        'VIVAME': raw_data['VIVAME'],
        'zaker': raw_data['zaker'],
        '苹果iphone股票': raw_data['苹果iphone股票'],
        'QQ音乐': raw_data['QQ音乐'],
        '百度音乐': raw_data['百度音乐'],
        '酷狗音乐': raw_data['酷狗音乐'],
        '酷我音乐': raw_data['酷我音乐'],
        '网易云音乐': raw_data['网易云音乐'],
        '虾米音乐': raw_data['虾米音乐'],
        '开心消消乐': raw_data['开心消消乐'],
        '天天爱消除': raw_data['天天爱消除'],
        '炉石传说': raw_data['炉石传说'],
        'FT中文网': raw_data['FT中文网'],
        'iTunes Store': raw_data['iTunes Store'],
        '百度外卖': raw_data['百度外卖'],
        '嘀嗒拼车': raw_data['嘀嗒拼车'],
        '广发证券易淘金': raw_data['广发证券易淘金'],
        '美柚': raw_data['美柚'],
        '探探': raw_data['探探'],
        '携程旅行': raw_data['携程旅行'],
        '一点资讯': raw_data['一点资讯'],
        '易游人': raw_data['易游人'],
        '支付宝': raw_data['支付宝'],
        '中国工商银行': raw_data['中国工商银行'],
        '墨迹天气': raw_data['墨迹天气'],
        '美图秀秀': raw_data['美图秀秀'],
        '快的打车': raw_data['快的打车'],
        'appstore': raw_data['appstore'],
        '大麦': raw_data['大麦'],
        '京东': raw_data['京东'],
        '京东到家': raw_data['京东到家'],
        '手机淘宝': raw_data['手机淘宝'],
        '唯品会': raw_data['唯品会'],
        '百度糯米': raw_data['百度糯米'],
        '大众点评': raw_data['大众点评'],
        '饿了么': raw_data['饿了么'],
        '肯德基': raw_data['肯德基'],
        '百度地图': raw_data['百度地图'],
        '高德地图': raw_data['高德地图'],
        '阿里旅行': raw_data['阿里旅行'],
        '去哪儿旅行': raw_data['去哪儿旅行'],
        '同程旅游': raw_data['同程旅游'],
        '易到用车': raw_data['易到用车'],
        'QQ': raw_data['QQ'],
        '微信': raw_data['微信'],
        '陌陌': raw_data['陌陌'],
        '搜狗输入法': raw_data['搜狗输入法'],
        '讯飞输入法': raw_data['讯飞输入法'],
        '百度云': raw_data['百度云'],
        '今日头条': raw_data['今日头条'],
        '知乎': raw_data['知乎'],
        '有道词典': raw_data['有道词典'],
        '喜马拉雅': raw_data['喜马拉雅'],
        '大智慧免费炒股软件': raw_data['大智慧免费炒股软件'],
        '工行手机银行': raw_data['工行手机银行'],
        '农行掌上银行': raw_data['农行掌上银行'],
        '中国建设银行': raw_data['中国建设银行'],
        '固定联络圈规模': raw_data['固定联络圈规模'],
        '访问视频网站的次数': raw_data['访问视频网站的次数'],
        '访问音乐网站的次数': raw_data['访问音乐网站的次数'],
        '访问动漫网站的次数': raw_data['访问动漫网站的次数'],
        '访问搜索网站的次数': raw_data['访问搜索网站的次数'],
        '访问生活网站的次数': raw_data['访问生活网站的次数'],
        '访问购物网站的次数': raw_data['访问购物网站的次数'],
        '访问房产网站的次数': raw_data['访问房产网站的次数'],
        '访问地图网站的次数': raw_data['访问地图网站的次数'],
        '访问餐饮网站的次数': raw_data['访问餐饮网站的次数'],
        '访问旅游网站的次数': raw_data['访问旅游网站的次数'],
        '访问聊天网站的次数': raw_data['访问聊天网站的次数'],
        '访问交友网站的次数': raw_data['访问交友网站的次数'],
        '访问社交网站的次数': raw_data['访问社交网站的次数'],
        '访问问答网站的次数': raw_data['访问问答网站的次数'],
        '访问新闻网站的次数': raw_data['访问新闻网站的次数'],
        '访问金融网站的次数': raw_data['访问金融网站的次数'],
        '访问股票网站的次数': raw_data['访问股票网站的次数']})
    return field


def get_result(predict_result):
    '''得到预测结果概率
    Params:
        predict_results: 原始预测结果
    Returns: 精确到小数点后四位的结果
    '''
    return round(predict_result[1], 6)


def compute_auc(predict_file, real_results):
    '''计算AUC评分
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
    data = pd.read_csv('/data/第1题：算法题数据/数据集1_用户标签_本地_训练集.csv')
    train_data_label = pd.read_csv('/data/第1题：算法题数据/用户是否去过迪士尼_训练集.csv')
    train_data = get_field(data)
    real_results = train_data_label[840000:]

    # 未使用手机品牌字段 counts = data['手机品牌'].value_counts()
    classifier = LogisticRegression()
    classifier.fit(train_data[0:840000], train_data_label['是否去过迪士尼'][0:840000])
    results = classifier.predict_proba(train_data[840000:])
    predict_results = map(get_result, results)

    # 预测结果写入文件
    with open('result.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['IMEI', 'SCORE'])
        for idx in range(len(train_data[840000:])):
            writer.writerow(
                [(data['用户标识'][840000:])[840000 + idx],
                 '%.6f' % predict_results[idx]])
    # 计算AUC评分
    print compute_auc('result.csv', real_results)
