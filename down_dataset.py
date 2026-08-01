# -*- encoding: utf-8 -*-
'''
@File         :tmp.py
@Date         :2026/06/22 16:07:17
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''

import os,sys
import requests

from config import license,tushare_token
import pandas as pd
import tushare as ts
import akshare as ak

def get_requests(url):
    # 发送HTTP GET请求
    response = requests.get(url)

    # 确保请求成功
    if response.status_code == 200:
        data = response.json()
    else:
        data = None

    return data



def get_stock_list(license,save_path='stock_list.csv'):
    """
    下载股票列表
    """
    url = f"https://api.biyingapi.com/hslt/list/{license}"
    list_json_data = get_requests(url)
    dataset = []
    for json_data in list_json_data:
        dm = json_data["dm"]
        mc = json_data["mc"]
        jys = json_data["jys"]
        dataset.append([dm,mc,jys])
    df = pd.DataFrame(dataset, columns=['dm', 'mc', 'jys'])
    df.to_csv(save_path, index=False, encoding='utf-8')

def get_min_data(license,code="002195.SZ",save_root="/home/ai_system/dataset",starttime="20260101",endtime="20260501"):
    """
    下载分钟数据
    """
    url = f"https://api.biyingapi.com/hsstock/history/{code}/5/n/{license}?st={starttime}&et={endtime}"
    datas= get_requests(url=url)
    cols = ["t","o","h","l","c","v","a","pc","sf"]
    dataset = {}
    for data in datas:
        daytime = data["t"].strip().split(" ")[0].replace("-","")
        if daytime in dataset.keys():
            dataset[daytime].append([data[v] for v in cols])
        else:
            dataset[daytime] = [[data[v] for v in cols]]
    save_folder = os.path.join(save_root,code)
    os.makedirs(save_folder,exist_ok=True)
    for  daytime,data in dataset.items():
        df = pd.DataFrame(data, columns=cols)
        df = df.sort_values(by="t").reset_index(drop=True)
        df.to_csv(os.path.join(save_folder,daytime+".csv"), index=False, encoding='utf-8')
        print(data)

def get_trade_date(save_path="/home/ai_system/dataset/trade_date.csv"):
    """
    下载交易日期
    """
    trade_date_df = ak.tool_trade_date_hist_sina()
    trade_date_df['trade_date'] = pd.to_datetime(trade_date_df['trade_date'])
    trade_date_df.to_csv(save_path, index=False, encoding='utf-8')

def each_trade(stock_list_path,save_root="/home/ai_system/dataset"):
    "逐笔交易"
    cols = ["d","t","v","p","ts"]
    df = pd.read_csv(stock_list_path)
    for idx,row in df.iterrows():
        code = row["dm"]
        url = f"https://api.biyingapi.com/hsrl/zbjy/{code.split('.')[0]}/{license}"
        datas= get_requests(url=url)
        if datas is None or len(datas) ==0 :continue
        daytime = datas[0]["d"].replace("-","")
        save_folder = os.path.join(save_root,code)
        os.makedirs(save_folder,exist_ok=True)
        dataset = []
        for data in datas:
            dataset.append([data[v] for v in cols])
        dataset_df = pd.DataFrame(dataset, columns=cols)
        dataset_df = dataset_df.sort_values(by="t").reset_index(drop=True)
        dataset_df.to_csv(os.path.join(save_folder,daytime+"_each_trade.csv"), index=False, encoding='utf-8')
        print(code,daytime)


if __name__ == '__main__':
    get_min_data(license,code="002195.SZ")
    # get_trade_date()
    # each_trade(stock_list_path="/home/ai_system/dataset/stock_list.csv")
    
    