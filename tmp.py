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
import numpy as np

import pandas as pd
from datetime import datetime,timedelta







if __name__ == '__main__':
    
    start_date = datetime(2026, 1, 5)
    end_date = datetime(2026, 7, 23)
    current = start_date
    while current <= end_date:

        old_date = "20260105"
        base_path = f"/workspace/afb5szh-01/models/stock/dataset/{old_date}.csv"
        df = pd.read_csv(base_path)

        date = current.strftime('%Y%m%d')
        current += timedelta(days=1)
        # date = "20260106"
        cols = ["t"]
        for col in cols:
            df[col] = df[col].str.replace(
                datetime.strptime(old_date, "%Y%m%d").strftime("%Y-%m-%d"), 
                datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d"), 
                regex=False)
        df.to_csv(base_path.replace(old_date,date), index=False)
    