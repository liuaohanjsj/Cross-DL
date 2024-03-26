# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 17:45:39 2021

@author: Admin
"""

import xlrd
import numpy as np

def load_cq500_original_label(file, naoshi=False):
    def vote(row, col):
        cnt = 0
        for i in range(3):
            cnt += sh.cell_value(row, col+i*14)
        return 1 if cnt>=2 else 0
    
    data = xlrd.open_workbook(file)
    sh = data.sheet_by_index(0)
    n = sh.nrows
    ret = {}
    for i in range(1,n):
        label = []
        label.append(vote(i,2)) #颅内出血 
        label.append(vote(i,3)) #脑内出血
        label.append(vote(i, 8)) #左
        label.append(vote(i, 9)) #右
        label.append(vote(i, 6)) #硬膜外
        label.append(vote(i, 5)) #硬膜下
        label.append(vote(i, 4)) #脑室
        label.append(vote(i, 7)) #蛛网膜下腔
        ret[sh.cell_value(i, 0)[9:]] = label
    return ret
