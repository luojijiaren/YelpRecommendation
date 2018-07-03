# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:16:24 2018

@author: fzhan
"""
import chardet
import pandas as pd

with open('recommate_rest.csv', 'rb') as f:
    result = chardet.detect(f.read()) 
rest=pd.read_csv('recommate_rest.csv',index_col=0,encoding=result['encoding'])

if __name__ == "__main__":
    i=input('Customer ID:   '   )
    l=rest.iloc[int(i),:]
    print('5 recommendated restaurants are:   ',list(l) )