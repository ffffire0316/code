from getTime import getKeyDatas

import pandas as pd
import os,sys
import csv

def set_file_name(filename):
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
   # file=filename.join(".csv")
    file=filename+'.csv'
    path = os.path.join(path, 'data', file)
    return path

def write_csv_using_csv(path,header,data):
    with open(path, 'w',newline="") as f:

        # writer=csv.DictWriter(f,fieldnames=header)
        writer=csv.writer(f)
        #writer.writeheader()
        writer.writerow(header)
        writer.writerow(data)

def write_data_using_pandas(write_data:list,write_path:str):
    import pandas as pd

    df=pd.DataFrame(data=write_data)

    if not os.path.exists(write_path):
        df.to_csv(write_path,header=header,index=False,mode='a+',chunksize=None)
    #mode a:only write ;a+:write or read  both no wiping
    #mode r:only write ,wipe from the head
    else:
        df.to_csv(write_path,header=False,index=False,mode='a+',chunksize=None)


if __name__=="__main__":
    myname='huangyi'
    path=set_file_name(myname)

    """封装下列"""
    header = ['press0', 'press0', 'press0', 'press0', 'press0', 'press0', 'press0', 'press0', 'press0', 'press10',
              'release', 'release', 'release', 'release', 'release', 'release', 'release', 'release', 'release9',
              'period', 'period', 'period', 'period', 'period', 'period', 'period', 'period', 'period9', ]

    all_data=getKeyDatas(3)
    write_data_using_pandas(all_data, path)  #写入数据