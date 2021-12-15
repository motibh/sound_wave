import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
def dict_all_csv(sample_rate=50, time_v=[9.5,60]):
    dict_all={}
    for box in [1,2,3]:
        for pair in [1,2]:
            for time in time_v:#,10.75,12,13.5,15.75,60]:
                 #reading excel file - glucose sheet
                try:
                    xls = pd.read_csv(os.getcwd()+f'/210829_experiment/box{box}/pair{pair}/{time}/{time}_1.csv')
                    xls=xls.drop([0])
                    xls=xls.astype(float)
                    dict_all[f'box{box}_pair{pair}_{time}']=xls.loc[xls.index % sample_rate == 0]
                   # xls.Time.diff().unique().
                except:
                    continue
    return dict_all
time_v= [9.5,10.75,12,13.5,15.75,60]
dict_csv=dict_all_csv(sample_rate=1,time_v=time_v)
fig, axs = plt.subplots(6)
fig.suptitle('??????????????????')
i=0
for box in [1, 2, 3]:
    for pair in [1, 2]:
        filter_delay = []
        fft_max = []
        for time in time_v:
            try:
                df_exp=dict_csv[f'box{box}_pair{pair}_{time}']
                conv_in_out = np.convolve( df_exp['Channel A']-df_exp['Channel A'].min(), np.power(df_exp['Channel B'],2))
                fft_out = np.fft.fft(df_exp['Channel B'])
                #plt.figure()
                #plt.plot(conv_in_out)
                ratio=1#np.max(df_exp['Channel A'])/np.max(df_exp['Channel B'])
                #plt.plot(df_exp.Time, df_exp['Channel B'], 'r')
                #plt.plot(df_exp.Time, df_exp['Channel A']/ratio, 'b')
                #plt.savefig(f'in_gv/box{box}_pair{pair}_{time} .png')
                filter_delay.append(np.argmax(conv_in_out))
                fft_max.append(np.argmax(np.abs(fft_out)))
            except:
                continue
        axs[i].plot(fft_max)# filter_delay
        fig.show()
        i+=1
plt.figure()
plt.plot(filter_delay)
dd=1