import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time

def dict_all_csv(sample_rate=50, time_v=[9.5,60]):
    """
           merge all .csv files into single dict
    """
    dict_all={}
    for box in [1,2,3]:
        for pair in [1,2]:
            for time in time_v:#,10.75,12,13.5,15.75,60]:
                 #reading excel file
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
s_rate=4
smoth = False
freq=1/(s_rate*0.0125e-6)
# merge all data to single dict
dict_csv=dict_all_csv(sample_rate=s_rate,time_v=time_v)
fig, axs = plt.subplots(6)
fig.suptitle('f_max vs. time (all sensors )')
fig3, axs3 = plt.subplots(6)
fig3.suptitle('f_max_intensity vs. time (all sensors )')
i=0
fig2, axs2 = plt.subplots(1)
fig2.suptitle('match filter delay')
for box in [1, 2, 3]:
    for pair in [1, 2]:
        filter_delay = []
        fft_max = []
        fft_max_intensty = []
        for time in time_v:
            df_exp=dict_csv[f'box{box}_pair{pair}_{time}']
            if smoth:  # smooth data noise
                # plt.figure()
                # plt.plot(df_exp['Channel B'].iloc[:700],'r')
                len_smot = 20
                df_exp['Channel B'] = df_exp['Channel B'].rolling(len_smot, center= True).sum()/len_smot
                #plt.plot(df_exp['Channel B'].iloc[:700]+0,'b')
                #plt.show()
            # Match fiter
            conv_in_out = np.convolve( df_exp['Channel A']-df_exp['Channel A'].min(), np.power(df_exp['Channel B'],2))
            # fft
            fft_out = np.fft.fft(df_exp['Channel B'])
            # plt.figure()
            # plt.plot(np.log10(np.abs(fft_out)))
            # plt.show()
            #plt.savefig(f'in_gv/box{box}_pair{pair}_{time} .png')
            filter_delay.append(np.argmax(conv_in_out))
            f_max= np.argmax(np.abs(fft_out))
            f_max_intn = np.max(np.abs(fft_out))
            if f_max>(len(fft_out)/2):
               f_max=f_max- len(fft_out)
            fft_max.append(f_max/len(fft_out))
            fft_max_intensty.append(f_max_intn)
        axs[i].plot(time_v,fft_max,'o')# filter_delay
        axs[i].plot(time_v,np.zeros(6), 'r')
        axs[i].set_xlabel('Time')
        axs3[i].plot(time_v, fft_max_intensty, 'o')  # filter_delay
        axs3[i].set_xlabel('Time')
#        axs[i].grid()
        fig.show()
        fig3.show()
        # match filter
        axs2.plot(time_v,filter_delay)#fft_max
        axs2.set_xlabel('Time')
        fig2.show()
        i+=1
plt.figure()
plt.plot(filter_delay)

time.sleep(5.5)    # Pause 5.5 seconds
