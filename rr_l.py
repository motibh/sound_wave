from PIL import Image
from skimage import io
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy import signal
from scipy.interpolate import interp1d
def func_lin(x, a, b):
    return a * x + b
def func_parb(x, a, b,c):
    # c=0
    return a * (x ** 2) + b * x + c
def func_sin(x,a, b):
    return  a*np.sin(x * b)

# rooling  all examples
for name in ['line','parabola','sine']:
    for num in range(5):
        #name=input('fun  type?')# parabola5 line1
        img1 =np.abs( io.imread(name+str(num+1)+'.png', as_gray=True)-1)
        # remove noise
        img1=img1.T
        filter = np.array([[ -1,1 ],
                           [  1,-1]])
        img = np.abs(signal.convolve2d(img1, filter, boundary='symm', mode='same'))
        #
        mean_gray_loc=[]
        idx_loc=[]
        for i  in range(img.shape[1]):
            arr = np.array(img[i])
            def condition(x): return x >0
            bool_arr = condition(arr)
            output = np.where(bool_arr)[0]
            if len(output)>1:
                output=( img[i][output])@output/np.sum( img[i][output])
                mean_gray_loc.append(np.random.normal(np.mean(output), np.std(output), 1)[0]  ) # std)
                idx_loc.append(i)
        # shift to the center
        idx_loc -= np.mean(idx_loc)
        mean_gray_loc -= np.mean(mean_gray_loc)
        # interpolation
        f = interpolate.interp1d(idx_loc, mean_gray_loc)
        xnew = np.arange(min(idx_loc), max(idx_loc), 1)
        ynew = f(xnew)  # use interpolation function returned by `interp1d`

        # find the optimal curve
        err=[]
        std_diff2 = []
        # find goodness of fit for all functions
        # ***  line ***
        popt, pcov = curve_fit(func_lin, xnew, ynew)
        y_pred=popt[0] * (xnew ** 1) + popt[1]
        err.append( r2_score(ynew,y_pred))
        # ***  parabola ***
        popt, pcov = curve_fit(func_parb, xnew, ynew)
        # for case that quardatic term can be neglect
        if np.abs(popt[1]/popt[0])>50:
            err.append(-1000000)
            #std_diff2.append(0)
        else:
            y_pred=popt[0] * (xnew ** 2) + popt[1] * xnew + popt[2]
            err.append( r2_score(ynew,y_pred))
            #std_diff2.append(np.std(np.diff(np.diff(np.diff(ynew)))))
        # ***  sine ***
        popt, pcov = curve_fit(func_sin, xnew, ynew)
        y_pred=popt[0] * np.sin(xnew *popt[1])
        err.append( r2_score(ynew,y_pred))
        std_diff2 = np.mean(np.abs(np.diff(ynew) / ynew[1:]))
        #std_diff2.append(np.std(np.diff(np.diff(np.diff(ynew)))))
        # for sine
        x_l=xnew[ynew<0.5*np.median(ynew)]
        x_h = xnew[ynew >1.5*np.median(ynew)]
        #x_ratio=(np.mean(x_l)/np.mean(x_h))  * (np.std(x_l)/np.std(x_h))
        #
        if max(err)==err[0]:
            print(f" predict: line. True value is {name}  ")
        if max(err)==err[1]:
            print(f" predict: parabola. True value is {name}  ")
        if max(err) == err[2]:  # or (std_diff2>2):
            print(f" predict: sine. True value is {name}  ")
