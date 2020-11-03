import os.path
import pandas as pd 
import numpy as np 
import joblib # 导入以训练好的模型
from sklearn.ensemble import RandomForestClassifier # 导入随机森林模型包
from sklearn.ensemble import GradientBoostingClassifier # 导入梯度森林模型包
from sklearn.svm import SVC # 导入向量机模型包
from sklearn.linear_model import LogisticRegression # 导入logistics模型包
from mlxtend.classifier import StackingClassifier # 导入stacking包
import tkinter as tk

def load_file(root, file_path):
    # 获取文件路径
    from tkinter import filedialog
    file_path = tk.filedialog.askopenfilename()
    return

# 软件窗口大小
sWidth = 480
sHeight = 360

# 初始化窗口
root = tk.Tk()
root.title('财务造假识别')

# 窗口居中
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (sWidth, sHeight, (screen_w-sWidth)/2, (screen_h-sHeight)/2)
root.geometry(alignstr)

file_path1 = ' '
file_path2 = ' '
file_path3 = ' '

load_button_1 = tk.Button(root, text='浏览', command=lambda: load_file(root, file_path1), width=8, height=2)
load_button_1.grid(row=0, column=0, sticky=tk.W, padx=5,pady=5)


root.mainloop()

def load_learn_model():

    # 导入弹窗提示
    import tkinter.messagebox

    # 导入以训练好的模型
    if (os.path.isfile("../PythonModel/RFC.pkl")):
        RFC_p = joblib.load("../PythonModel/RFC.pkl")
    else:
        RFC_p = False
        result = tkinter.messagebox.askokcancel(title = '警告',message='模型导入失败或模型不存在')
        print(result)
    if (os.path.isfile("../PythonModel/GBC.pkl")):
        GBC_p = joblib.load("../PythonModel/GBC.pkl")
    else:
        GBC_p = False
        result = tkinter.messagebox.askokcancel(title = '警告',message='模型导入失败或模型不存在')
        print(result)
    if (os.path.isfile("../PythonModel/SVR.pkl")):
        SVR_p = joblib.load("../PythonModel/SVR.pkl")
    else:
        SVR_p = False
        result = tkinter.messagebox.askokcancel(title = '警告',message='模型导入失败或模型不存在')
        print(result)
    if (os.path.isfile("../PythonModel/LOR.pkl")):
        LOR_p = joblib.load("../PythonModel/LOR.pkl")
    else:
        LOR_p = False
        result = tkinter.messagebox.askokcancel(title = '警告',message='模型导入失败或模型不存在')
        print(result)
    return RFC_p, GBC_p, SVR_p, LOR_p

    
