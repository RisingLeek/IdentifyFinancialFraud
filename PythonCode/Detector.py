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
from tkinter import filedialog # 用于选择文件路径

root = tk.Tk()
root.withdraw()

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

    
