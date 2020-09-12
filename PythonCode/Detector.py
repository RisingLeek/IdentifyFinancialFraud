import pandas as pd 
import numpy as np 
import joblib # 导入以训练好的模型
from sklearn.ensemble import RandomForestClassifier # 导入随机森林模型包
import tkinter as tk
from tkinter import filedialog # 用于选择文件路径

root = tk.Tk()
root.withdraw()

modelPath = filedialog.askopenfilename()
modelPath = modelPath.replace('/', '\\')
RFC = joblib.load(modelPath) # 导入以训练好的模型
