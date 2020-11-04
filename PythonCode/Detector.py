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
from tkinter import Label
from tkinter import Entry
from tkinter import IntVar
from tkinter import Radiobutton

global file_path # 文件路径
global model_type_selected # 模型模式
file_path = ' '
model_type_selected = 0

def load_file(entry):
    # 获取文件路径
    global file_path
    from tkinter import filedialog
    entry.delete(0, 'end')
    file_path = tk.filedialog.askopenfilename()
    entry.insert(10, file_path)

def choose_model(v):
    # 确定模型模式
    global model_type_selected
    model_type_selected = v.get()
    model_type_selected = int(model_type_selected)

def test(root, entry):
    global file_path
    global model_type_selected

    #file_path = file_path.decode('utf-8')

    # 检测文件路径是否合法
    if (file_path == ' '):
        root_tmp = tk.Tk()
        root_tmp.title('警告')
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (200, 200, (screen_w-200)/2, (screen_h-200)/2)
        root_tmp.geometry(alignstr)
        Label(root_tmp, text='未选择数据源！', fg='red', width=20, height=6, font=(6)).grid(row=0)
        tk.Button(root_tmp, text='确定', width=3, height=1, command=root_tmp.destroy, font=(6)).grid(row=1, padx=3, pady=3)
        return

    # 检测文件
    file_token = file_path.split('.')
    file_token = file_token[-1]
    if (file_token != 'xls' and file_token != 'xlsx'):
        root_tmp = tk.Tk()
        root_tmp.title('警告')
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (200, 200, (screen_w-200)/2, (screen_h-200)/2)
        root_tmp.geometry(alignstr)
        Label(root_tmp, text='文件类型错误！', fg='red', width=20, height=6, font=(6)).grid(row=0)
        tk.Button(root_tmp, text='确定', width=3, height=1, command=root_tmp.destroy, font=(6)).grid(row=1, padx=3, pady=3)
        return
    
    # 打开文件
    root_tmp = tk.Tk()
    root_tmp.title('提示')
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    alignstr = '%dx%d+%d+%d' % (200, 200, (screen_w-200)/2, (screen_h-200)/2)
    root_tmp.geometry(alignstr)
    Label(root_tmp, text='处理中', fg='red', width=20, height=6, font=(6)).grid(row=0)
    
    data_x = pd.read_excel(file_path)
    
    # 处理数据
    RFC = joblib.load("../PythonModel/RFC.pkl")
    SVR = joblib.load("../PythonModel/SVR.pkl")
    SCLF = joblib.load("../PythonModel/SCLF.pkl")

    data_x = data_x[['流动比率', '速动比率', '资产负债率',\
                        '经营活动现金流量总额/负债合计', '应收账款与收入比',\
                        '存货与收入比', '存货周转率', '流动资产与收入比',\
                        '总资产周转率', '其他应收款占流动资产比例',\
                        '资产报酬率', '总资产净利润率', '营业毛利率']]
    print(data_x.head())

    if (model_type_selected == 0):
        res = SVR.predict(data_x)
        #res_pro = SVR.predict_proba(data_x)
    elif (model_type_selected == 1):
        res = RFC.predict(data_x)
        res_pro = RFC.predict_proba(data_x)
    elif (model_type_selected == 2):
        res = SCLF.predict(data_x)
        res_pro = SCLF.predict_proba(data_x)
    else:
        root_tmp.destroy()
        root_tmp = tk.Tk()
        root_tmp.title('警告')
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (200, 200, (screen_w-200)/2, (screen_h-200)/2)
        root_tmp.geometry(alignstr)
        Label(root_tmp, text='预测失败！', fg='red', width=20, height=6, font=(6)).grid(row=0)
        tk.Button(root_tmp, text='确定', width=3, height=1, command=root_tmp.destroy, font=(6)).grid(row=1, padx=3, pady=3)
        return
    root_tmp.destroy()
    root_tmp = tk.Tk()
    root_tmp.title('提示')
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    alignstr = '%dx%d+%d+%d' % (200, 200, (screen_w-200)/2, (screen_h-200)/2)
    root_tmp.geometry(alignstr)
    Label(root_tmp, text='预测成功！', fg='red', width=20, height=6, font=(6)).grid(row=0)
    tk.Button(root_tmp, text='确定', width=3, height=1, command=root_tmp.destroy, font=(6)).grid(row=1, padx=3, pady=3)
    entry.delete(0, 'end')
    if (res == 1 and model_type_selected != 0):
        result = '此份财务数据涉嫌造假，概率为' + str(int(res_pro[0][0]*100)) + '%'
    elif (res == 0 and model_type_selected != 0):
        result = '此份财务数据正常，概率为' + str(int(res_pro[0][0]*100)) + '%'
    elif  (res == 1 and model_type_selected == 0):
        result = '此份财务数据涉嫌造假'
    elif  (res == 0 and model_type_selected == 0):
        result = '此份财务数据正常'
    entry.insert(10, result)
    return

def UI():
    # 软件窗口大小
    sWidth = 400
    sHeight = 180

    # 初始化窗口
    root = tk.Tk()
    root.title('财务造假识别 V1.001')

    # 窗口居中
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    alignstr = '%dx%d+%d+%d' % (sWidth, sHeight, (screen_w-sWidth)/2, (screen_h-sHeight)/2)
    root.geometry(alignstr)

    # 第一行布局
    Label(root, text="文件路径", font=(6)).grid(row=0)
    entry1 = Entry(root, width=30, xscrollcommand=True)
    entry1.grid(row=0, column=1)
    # 设置载入按钮
    load_button = tk.Button(root, text='浏览', command=lambda: load_file(entry1), font=(6))
    load_button.grid(row=0, column=2, padx=3,pady=3)

    # 第二行布局
    v = IntVar()
    select_model = [('支持向量机', 0), ('随机森林',1), ('融合模型', 2)]
    for model_type, num in select_model:
        model_radio_button = Radiobutton(root, text=model_type, value=num, command=lambda: choose_model(v), variable=v, font=(6))
        model_radio_button.grid(row=1, column=num)

    # 第三行布局
    Label(root, text="结果", font=(6)).grid(row=2)
    entry2 = Entry(root, width=30, xscrollcommand=True)
    entry2.grid(row=2, column=1)
    # 设置预测按钮
    load_button = tk.Button(root, text='开始', command=lambda: test(root, entry2), font=(6))
    load_button.grid(row=2, column=2, padx=3,pady=3)

    root.mainloop()

def main():
    UI()

if __name__ == '__main__':
    main()
