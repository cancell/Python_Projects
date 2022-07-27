# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:32:10 2022

@author: Dell User
"""

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd              
import numpy as np               
import matplotlib.pyplot as plt  
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set()
import glob
import random
import os
import pickle
from tqdm import tqdm
# Core
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set(rc={'figure.figsize':[7,7]},font_scale=1.2)

import sklearn
from  datasist.structdata import detect_outliers
from sklearn.preprocessing import LabelEncoder
# Pre Processing
from sklearn.impute import SimpleImputer 
from tkinter import *
from tkinter import messagebox

df = pd.read_csv("https://github.com/cancell/Python_Projects/blob/master/Pandas%26Numpy%26Scikit%20Learn%20Projects/googleplaystore.csv") 
#or you can added your own link from your desktop

def to_numeric(field):
    # case 'M'
    if field[-1] == 'M':
        return float(field[:-1])
        #cas 'k'
    elif field[-1] == 'k':
        return float(field[:-1])/1024
        #   case 1,000+
    elif  field == '1,000+':
        return 1.0     
    else :
        return np.nan   
# this function usde to fix issue of price column 
def fix_price(field):
    #cas  price start with #
    if field.startswith('$'):
        return float(field[1:])
    else:
        return 0.0

df['Reviews'] = pd.to_numeric(df['Reviews'],errors='coerce')
df['Size']    = df['Size'].apply(to_numeric)  
df['Price']   = df['Price'].apply(fix_price) 

df.drop(['Genres', 'Current Ver','Last Updated'],inplace=True,axis=1)
idx = df[ df['Rating'] > 5].index
df.drop(idx,inplace=True,axis=0)
df.describe()

imputer           = SimpleImputer(missing_values= np.nan, strategy='mean') 
cat_imputer       = SimpleImputer(missing_values= np.nan, strategy='most_frequent')
df['Size']        = imputer.fit_transform(df[['Size']])
df['Rating']      = imputer.fit_transform(df[['Rating']])
df['Android Ver'] = cat_imputer.fit_transform(df[['Android Ver']])
df['Type']        = cat_imputer.fit_transform(df[['Type']])


window= Tk()
window.title("Data Predictor")
window.geometry("1366x768")

C = Canvas(window, bg="blue", height=400, width=400)
filename = PhotoImage(file = "/Users/apple1/Desktop/ai.png")
background_label = Label(window, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
C.pack()


lbl4=Label(window,width=30,text="Welcome to Data Predictor :)")
lbl4.config(font="Arial 30")
lbl4.place(x=100,y=300)

def start():
    new_window = Tk()
    new_window.title("Data Predictor")
    new_window.geometry("1366x768")
    window.withdraw()
    new_window.deiconify() 
    
    def game():
        v_window = Tk()
        v_window.title("GOOGLE PLAY STORE APPS INFO")
        v_window.geometry("1366x768")
        new_window.withdraw()
        v_window.deiconify() 
        ozellik = ["5 Star", "1 Star", "Most expensive", "Free","Most downloaded"] 
        veri= StringVar(v_window)
        veri.set("------") 
        w = OptionMenu(v_window, veri, *ozellik)
        w.config(width=90, font=('Helvetica', 12))
        w.place(x=220,y=105)
        

        def backApp():
            v_window.withdraw()
            new_window.deiconify()
        def callback(*args):
            if veri.get()=="5 Star":
                
                text = Text(v_window, height=10, width=80)
                text.place(x=220, y= 150)
                scroll = Scrollbar(v_window)
                text.configure(yscrollcommand=scroll.set)
                #text.pack(side=LEFT)
                  
                scroll.config(command=text.yview)
                scroll.pack(side=RIGHT, fill=Y)
                  
                insert_text = df[df['Rating']==5]["App"].head(10)
                text.insert(END, insert_text)
            elif veri.get()=="1 Star":
                
                text = Text(v_window, height=10, width=80)
                text.place(x=220, y= 150)
                scroll = Scrollbar(v_window)
                text.configure(yscrollcommand=scroll.set)
                #text.pack(side=LEFT)
                  
                scroll.config(command=text.yview)
                scroll.pack(side=RIGHT, fill=Y)
                  
                insert_text = df[df['Rating']==1]["App"].head(10)
                text.insert(END, insert_text)
            elif veri.get()=="Most expensive":

                
                text = Text(v_window, height=10, width=80)
                text.place(x=220, y= 150)
                scroll = Scrollbar(v_window)
                text.configure(yscrollcommand=scroll.set)
                #text.pack(side=LEFT)
                  
                scroll.config(command=text.yview)
                scroll.pack(side=RIGHT, fill=Y)
                  
                
                insert_text = df[["App","Price"]].sort_values(by=["Price"],ascending=False)[0:10]
                text.insert(END, insert_text)
            
            elif veri.get()=="Most downloaded":
                
                text = Text(v_window, height=10, width=80)
                text.place(x=220, y= 150)
                scroll = Scrollbar(v_window)
                text.configure(yscrollcommand=scroll.set)
                #text.pack(side=LEFT)
                  
                scroll.config(command=text.yview)
                scroll.pack(side=RIGHT, fill=Y)
               
                insert_text = df[["App","Installs"]].sort_values(by=["Installs"],ascending=False)[0:10]
                text.insert(END, insert_text)
                
            elif veri.get()=="Free":
                
                text = Text(v_window, height=10, width=80)
                text.place(x=220, y= 150)
                scroll = Scrollbar(v_window)
                text.configure(yscrollcommand=scroll.set)
                #text.pack(side=LEFT)
                  
                scroll.config(command=text.yview)
                scroll.pack(side=RIGHT, fill=Y)
                  
                insert_text = df[df['Price']==0]["App"].head(10)
                text.insert(END, insert_text)
        

        veri.trace("w", callback)
        btnback=Button(v_window,width=10,text=">>BACK<<",command=backApp)
        btnback.config(font="Arial 30", fg="black")
        btnback.place(x=1000,y=700)    
    
    def car():
        car = pd.read_csv("https://github.com/cancell/Python_Projects/blob/master/Pandas%26Numpy%26Scikit%20Learn%20Projects/carprice.csv")
        #or you can added your own link from your desktop
        predict = "price"
        car = car[["curbweight", "enginesize", "horsepower", "price"]]
        x = np.array(car.drop([predict], 1))
        y = np.array(car[predict])

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
        
        m_window = Tk()
        m_window.title("CAR PREDICTION")
        m_window.geometry("1366x768")
        new_window.withdraw()
        m_window.deiconify() 
        ozellik = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regressor", "Random Forest Regressor"] 
        veri= StringVar(m_window)
        veri.set("------") 
        w = OptionMenu(m_window, veri, *ozellik)
        w.config(width=90, font=('Helvetica', 12))
        w.place(x=220,y=105)
        
        # labelTest = Label(m_window, text="")
        # labelTest.config(width=90, font=('Helvetica', 16))
        # labelTest.place(x=220, y= 250)

        def callback(*args):
            messagebox.showinfo('Info', 'Please estimate Brake Power and Engine Size, Horsepower and click the button.')
            if veri.get()=="Linear Regression":
                
                lbl1 = Label(m_window, text= "Break Power")
                lbl1.place(x=220, y=145)

                lbl2 = Label(m_window, text= "Horse Power")
                lbl2.place(x=220, y=185)

                lbl3 = Label(m_window, text= "Engine Size")
                lbl3.place(x=220, y=225)
                
                lbl4 = Label(m_window, text= "Price")
                lbl4.place(x=220, y=265)
                
                def click(event):
                    txt1.config(state=NORMAL)
                    txt1.delete(0, END)
                def click2(event):
                    txt2.config(state=NORMAL)
                    txt2.delete(0, END)
                def click3(event):
                    txt4.config(state=NORMAL)
                    txt4.delete(0, END)
                

                txt1 =Entry(m_window, width=25)
                txt1.insert(0, "Min: 1488 - Max: 4066")
                txt1.configure(state=DISABLED)
                txt1.bind("<Button-1>",click)
                txt1.place(x=400, y=145)

                txt2 =Entry(m_window, width=25)
                txt2.insert(0, "Min: 48 - Max: 288")
                txt2.configure(state=DISABLED)
                txt2.bind("<Button-1>",click2)
                txt2.place(x=400, y=185)

                txt3 =Entry(m_window, width=25)
                txt3.place(x=400, y=265)

                txt4 =Entry(m_window, width=25)
                txt4.insert(0, "Min: 61 - Max: 326")
                txt4.configure(state=DISABLED)
                txt4.bind("<Button-1>",click3)
                txt4.place(x=400, y=225)

                def get(number1, number2, number3):
                    
                    txt1.delete(0,END)
                    txt2.delete(0,END)
                    txt3.delete(0,END)
                    txt4.delete(0,END)
                    model = LinearRegression()
                    model.fit(xtrain, ytrain)
                    new_data = [float(number1), float(number2),float(number3)]
                    new_data = pd.DataFrame(new_data).T
                    model.predict(new_data)
                    txt3.insert(END, str(model.predict(new_data)))

                btn = Button(m_window, text="Hesapla", command=lambda: get(float(txt1.get()), float(txt2.get()), float(txt4.get())))
                btn.place(x=400, y=305)
                
            elif veri.get()=="Ridge Regression":
                                
                lbl1 = Label(m_window, text= "Break Power")
                lbl1.place(x=220, y=145)

                lbl2 = Label(m_window, text= "Horse Power")
                lbl2.place(x=220, y=185)

                lbl3 = Label(m_window, text= "Engine Size")
                lbl3.place(x=220, y=225)
                
                lbl4 = Label(m_window, text= "Price")
                lbl4.place(x=220, y=265)
                
                def click(event):
                    txt1.config(state=NORMAL)
                    txt1.delete(0, END)
                def click2(event):
                    txt2.config(state=NORMAL)
                    txt2.delete(0, END)
                def click3(event):
                    txt4.config(state=NORMAL)
                    txt4.delete(0, END)
                

                txt1 =Entry(m_window, width=25)
                txt1.insert(0, "Min: 1488 - Max: 4066")
                txt1.configure(state=DISABLED)
                txt1.bind("<Button-1>",click)
                txt1.place(x=400, y=145)

                txt2 =Entry(m_window, width=25)
                txt2.insert(0, "Min: 48 - Max: 288")
                txt2.configure(state=DISABLED)
                txt2.bind("<Button-1>",click2)
                txt2.place(x=400, y=185)

                txt3 =Entry(m_window, width=25)
                txt3.place(x=400, y=265)

                txt4 =Entry(m_window, width=25)
                txt4.insert(0, "Min: 61 - Max: 326")
                txt4.configure(state=DISABLED)
                txt4.bind("<Button-1>",click3)
                txt4.place(x=400, y=225)

                def get(number1, number2, number3):
                    
                    txt1.delete(0,END)
                    txt2.delete(0,END)
                    txt3.delete(0,END)
                    txt4.delete(0,END)
                    model = Ridge()
                    model.fit(xtrain, ytrain)
                    new_data = [float(number1), float(number2),float(number3)]
                    new_data = pd.DataFrame(new_data).T
                    model.predict(new_data)
                    txt3.insert(END, str(model.predict(new_data)))

                btn = Button(m_window, text="Hesapla", command=lambda: get(float(txt1.get()), float(txt2.get()), float(txt4.get())))
                btn.place(x=400, y=305)

            elif veri.get()=="Lasso Regression":
                
                lbl1 = Label(m_window, text= "Break Power")
                lbl1.place(x=220, y=145)

                lbl2 = Label(m_window, text= "Horse Power")
                lbl2.place(x=220, y=185)

                lbl3 = Label(m_window, text= "Engine Size")
                lbl3.place(x=220, y=225)
                
                lbl4 = Label(m_window, text= "Price")
                lbl4.place(x=220, y=265)
                
                def click(event):
                    txt1.config(state=NORMAL)
                    txt1.delete(0, END)
                def click2(event):
                    txt2.config(state=NORMAL)
                    txt2.delete(0, END)
                def click3(event):
                    txt4.config(state=NORMAL)
                    txt4.delete(0, END)
                

                txt1 =Entry(m_window, width=25)
                txt1.insert(0, "Min: 1488 - Max: 4066")
                txt1.configure(state=DISABLED)
                txt1.bind("<Button-1>",click)
                txt1.place(x=400, y=145)

                txt2 =Entry(m_window, width=25)
                txt2.insert(0, "Min: 48 - Max: 288")
                txt2.configure(state=DISABLED)
                txt2.bind("<Button-1>",click2)
                txt2.place(x=400, y=185)

                txt3 =Entry(m_window, width=25)
                txt3.place(x=400, y=265)

                txt4 =Entry(m_window, width=25)
                txt4.insert(0, "Min: 61 - Max: 326")
                txt4.configure(state=DISABLED)
                txt4.bind("<Button-1>",click3)
                txt4.place(x=400, y=225)

                def get(number1, number2, number3):
                    
                    txt1.delete(0,END)
                    txt2.delete(0,END)
                    txt3.delete(0,END)
                    txt4.delete(0,END)
                    model = Lasso()
                    model.fit(xtrain, ytrain)
                    new_data = [float(number1), float(number2),float(number3)]
                    new_data = pd.DataFrame(new_data).T
                    model.predict(new_data)
                    txt3.insert(END, str(model.predict(new_data)))

                btn = Button(m_window, text="Hesapla", command=lambda: get(float(txt1.get()), float(txt2.get()), float(txt4.get())))
                btn.place(x=400, y=305)
            elif veri.get()=="Decision Tree Regressor":
                
                lbl1 = Label(m_window, text= "Break Power")
                lbl1.place(x=220, y=145)

                lbl2 = Label(m_window, text= "Horse Power")
                lbl2.place(x=220, y=185)

                lbl3 = Label(m_window, text= "Engine Size")
                lbl3.place(x=220, y=225)
                
                lbl4 = Label(m_window, text= "Price")
                lbl4.place(x=220, y=265)
                
                def click(event):
                    txt1.config(state=NORMAL)
                    txt1.delete(0, END)
                def click2(event):
                    txt2.config(state=NORMAL)
                    txt2.delete(0, END)
                def click3(event):
                    txt4.config(state=NORMAL)
                    txt4.delete(0, END)
                

                txt1 =Entry(m_window, width=25)
                txt1.insert(0, "Min: 1488 - Max: 4066")
                txt1.configure(state=DISABLED)
                txt1.bind("<Button-1>",click)
                txt1.place(x=400, y=145)

                txt2 =Entry(m_window, width=25)
                txt2.insert(0, "Min: 48 - Max: 288")
                txt2.configure(state=DISABLED)
                txt2.bind("<Button-1>",click2)
                txt2.place(x=400, y=185)

                txt3 =Entry(m_window, width=25)
                txt3.place(x=400, y=265)

                txt4 =Entry(m_window, width=25)
                txt4.insert(0, "Min: 61 - Max: 326")
                txt4.configure(state=DISABLED)
                txt4.bind("<Button-1>",click3)
                txt4.place(x=400, y=225)

                def get(number1, number2, number3):
                    
                    txt1.delete(0,END)
                    txt2.delete(0,END)
                    txt3.delete(0,END)
                    txt4.delete(0,END)
                    model = DecisionTreeRegressor(random_state=0)
                    model.fit(xtrain, ytrain)
                    new_data = [float(number1), float(number2),float(number3)]
                    new_data = pd.DataFrame(new_data).T
                    model.predict(new_data)
                    txt3.insert(END, str(model.predict(new_data)))

                btn = Button(m_window, text="Hesapla", command=lambda: get(float(txt1.get()), float(txt2.get()), float(txt4.get())))
                btn.place(x=400, y=305)
                
            elif veri.get()=="Random Forest Regressor":
                
                lbl1 = Label(m_window, text= "Break Power")
                lbl1.place(x=220, y=145)

                lbl2 = Label(m_window, text= "Horse Power")
                lbl2.place(x=220, y=185)

                lbl3 = Label(m_window, text= "Engine Size")
                lbl3.place(x=220, y=225)
                
                lbl4 = Label(m_window, text= "Price")
                lbl4.place(x=220, y=265)
                
                def click(event):
                    txt1.config(state=NORMAL)
                    txt1.delete(0, END)
                def click2(event):
                    txt2.config(state=NORMAL)
                    txt2.delete(0, END)
                def click3(event):
                    txt4.config(state=NORMAL)
                    txt4.delete(0, END)
                

                txt1 =Entry(m_window, width=25)
                txt1.insert(0, "Min: 1488 - Max: 4066")
                txt1.configure(state=DISABLED)
                txt1.bind("<Button-1>",click)
                txt1.place(x=400, y=145)

                txt2 =Entry(m_window, width=25)
                txt2.insert(0, "Min: 48 - Max: 288")
                txt2.configure(state=DISABLED)
                txt2.bind("<Button-1>",click2)
                txt2.place(x=400, y=185)

                txt3 =Entry(m_window, width=25)
                txt3.place(x=400, y=265)

                txt4 =Entry(m_window, width=25)
                txt4.insert(0, "Min: 61 - Max: 326")
                txt4.configure(state=DISABLED)
                txt4.bind("<Button-1>",click3)
                txt4.place(x=400, y=225)

                def get(number1, number2, number3):
                    
                    txt1.delete(0,END)
                    txt2.delete(0,END)
                    txt3.delete(0,END)
                    txt4.delete(0,END)
                    model = RandomForestRegressor(n_estimators = 10, random_state=0)
                    model.fit(xtrain, ytrain)
                    new_data = [float(number1), float(number2),float(number3)]
                    new_data = pd.DataFrame(new_data).T
                    model.predict(new_data)
                    txt3.insert(END, str(model.predict(new_data)))

                btn = Button(m_window, text="Hesapla", command=lambda: get(float(txt1.get()), float(txt2.get()), float(txt4.get())))
                btn.place(x=400, y=305)


        veri.trace("w", callback)
        
        def back_first():
            m_window.withdraw()
            new_window.deiconify() 
        
        btnb2=Button(m_window,width=15,text="<- BACK",command=back_first)
        btnb2.config(font="Arial 30", fg="black")
        btnb2.place(x=1000,y=700)  
        #temizle button'u eklenecek.
        
    def own_file():
        messagebox.showinfo('Info', 'Under developing') 
    def back_first():
        new_window.withdraw()
        window.deiconify() 
        

    btnstr1=Button(new_window,width=30,text="Google Play Store App Values =>",command=game)
    btnstr1.config(font="Arial 30", fg="black")
    btnstr1.place(x=200,y=100)

    btnstr2=Button(new_window,width=30,text="Car Data Prediction =>",command=car)
    btnstr2.config(font="Arial 30", fg="black")
    btnstr2.place(x=200,y=300)

    btnstr3=Button(new_window,width=30,text="Use Your Own File =>",command=own_file)
    btnstr3.config(font="Arial 30", fg="black")
    btnstr3.place(x=200,y=500)  
    
    btnstr4=Button(new_window,width=15,text="<- BACK",command=back_first)
    btnstr4.config(font="Arial 30", fg="black")
    btnstr4.place(x=1000,y=700)  
    
    new_window.mainloop()
    

btnstr=Button(window,width=20,text="START",command=start)
btnstr.config(font="Arial 30", fg="black")
btnstr.place(x=150,y=400)
btnexit=Button(window,width=10,text=">>EXIT<<",command=window.destroy)
btnexit.config(font="Arial 30", fg="black")
btnexit.place(x=1000,y=700)


##############################################################

  
    

window.mainloop()



