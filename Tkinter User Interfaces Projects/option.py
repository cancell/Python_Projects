from tkinter import *


GUNLER = ["Pazartesi", "Salı", "Çarşamba", "Perşembe","Cuma","Cumartesi","Pazar"] 

pencere = Tk()

pencere.geometry('350x200')

veri= StringVar(pencere)
veri.set(GUNLER[0]) 

w = OptionMenu(pencere, veri, *GUNLER)
w.pack()

def ok():
    global label
    label.destroy()
    label=Label(text=veri.get())
    label.pack()
    
button = Button(pencere, text="OK", command=ok)
button.pack()
label=Label(text="")
label.pack()
mainloop()
