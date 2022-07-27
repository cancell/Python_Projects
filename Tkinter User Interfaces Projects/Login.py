from tkinter import *

from tkinter import messagebox 
window = Tk()  #Tk class name'i
 
window.title("Welcome My App")
 
window.geometry('350x200') #pencere boyutunu ayarlar.
 
lbl = Label(window, text="Name") #label oluşturur.
lbl2 = Label(window, text="Surname") #label oluşturur.

lbl.grid(column=0, row=0)  #label boyutunu ayarlar.
lbl2.grid(column=0, row=1)  #label boyutunu ayarlar.

txt = Entry(window, width=10)
txt.grid(column=1, row=0)
txt2 = Entry(window, width=10)
txt2.grid(column=1, row=1)

def clicked():
    if txt.get()=="cansel" and txt2.get()=="küçük":
        messagebox.showinfo('Giriş Sayfası', 'Başarılı şekilde giriş yaptınız')
    else:
        messagebox.showinfo('Giriş Sayfası', 'Yanlış giriş yaptınız')
    
btn = Button(window, text="Click Me",command=clicked)

btn.grid(column=2, row=0)
 
window.mainloop()
