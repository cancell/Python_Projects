import time
from tkinter import *
from tkinter import messagebox #ekrana uyarı verdiriyor.

window = Tk()

window.title("Facebook Login")
window.geometry("350x200")
window.configure(background="red")

lbl1 = Label(window, text="Username", bg = "pink", fg= "gray", font = "Times 24 bold")
lbl2 = Label(window, text="Password", bg = "pink", fg= "gray", font = "Times 24 bold")

lbl1.place(x=60, y=40)
lbl2.place(x=60, y=80)

txt1 = Entry(window, width=10)
txt2= Entry(window, width=10, show="*")

txt1.place(x=200, y=40)
txt2.place(x=200, y=80)

def clicked():
    if txt1.get() =="Cansel".lower() and txt2.get()=="1234":
        print("Sayfa yükleniyor....")
        time.sleep(5)
        messagebox.showinfo('Home Page', 'You entered')
    else:
        messagebox.showinfo('Home Page', 'Wrong password. Please try again.')

def new_winF(): # new window definition
    newwin = Toplevel(window)
    newwin.geometry('350x200')
    newwin.title("New Window")
    display = Label(newwin, text="Humm, see a new window !")
    display.pack()    
btn = Button(window, text="Sign In", command=clicked, bg = "pink", fg= "gray", font = "Times 24 bold")

btn.place(x=60, y=120)

button1 =Button(window, text ="open new window", command =new_winF) #command linked
button1.place(x=200, y=120)


window.mainloop()

