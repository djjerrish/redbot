#import modules
 
from tkinter import *
import os
from PIL import Image, ImageTk
 
# Designing window for registration
 
def register():
    global register_screen
    register_screen = Toplevel(main_screen)
    register_screen.title("Register")
    register_screen.geometry("400x300")
 
    global username
    global password
    global username_entry
    global password_entry
    username = StringVar()
    password = StringVar()
 
    Label(register_screen, text="Please enter details below",fg='white',bg="blue").pack()
    Label(register_screen, text="").pack()
    username_lable = Label(register_screen, text="Username",font=("Calibri", 20))
    username_lable.pack()
    username_entry = Entry(register_screen, textvariable=username)
    username_entry.pack()
    Label(register_screen, text="").pack()
    password_lable = Label(register_screen, text="Password",font=("Calibri", 20))
    password_lable.pack()
    password_entry = Entry(register_screen, textvariable=password, show='*')
    password_entry.pack()
    Label(register_screen, text="").pack()
    Button(register_screen, text="Register",font=("Calibri", 20),fg='white',bg="blue", command = register_user).pack()
 
 
# Designing window for login 
 
def login():
    global login_screen
    login_screen = Toplevel(main_screen)
    login_screen.title("Login")
    login_screen.geometry("400x300")
    Label(login_screen, text="Please enter details below to login",fg='white',bg="blue").pack()
    Label(login_screen, text="").pack()
 
    global username_verify
    global password_verify
 
    username_verify = StringVar()
    password_verify = StringVar()
 
    global username_login_entry
    global password_login_entry
 
    Label(login_screen, text="Username",font=("Calibri", 20)).pack()
    username_login_entry = Entry(login_screen,textvariable=username_verify)
    username_login_entry.pack()
    Label(login_screen, text="",width="20",height="1",font=("Calibri", 16)).pack()
    Label(login_screen, text="Password",font=("Calibri", 20)).pack()
    password_login_entry = Entry(login_screen,textvariable=password_verify, show= '*')
    password_login_entry.pack()
    Label(login_screen, text="").pack()
    Button(login_screen, text="Login",font=("Calibri", 20),bg="Blue",fg="white",command = login_verify).pack()
 
# Implementing event on register button
 
def register_user():
 
    username_info = username.get()
    password_info = password.get()
 
    file = open(username_info, "w")
    file.write(username_info + "\n")
    file.write(password_info)
    file.close()
 
    username_entry.delete(0, END)
    password_entry.delete(0, END)
 
    Label(register_screen, text="Registration Success", fg="green", font=("calibri", 11)).pack()
 
# Implementing event on login button 
 
def login_verify():
    global login_verify_screen
    username1 = username_verify.get()
    password1 = password_verify.get()
    username_login_entry.delete(0, END)
    password_login_entry.delete(0, END)
 
    list_of_files = os.listdir()
    if username1 in list_of_files:
        file1 = open(username1, "r")
        verify = file1.read().splitlines()
        if password1 in verify:
            login_sucess()
 
        else:
            password_not_recognised()
 
    else:
        user_not_found()
 
# Designing popup for login success
 
def login_sucess():
    global login_success_screen
    login_success_screen = Toplevel(login_screen)
    login_success_screen.title("Success")
    login_success_screen.geometry("300x200")
    Label(login_success_screen, text="Login Success",font=("Calibri", 20,'bold')).pack()
    Label(login_success_screen, text="").pack()
    Button(login_success_screen, text="OK",font=("Calibri", 20,'bold'),width=6, height=1,bg="Blue",fg="white", command=delete_login_success).pack()
    
 
# Designing popup for login invalid password
 
def password_not_recognised():
    global password_not_recog_screen
    password_not_recog_screen = Toplevel(login_screen)
    password_not_recog_screen.title("Unsuccessful")
    password_not_recog_screen.geometry("300x200")
    Label(password_not_recog_screen, text="Invalid Password ",font=("Calibri", 20,'bold')).pack()
    Label(password_not_recog_screen, text="").pack()
    Button(password_not_recog_screen, text="OK",font=("Calibri", 20,'bold'),width=6, height=1,bg="Blue",fg="white", command=delete_password_not_recognised).pack()
 
# Designing popup for user not found
 
def user_not_found():
    global user_not_found_screen
    user_not_found_screen = Toplevel(login_screen)
    user_not_found_screen.title("Unsuccessful")
    user_not_found_screen.geometry("300x200")
    Label(user_not_found_screen, text="User Not Found",font=("Calibri", 20,'bold')).pack()
    Label(user_not_found_screen, text="").pack()
    Button(user_not_found_screen, text="OK",font=("Calibri", 20,'bold'),width=6, height=1,bg="Blue",fg="white", command=delete_user_not_found_screen).pack()
 
# Deleting popups
 
def delete_login_success():
    login_success_screen.destroy()
    login_screen.destroy()
    main_screen.destroy()
 
 
def delete_password_not_recognised():
    password_not_recog_screen.destroy()
 
 
def delete_user_not_found_screen():
    user_not_found_screen.destroy()
 
def on_closing():
    exit()

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        load = Image.open("./images/logo1.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render  
        img.place(x=10, y=20)   

def main_account_screen():
    global main_screen
    main_screen = Tk()
    # main_screen.configure(bg='Black')
    main_screen.geometry("400x550")
    
    Label(text="RedBot Authentication", width="200", height="1", font=("Calibri", 30,'bold')).pack()
    app = Window(main_screen)
    main_screen.title("Account Login")
    Label(text="").pack()
    Label(text="Select Your Choice", bg="blue",fg='white',width="300", height="2", font=("Calibri", 13)).pack()
    Label(text="").pack()
    Button(text="Login", height="2", width="30",bd=5,font=("Calibri", 18), command = login).pack()
    Label(text="").pack()
    Button(text="Register", height="2", width="30",bd=5,font=("Calibri", 18), command=register).pack()
    Label(text="").pack()
    Label(text="").pack()
    Label(text="").pack()
    main_screen.protocol("WM_DELETE_WINDOW", on_closing)
    main_screen.mainloop()
 
 
main_account_screen()
