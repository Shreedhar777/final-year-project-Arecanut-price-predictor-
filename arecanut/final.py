import os
import cv2
from PIL import ImageTk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image
import newfinal

cwd = os.getcwd()


root = Tk()
frame = Frame(root)
frame.pack()


global price,d1,d2,d3,d4,d5
image = Image.open("wall.jpg")
photo = ImageTk.PhotoImage(image)

label = Label(root, image=photo)
label.image = photo  # keep a reference!
label.pack()

label1 = Label(root, text='Arecanut Price Prediction',font=("Arial", 20),fg="blue")
label1.pack()
# Dropdown menu options

label2 = Label(root, text='',font=("Arial", 20),fg="blue")
label2.pack()

label3 = Label(root, text='',font=("Arial", 20),fg="blue")
label3.pack()


# def sms(mssg,number):
#         import requests
#         url = "https://www.fast2sms.com/dev/bulk"
#         payload = "sender_id=FSTSMS&message="+mssg+"&language=english&route=p&numbers="+number
#         headers = {
#         'authorization': "HTpcZ6wKo1ChfI45MWesQ3GbRtka7njgiSYqXxPdm0UyJvL9AOvESNr7YW6u1G3TRD480igxhCbe9FnK",
#         'Content-Type': "application/x-www-form-urlencoded",
#         'Cache-Control': "no-cache",
#         }
#         response = requests.request("POST", url, data=payload, headers=headers)
#         print(response.text)




def output():
    price,date=newfinal.main()
    d1=str(date[0])
    d2=str(date[1])
    d3=str(date[2])
    d4=str(date[3])
    d5=str(date[4])
    d1.replace("Timestamp","").replace("0","").replace("(","").replace(")","")
    d1=d1[:10]
    d2.replace("Timestamp","").replace("0","").replace("(","").replace(")","")
    d2=d2[:10]
    d3.replace("Timestamp","").replace("0","").replace("(","").replace(")","")
    d3=d3[:10]
    d4.replace("Timestamp","").replace("0","").replace("(","").replace(")","")
    d4=d4[:10]
    d5.replace("Timestamp","").replace("0","").replace("(","").replace(")","")
    d5=d5[:10]
    print(d1,d2,d3,d4,d5)
    class Table:
      
        def __init__(self,root):
              
            # code for creating table
            for i in range(total_rows):
                for j in range(total_columns):
                      
                    self.e = Entry(root1, width=20, fg='blue',
                                   font=('Arial',16,'bold'))
                      
                    self.e.grid(row=i, column=j)
                    self.e.insert(END, lst[i][j])
  
    # take the data
    lst = [(1,d1,price[0]),
           (2,d2,price[1]),
           (3,d3,price[2]),
           (4,d4,price[3]),
           (5,d5,price[4])]
       
    # find total number of rows and
    # columns in list
    total_rows = len(lst)
    total_columns = len(lst[0])
       
    # create root window
    root1 = Tk()
    root1.title("predictions")
    #root1 = Toplevel(root)
    t = Table(root1)



def modify2():
    os.system("predictionempty.xls")




       
def callback():
    global price,d1,d2,d3,d4,d5
    price,date=newfinal.main()
    d1=str(date[0])
    d2=str(date[1])
    d3=str(date[2])
    d4=str(date[3])
    d5=str(date[4])
    d1.replace("Timestamp","").replace("0","").replace("(","").replace(")","")
    d1=d1[:10]
    d2.replace("Timestamp","").replace("0","").replace("(","").replace(")","")
    d2=d2[:10]
    d3.replace("Timestamp","").replace("0","").replace("(","").replace(")","")
    d3=d3[:10]
    d4.replace("Timestamp","").replace("0","").replace("(","").replace(")","")
    d4=d4[:10]
    d5.replace("Timestamp","").replace("0","").replace("(","").replace(")","")
    d5=d5[:10]
    print(d1,d2,d3,d4,d5)
    res="Predicted price is "+str(price)
    res="Predicted date is "+str(date)
    #label2.config( text = res)
    # score = randomf.main()
    
    image = Image.open("final.jpg")
    newsize = (400, 200)
    image = image.resize(newsize)
    photo = ImageTk.PhotoImage(image)
    label3.configure(image=photo)
    label3.image = photo

    # label1.configure(text=output,fg="green",font=("Arial", 20))
    # label1.text=output


    
root.title('Arecanut Price Prediction')
root.geometry('700x700') # Size 200, 200
root.resizable(width=True, height=True)



Button(text='Predict algorithm', command=callback,width=40,height=2,fg='green').pack(side=TOP, padx=10,pady=10)
Button(text='Predict date series', command=output,width=40,height=2,fg='red').pack(side=TOP, padx=10,pady=10)
Button(text='ADD Dates For Input', command=modify2,width=40,height=2,fg='blue').pack(side=TOP, padx=10,pady=10)

root.mainloop()
