import cv2
import numpy as np
import tkinter as tk
import os

from tkinter import * 
from tkinter import messagebox 
from threading import Thread

stop = False
def camera():
    net = cv2.dnn.readNet('weight.weights', 'cfg.cfg')
    classes = ["Helmet","No Helmet"]

    try:
        cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(3,800)
        cap.set(4,600)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(100, 3))

        while True:
            _, img = cap.read()
            height, width, _ = img.shape


            blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)

                        x = int(center_x - w/2)
                        y = int(center_y - h/2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

            if len(indexes)>0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    color = colors[i]
                    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, color, 2)

            cv2.imshow("A.S.H.D.S", img)
            if stop or cv2.waitKey(1)==27:
                break
    except:
        tk.messagebox.showerror(title="A.S.H.D.S", message="Camera not found or it's look like other app using camera.\nPlease plug in camer or close other app.")        

    cap.release()
    cv2.destroyAllWindows()


#####  about button command
def about():     
    about_win = tk.Toplevel()
    about_win.iconbitmap(r"icons/icon.ico")
    about_win.geometry('300x400')
    about_win.resizable(width=False,height=False)
    about_win.title("ABOUT")
    
    about_img = tk.PhotoImage(file='icons/icon.png')    
    about_con = tk.Label(about_win,image=about_img)
    about_con .image = about_img
    about_con.pack(side=tk.TOP,pady=10)

    about_title = tk.Label(about_win,text="A.S.H.D.S\n")               
    about_title.pack(side=tk.TOP,pady=10)
    
    about_des = tk.Label(about_win,text="Version : 1.0.0\n \
DEVELOPED By :\n SHUBHAM MAURYA & YOGESH VERMA")               
    about_des.pack(fill=tk.X)

def quit():
    stop = True
    try:
        root.destroy()
    except:
        pass
##################################################################################################################################
############# GUI PROGRAMING  #############

################# ROOT SETTING 
root = Tk()
root.geometry('700x400')
root.title("A.S.H.D.S")
root.configure(bg="#F0F0F0") 
root.resizable(width=False,height=False)
root.iconbitmap(r"icons/icon.ico")

################# Logo
logo = tk.PhotoImage(file='icons/icon5.png')
img = Label( image=logo)
img.image = logo
img.place(x=0, y=0)

################ BUTTONS
##### start button
start_button_image = tk.PhotoImage(file='icons/run2.png')
start_button = tk.Button(root,width=200,height=70,bg="white",fg="green",text ="BEGIN",image=start_button_image, compound="left", command = camera)
start_button.place(x=400,y=50)

##### info button 
info_button_image = tk.PhotoImage(file='icons/info.png')
info_button = tk.Button(root,width=200,height=70,bg="white",fg="black",text ="ABOUT",image=info_button_image,compound="left",command = about)
info_button.place(x=400,y=150)

##### Quit button 
quit_button_image = tk.PhotoImage(file='icons/quit.png')
quit_button = tk.Button(root,width=200,height=70,bg="white",fg="red",text ="CLOSE",image=quit_button_image,compound="left",command = quit)
quit_button.place(x=400,y=250)

##### Note 
label = tk.Label( root,bg="white",fg="red",height=2,text="*Press 'Esc' to quit the camera window", relief=RAISED )
label.pack(side=tk.BOTTOM,fill=tk.X)

### ending 
root.mainloop()