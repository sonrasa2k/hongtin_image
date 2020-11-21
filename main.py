import tkinter as tk
from PIL import Image, ImageTk
from tensorflow import keras
import cv2
import numpy as np

h = 0
model = keras.models.load_model('generator.h5')

inputs = keras.Input((None, None, 3))

outputs = model(inputs)

model = keras.models.Model(inputs, outputs)

path = []
# Create the master object
master = tk.Tk()

# Create the label objects and pack them using grid
tk.Label(master, text="Path Image To Conver ").grid(row=0, column=0)

# Create the entry objects using master
e1 = tk.Entry(master ,width = 40)
e2 = tk.Entry(master)

def show_img(path):
    # Create the PIL image object
    image = Image.open(path)
    image = image.resize((700,500))
    photo = ImageTk.PhotoImage(image)

    # Create an image label
    img_label = tk.Label(image=photo)
    img_label.grid_remove()
    # Store a reference to a PhotoImage object, to avoid it
    # being garbage collected! This is necesary to display the image!
    img_label.image = photo

    img_label.grid(row=0, column=2)

def get_entry_e1():
    global path
    path = e1.get()
    show_img(path)

def process():
    global h
    global path
    if path == "":
        get_entry_e1()
    else:
        low_res = cv2.imread(path, 1)
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)
        # Rescale to 0-1.
        low_res = low_res / 255.0

        # Get super resolution image
        sr = model.predict(np.expand_dims(low_res, axis=0))[0]

        # Rescale values in range 0-255
        sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

        # Convert back to BGR for opencv
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        path_out = str(h)+".png"
        h += 1
        cv2.imwrite(path_out, sr)
        print("complete")
        show_img(path_out)

# Pack them using grid
e1.grid(row=0, column=1)

button1 = tk.Button(master, text="Load Image",command=get_entry_e1)
button1.grid(columnspan=2, row=2, column=0)



# Create another button
button2 = tk.Button(master, text="Conver",command= process)
button2.grid(columnspan=2, row=2, column=2)




# The mainloop
master.geometry("1200x600+200+120")
tk.mainloop()