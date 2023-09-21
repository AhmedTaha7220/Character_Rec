import tensorflow as tf
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
import glob as gb
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
model = tf.keras.models.load_model('speech.model')
import librosa #for audio processing
#import IPython.disp1ay as ipd
import numpy as np
import os
train_audio_path = 'data'
classes=os.listdir(train_audio_path)
def predict(audio):
prob=model.predict(audio. reshape (1, 8000, 1) )
index=np. argmax (prob[0])
return classes[index]
#reading the voice
def take_path(p):
samples, sample_rate = librosa.load( p, sr = 8000)
samples= librosa.resample(samples , orig_sr=sample_rate,target_sr=8000)
#ipd. Audio(samples, rate=8000)
return (predict(samples))
#take_path('F:/Datasets/Speech/data/five/00b01445_nohash_0.wav')
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
global_result = None
# Function to browse for an image file and display it in a label
def browse_audio():
global global_result
# Open a file dialog box to select an image file
file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3;*.wav;*.ogg")])
global_result = file_path
# Functions to be activated by the buttons
def function1():
lbl['text']=take_path(global_result)
lbl.pack()
# Create a Tkinter window with a blue background
# #Adding Background image
root = Tk()
root.title("Neural Networks With Health")
root.geometry("1200x600")
from tkinter import ttk
img = PhotoImage(file="D:\\a.png")
img = img.zoom(4,4)
lbl = Label(root,image = img).place(x=0,y=0)
# Create a button to browse for an image
browse_button = Button(root, text="Browse Your Audio File",font=("Arial",26), command=browse_audio)
browse_button.pack()
# Create buttons to activate the functions
function1_button = Button(root, text="Predict",font=("Arial",26), command=function1)
function1_button.pack()
lbl = Label(root, text="Browse Your Audio File",foreground="white",background="blue",font=("Arial",26))
# Run the Tkinter event loop
root.mainloop()