# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:24:48 2018

@author: varinder
"""

import tkinter as tk
from tkinter.colorchooser import askcolor
from PIL import Image , ImageDraw, ImageChops , ImageOps
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
import math
class Paint(object):
    
    DEFAULT_PEN_SIZE = 2.0
    DEFAULT_COLOR = 'black'
    MODEL=None
    
    def __init__(self,model):
        
        self.root = tk.Tk()
        self.MainFrame=tk.Frame(self.root)
        self.UpperFrame=tk.Frame(self.MainFrame)
        self.MidFrame1=tk.Frame(self.MainFrame)
        self.MidFrame2=tk.Frame(self.MainFrame)
        self.LowerFrame=tk.Frame(self.MainFrame)
        self.sc=tk.Scrollbar(self.MainFrame)
        white = (255, 255, 255)
        self.image1=Image.new("RGB", (600,600), white)
        self.draw=ImageDraw.Draw(self.image1)
        self.MODEL=model
        topLabel=tk.Label(self.UpperFrame, text="OpticalCalculator" , font=("Comic Sans MS", "16"))
        topLabel.pack()
        
        textLabel=tk.Label(self.MidFrame2, text="Converted Text")
        textLabel.pack(padx=10,pady=10)
        self.text=tk.Text(self.MidFrame2 , height=23,width=40)
        self.text.pack(padx=5,pady=5)
        
        self.pen_button = tk.Button(self.MidFrame1, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)
        
        self.brush_button = tk.Button(self.MidFrame1, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)
        
        self.color_button = tk.Button(self.MidFrame1, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)
        
        self.eraser_button = tk.Button(self.MidFrame1, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)
        
        self.choose_size_button = tk.Scale(self.MidFrame1, from_=1, to=10, orient=tk.HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)
        
        self.c = tk.Canvas(self.MidFrame1, bg='white', width=400, height=400)
        self.c.grid(row=1, columnspan=5)
        
        self.photoclr=tk.PhotoImage(file="resize.png")
        self.clearButton = tk.Button(self.MidFrame1 , text='clear' , command= self.clear)
        self.clearButton.config(image=self.photoclr,width="150",height="30")
        self.clearButton.grid(row=2 , column =0 , columnspan=2 , padx=5,pady=5)
        
        self.photonxt=tk.PhotoImage(file="print.png")
        self.nextButton = tk.Button(self.MidFrame1 , text='print character' , command= self.getter)
        self.nextButton.config(image=self.photonxt,width="150",height="30")
        self.nextButton.grid(row=2 , column = 3, columnspan=2 , padx=5,pady=5)
        
        self.photok=tk.PhotoImage(file="solve.png")
        self.OKButton=tk.Button(self.MidFrame2, text='Solve the equation', command=self.solveeq)
        self.OKButton.config(image=self.photok,width="150",height="30")
        self.OKButton.pack(padx=5,pady=5 )
        
        self.sc.pack(side = tk.RIGHT, fill = tk.Y)
        self.UpperFrame.pack(side=tk.TOP,padx=10,pady=10)
        self.MidFrame1.pack(side=tk.LEFT,padx=10,pady=10)
        self.MidFrame2.pack(side=tk.RIGHT,padx=10,pady=10)
        self.LowerFrame.pack(side=tk.BOTTOM,padx=10,pady=10)
        self.MainFrame.pack()
        self.setup()
        self.root.mainloop()
    
    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
    
    
    def use_pen(self):
        self.activate_button(self.pen_button)
    
    def use_brush(self):
        self.activate_button(self.brush_button)
    
    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]
    
    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)
    
    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=tk.RAISED)
        some_button.config(relief=tk.SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode
    
    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
            self.draw.line([self.old_x, self.old_y, event.x, event.y,], paint_color, width=10)
        
        self.old_x = event.x
        self.old_y = event.y
    
    def reset(self, event):
        self.old_x, self.old_y = None, None
    
    def getter(self):
        self.c.delete('all')
        filename = "test.jpg"
        self.image1.save(filename)
        self.cropimg()
        printstr=self.nueralnet()
        self.text.insert(tk.END, printstr)
        
    def cropimg(self):
        def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
            dim = None
            (h, w) = image.shape[:2]
            if width is None and height is None:
                return image
            if width is None:
                r = height / float(h)
                dim = (int(w * r), height)
            else:
                r = width / float(w)
                dim = (width, int(h * r))
            resized = cv2.resize(image, dim, interpolation = inter)
            return resized
        def resize_with_pad(image, height=45, width=45):

            def get_padding_size(image):
                h, w, _ = image.shape
                longest_edge = max(h, w)
                top, bottom, left, right = (0, 0, 0, 0)
                if h < longest_edge:
                    dh = longest_edge - h
                    top = dh // 2
                    bottom = dh - top
                elif w < longest_edge:
                    dw = longest_edge - w
                    left = dw // 2
                    right = dw - left
                else:
                    pass
                return top, bottom, left, right

            top, bottom, left, right = get_padding_size(image)
            WHITE = [255, 255, 255]
            constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=WHITE)

            resized_image = cv2.resize(constant, (height, width))

            return resized_image





        def trim(im):
            bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
            diff = ImageChops.difference(im, bg)
            diff = ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox()
            if bbox:
                return im.crop(bbox)

        im = Image.open("test.jpg").convert('RGB')
        im = trim(im)
        im=np.asarray(im)
        im=image_resize(im,45)
        im=resize_with_pad(im)
        im=Image.fromarray(im)
        
        
        im.save("test.jpg")
        white=(255,255,255)
        self.image1 = Image.new("RGB", (600,600), white)
        self.draw=ImageDraw.Draw(self.image1)

        
    def clear(self):
        self.c.delete('all')
     #   self.text.delete(1.0,tk.END)
        white=(255,255,255)
        self.image1 = Image.new("RGB", (600,600), white)
        self.draw=ImageDraw.Draw(self.image1)
    
    def solveeq(self):
        eq = self.text.get(1.0, "end-1c")
        try:
            s=eq
            x=0
            index=None;
            index2=None;
            if '!'in s:
                while x<len(s):
                    if(s[x]=='!'):    
                        index=x
                        break
                    x=x+1
                while x>0:
                    x=x-1
        
                if(0<int(s[x])<9):
                    index2=x

                y=int(s[index2:index])

                fact=1
                while(y>1):
                    fact=fact*y
                    y=y-1
                a=fact
            else:
                a=eval(eq)
            self.text.delete(1.0,tk.END)
            self.text.insert(tk.END, a)
        except:
            self.text.delete(1.0,tk.END)
            self.text.insert(tk.END, 'the equation : %s is incomplete or wrong' %(eq))
    
    
    def nueralnet(self):
        img=Image.open('test.jpg').convert('L') 
        img=np.asarray(img)
        img=img.reshape(-1,45,45,1)
        
        hello=self.MODEL.predict(img)
        p=np.argmax(hello,axis=1)


        if(p==0):
            s='!'
        elif(p==1):
            s='('
        elif(p==2):
            s=')'
        elif(p==3):
            s='+'
        elif(p==4):
            s='-'
        elif(p==5):
            s='0'
        elif(p==6):
            s='1'
        elif(p==7):
            s='2'
        elif(p==8):
            s='3'
        elif(p==9):
            s='4'
        elif(p==10):
            s='5'
        elif(p==11):
            s='6'
        elif(p==12):
            s='7'
        elif(p==13):
            s='8'
        elif(p==14):
            s='9'
        elif(p==15):
            s='='
        elif(p==16):
            s='['
        elif(p==17):
            s=']'
        elif(p==18):
            s='cos'
        elif(p==19):
            s='/'
        elif(p==20):
            s='>='
        elif(p==21):
            s='>'
        elif(p==22):
            s='<='
        elif(p==23):
            s='log'
        elif(p==24):
            s='<'
        elif(p==25):
            s='3.14'
        elif(p==26):
        
            s='+-'
        elif(p==27):
            s='sin'
        elif(p==28):
            s='math.sqrt'
        elif(p==29):
            s='summition'
        elif(p==30):
            s='tan'
        elif(p==31):
            s='*'
        elif(p==32):
            s='{'
        elif(p==33):
            s='}'

        return s





    '''    for sign,val in oi.items():
            if val==p:
                s=sign
'''

if __name__ == '__main__':
    model = load_model('modelfinalvarpc.h5')
    Paint(model)
    
    
'''model.evaluate_generator(test_set)'''
