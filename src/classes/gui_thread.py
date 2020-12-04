import time
import threading
import queue

from PIL import Image, ImageTk 
import tkinter as tk 
from tkinter import Canvas, Frame

import PIL
import PIL.Image
import PIL.ImageTk



class GUIThread(threading.Thread):
    def __init__(self, queue=queue.Queue(), args=(), kwargs=None):
        
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.setDaemon(True)
        self.queue = queue
        self.canvas = None
        self.tk_img_id = None
        self.tk_image = None
        self.tk_root = None

    def stop(self):
        if self.tk_root:
            self.tk_root.quit()

    def run(self):
        self.tk_root = tk.Tk()
        #root.update()

        #self.frame = Frame(tk_root)
        #self.frame.pack(fill='both', expand=True)

        width=600
        height = 600

        self.tk_root.resizable(width=False, height=False)

        #if self.tk_image:
        #    width=self.tk_image.width()
        #    height=self.tk_image.height()

        self.canvas = Canvas(self.tk_root, width=width, height=height)
        self.canvas.pack(fill='both', expand=False)
        self.tk_root.mainloop()
        print("GUIThread ending!")
        #import IPython
        #IPython.embed()
        return

    @staticmethod
    def parseImage(image):
        """
            Check image dimensions and resize Pillow image
        """
        #resize image
        #1280x800
        #MAX_WIDTH, MAX_HEIGHT = 1280, 800
        MAX_WIDTH, MAX_HEIGHT = 1000, 600

        width, height = image.size
        if width > MAX_WIDTH:
            new_width  = MAX_WIDTH
            new_height = int(new_width * (height / width))
            image = image.resize((new_width, new_height), Image.ANTIALIAS)

        width, height = image.size
        if height > MAX_HEIGHT:
            new_height  = MAX_HEIGHT
            new_width = int(new_height * (width / height))
            image = image.resize((new_width, new_height), Image.ANTIALIAS)

        return image



    def addNewImage(self, image):
        if self.tk_img_id:
            self.canvas.delete( self.tk_img_id )

        image = GUIThread.parseImage(image)
        self.tk_image = ImageTk.PhotoImage(image)

        w,h = image.size
        self.tk_img_id = self.canvas.create_image((w/2,h/2), image=self.tk_image)
        self.canvas.config(width=w, height=h)
        self.canvas.pack(fill='both', expand=True)
        self.tk_root.geometry('{}x{}'.format(w, h))

