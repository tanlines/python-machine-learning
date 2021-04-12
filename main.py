import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision
import time
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import multiprocessing

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def process1(queue):
    wincap = WindowCapture('Old School Runescape')
    while True:
        queue.put(wincap.get_screenshot())

def process2(queue,display_queue):
    cascade = cv.CascadeClassifier('cascade.xml')
    vision = Vision()
    flip = 0
    rectangles = []
    while True:
        while not queue.empty():
            screenshot = queue.get()
            flip += 1
            if flip % 15 == 0:
                rectangles = cascade.detectMultiScale(
                    screenshot,
                    scaleFactor=1.1
                    #minNeighbors=3
                    #minSize=(40,40)
                    #flags=cv.CASCADE_SCALE_IMAGE
                )
            
            if len(rectangles) > 0:
                detection_image = vision.draw_rectangles(screenshot, rectangles)
            else:
                detection_image = screenshot
            display_queue.put(detection_image)

def process3(display_queue):
    while True:
        while not display_queue.empty():
            screenshot = display_queue.get()
            cv.imshow('Matches', screenshot)
            key = cv.waitKey(1)

if __name__ == "__main__":
    queue = multiprocessing.Queue()
    display_queue = multiprocessing.Queue()
    p1 = Process(target=process1, args=(queue,))
    p2 = Process(target=process2, args=(queue,display_queue,))
    p3 = Process(target=process3, args=(display_queue,))
    
    p1.start()
    p2.start()
    p3.start()