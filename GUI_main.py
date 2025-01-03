# Name: Aiman Adel Awadh Abdullah Mahmood
# TP: TP065994
# Project Title :
# Supervisor: Dr. Murugananthan Velayutham
# 2nd Marker: Assoc. Prof. Dr. Sivakumar Vengusamy
# Degree : B.Sc. Computer Science (Hons) (Artificial intelligence)

import os
import cv2
from tkinter import *
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import Vehicle_detection as od #Importing a custom module for vehicle detection

class GUIWindow(Frame):
    def __init__(self, window=None):
        Frame.__init__(self, window)

        # Initialize attributes
        self.window = window
        self.crosshair_positions = []
        self.line_positions = []
        self.rectangle_positions = []
        self.window.title("GUI")
        self.pack(fill=BOTH, expand=1)
        self.crosshair_counter = 0

        # Define menu and add options
        main_menu = Menu(self.window)
        self.window.config(menu=main_menu)

        main_menu.add_command(label="Import", command=self.open_file)  # Option to import a video file
        main_menu.add_cascade(label="Mark RoI", command=self.draw_region_of_interest)  # Option to mark Region of Interest
        main_menu.add_command(label="Exit", command=self.exit_client)  # Option to exit the program
        self.window.resizable(False, False)

        # Initial Home page image to show
        self.image_path = "resources/splash_screen.png"
        self.image_to_show = Image.open(self.image_path)
        self.photo_image = ImageTk.PhotoImage(self.image_to_show)
        self.canvas_width, self.canvas_height = (1366, 768)

        # Define canvas and add initial image
        self.canvas = Canvas(master=window, width=self.canvas_width, height=self.canvas_height)
        self.canvas = Canvas(master=window, width=self.canvas_width, height=self.canvas_height, bg="#12054d")
        self.canvas.create_image(20, 20, image=self.photo_image, anchor='nw')

        self.canvas.pack()

    def open_file(self):
        self.image_path = filedialog.askopenfilename()  # Open file dialog to select a video file
        video = cv2.VideoCapture(self.image_path)
        _, image = video.read()
        self.update_image(image)  # Update the canvas with the selected video frame

    def update_image(self, image):
        img_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_to_show = Image.fromarray(img_cv2)
        self.photo_image = ImageTk.PhotoImage(self.image_to_show)
        self.canvas_width, self.canvas_height = (1366, 768)

        # Destroy the previous canvas and create a new one
        self.canvas.destroy()
        self.canvas = Canvas(master=root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.create_image(0, 0, image=self.photo_image, anchor='nw')
        self.canvas.pack()

    def draw_region_of_interest(self):
        root.config(cursor="plus")
        self.canvas.bind("<Button-1>", self.record_crosshair_position)  # Bind left mouse click to record crosshair position

    def exit_client(self):
        exit()  # Exit the program

    def record_crosshair_position(self, event):
        if self.crosshair_counter < 2:
            x = int(self.canvas.canvasx(event.x))
            y = int(self.canvas.canvasy(event.y))
            self.line_positions.append((x, y))
            self.crosshair_positions.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
            self.crosshair_positions.append(self.canvas.create_line(x, y - 5, x, y + 5, fill="red", tags="crosshair"))
            self.crosshair_counter += 1

        if self.crosshair_counter == 2:
            self.canvas.unbind("<Button-1>")  # Unbind left mouse click
            root.config(cursor="arrow")
            self.crosshair_counter = 0

            self.draw_line_and_process_image()  # Draw line on the canvas and process the image

            # Clearing things
            self.line_positions.clear()
            self.rectangle_positions.clear()
            for i in self.crosshair_positions:
                self.canvas.delete(i)

    def draw_line_and_process_image(self):
        img = np.array(self.image_to_show)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.line(img, self.line_positions[0], self.line_positions[1], (0, 255, 0), 3)  # Draw a line on the image
        self.update_image(img)  # Update the canvas with the modified image

        self.detect_traffic_violation()  # Detect traffic violations using the marked Region of Interest

    def detect_traffic_violation(self):
        video_src = self.image_path
        video = cv2.VideoCapture(video_src)
        frame_counter = 1

        while True:
            ret, image = video.read()
            if image is None:
                break
            image_h, image_w, _ = image.shape
            new_image = od.preprocess_input(image, od.net_h, od.net_w)
            yolos = od.yolov3.predict(new_image)
            boxes = []

            for i in range(len(yolos)):
                boxes += od.decode_netout(yolos[i][0], od.anchors[i], od.obj_thresh, od.net_h, od.net_w)
            od.correct_yolo_boxes(boxes, image_h, image_w, od.net_h, od.net_w)
            od.do_nms(boxes, od.nms_thresh)
            image2 = od.draw_boxes(image, boxes, self.line_positions, od.labels, od.obj_thresh, frame_counter)
            cv2.imshow('Traffic Violation', image2)
            cv2.waitKey(1)
            print(frame_counter)
            frame_counter += 1
        cv2.destroyAllWindows()


if not os.path.exists('violations'):
    os.makedirs('violations')  # Create a directory called 'violations' if it doesn't exist

root = Tk()  # Create the Tkinter root window
gui_window = GUIWindow(root)  # Create an instance of the GUIWindow class, passing the root window
root.geometry("1330x770")  # Set the size of the root window
root.title("Traffic Signal Violation Detection System Simulation")  # Set the title of the root window

root.mainloop()  # Start the Tkinter event loop, which handles user interactions and updates the GUI

