# Nayak, Anil Kumar



# Widrow-Huff learning and adaptive filters

import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import Nayak_03_02 as N03  # This module is for plotting components


class WidgetsWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.minsize(self.winfo_screenwidth(), self.winfo_screenheight() - 100)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(2, weight=1)
        self.menu_bar = None
        self.tool_bar = None
        self.status_bar = StatusBar(self, self, bg='red', bd=1, relief=tk.SUNKEN)

        self.center_frame = tk.Frame(self)

        # # Create a frame for plotting graphs
        self.left_frame = PlotsDisplayFrame(self, self.center_frame, bg='white')
        self.display_activation_functions = N03.Adaline(self, self.left_frame)
        #
        self.center_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # self.center_frame.grid_propagate(True)
        # self.center_frame.rowconfigure(1, weight=1, uniform='xx')
        # self.center_frame.columnconfigure(0, weight=1, uniform='xx')
        #
        self.left_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # self.status_bar.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # self.status_bar.rowconfigure(1, minsize=30)


class StatusBar(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.label = tk.Label(self)
        self.label.grid(row=0, sticky=tk.N + tk.E + tk.S + tk.W)

    def set(self, format, *args):
        self.label.config(text=format % args)
        self.label.update_idletasks()

    def clear(self):
        self.label.config(text="")
        self.label.update_idletasks()


class PlotsDisplayFrame(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.root = root
        # self.configure(width=500, height=500)
        self.bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.bind("<ButtonRelease-1>", self.left_mouse_release_callback)
        self.bind("<B1-Motion>", self.left_mouse_down_motion_callback)
        self.bind("<ButtonPress-3>", self.right_mouse_click_callback)
        self.bind("<ButtonRelease-3>", self.right_mouse_release_callback)
        self.bind("<B3-Motion>", self.right_mouse_down_motion_callback)
        self.bind("<Key>", self.key_pressed_callback)
        self.bind("<Up>", self.up_arrow_pressed_callback)
        self.bind("<Down>", self.down_arrow_pressed_callback)
        self.bind("<Right>", self.right_arrow_pressed_callback)
        self.bind("<Left>", self.left_arrow_pressed_callback)
        self.bind("<Shift-Up>", self.shift_up_arrow_pressed_callback)
        self.bind("<Shift-Down>", self.shift_down_arrow_pressed_callback)
        self.bind("<Shift-Right>", self.shift_right_arrow_pressed_callback)
        self.bind("<Shift-Left>", self.shift_left_arrow_pressed_callback)
        self.bind("f", self.f_key_pressed_callback)
        self.bind("b", self.b_key_pressed_callback)

    def key_pressed_callback(self, event):
        self.root.status_bar.set('%s', 'Key pressed')

    def up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Up arrow was pressed")

    def down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Down arrow was pressed")

    def right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Right arrow was pressed")

    def left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Left arrow was pressed")

    def shift_up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift up arrow was pressed")

    def shift_down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift down arrow was pressed")

    def shift_right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift right arrow was pressed")

    def shift_left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift left arrow was pressed")

    def f_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "f key was pressed")

    def b_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "b key was pressed")

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y
        self.canvas.focus_set()

    def left_mouse_release_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was released. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def left_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse down motion. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_release_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse button was released. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def right_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y
        self.focus_set()


class GraphicsDisplayFrame(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.root = root
        self.master = master
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg='yellow')
        self.canvas.rowconfigure(0, weight=1)
        self.canvas.columnconfigure(0, weight=1)
        self.canvas.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.canvas.bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.canvas.bind("<ButtonRelease-1>", self.left_mouse_release_callback)
        self.canvas.bind("<B1-Motion>", self.left_mouse_down_motion_callback)
        self.canvas.bind("<ButtonPress-3>", self.right_mouse_click_callback)
        self.canvas.bind("<ButtonRelease-3>", self.right_mouse_release_callback)
        self.canvas.bind("<B3-Motion>", self.right_mouse_down_motion_callback)
        self.canvas.bind("<Key>", self.key_pressed_callback)
        self.canvas.bind("<Up>", self.up_arrow_pressed_callback)
        self.canvas.bind("<Down>", self.down_arrow_pressed_callback)
        self.canvas.bind("<Right>", self.right_arrow_pressed_callback)
        self.canvas.bind("<Left>", self.left_arrow_pressed_callback)
        self.canvas.bind("<Shift-Up>", self.shift_up_arrow_pressed_callback)
        self.canvas.bind("<Shift-Down>", self.shift_down_arrow_pressed_callback)
        self.canvas.bind("<Shift-Right>", self.shift_right_arrow_pressed_callback)
        self.canvas.bind("<Shift-Left>", self.shift_left_arrow_pressed_callback)
        self.canvas.bind("f", self.f_key_pressed_callback)
        self.canvas.bind("b", self.b_key_pressed_callback)

    def key_pressed_callback(self, event):
        self.root.status_bar.set('%s', 'Key pressed')

    def up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Up arrow was pressed")

    def down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Down arrow was pressed")

    def right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Right arrow was pressed")

    def left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Left arrow was pressed")

    def shift_up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift up arrow was pressed")

    def shift_down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift down arrow was pressed")

    def shift_right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift right arrow was pressed")

    def shift_left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift left arrow was pressed")

    def f_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "f key was pressed")

    def b_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "b key was pressed")

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y
        self.canvas.focus_set()

    def left_mouse_release_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was released. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def left_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse down motion. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_release_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse button was released. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def right_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y
        self.focus_set()

    def frame_resized_callback(self, event):
        print("frame resize callback")


# z = WidgetsWindow()
# z.mainloop()

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit, Hebbian Learning?"):
        root.destroy()


widgets_window = WidgetsWindow()

# widgets_window.geometry("500x500")
# widgets_window.wm_state('zoomed')
widgets_window.title('Assignment_03 -- Nayak -- Widrow-Huff learning and adaptive filters')
# widgets_window.minsize(screen_width,screen_height)
widgets_window.protocol("WM_DELETE_WINDOW", lambda root_window=widgets_window: close_window_callback(root_window))
widgets_window.mainloop()
