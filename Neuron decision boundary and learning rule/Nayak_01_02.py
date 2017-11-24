# Nayak, Anil Kumar



import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
class DisplayActivationFunctions:


    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root

        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.weight = []
        self.input_weight_w1 = 1
        self.input_weight_w2 = 1
        self.bias = 0
        self.activation_function = "Symmetrical Hard limit"
        self.ts = []
        self.points = []

        self.x_values = []
        self.y_values = []

        self.plotpoint = False
        self.draw_color_boundary = False
        self.numberOfPoint_no = 4
        self.training_started = False
        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=3, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('')
        self.axes.set_ylabel('')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=0, column=3) #, sticky=tk.N + tk.E + tk.S + tk.W
        self.sliders_frame.rowconfigure(0, weight=1)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='xx')

        # set up the sliders w1
        self.input_weight_slider_w1 = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",length=200,
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight W1",
                                            command=lambda event: self.input_weight_slider_callback_w1())
        self.input_weight_slider_w1.set(self.input_weight_w1)
        self.input_weight_slider_w1.bind("<ButtonRelease-1>", lambda event: self.input_weight_slider_callback_w1())
        self.input_weight_slider_w1.grid(row=0, column=0) #, sticky=tk.N + tk.E + tk.S + tk.W

        # set up the sliders w2
        self.input_weight_slider_w2 = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",length=200,
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight W2",
                                            command=lambda event: self.input_weight_slider_callback_w2())
        self.input_weight_slider_w2.set(self.input_weight_w2)
        self.input_weight_slider_w2.bind("<ButtonRelease-1>", lambda event: self.input_weight_slider_callback_w2())
        self.input_weight_slider_w2.grid(row=1, column=0) #, sticky=tk.N + tk.E + tk.S + tk.W

        self.bias_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",length=200,
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Bias",
                                    command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.bias_slider_callback())
        self.bias_slider.grid(row=2, column=0) #, sticky=tk.N + tk.E + tk.S + tk.W

        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################
        self.label_for_activation_function = tk.Label(self.sliders_frame, text="Activation Function", justify="center")
        self.label_for_activation_function.grid(row=4, column=0) #, sticky=tk.N + tk.E + tk.S + tk.W

        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.sliders_frame, self.activation_function_variable,
                                                          "Symmetrical Hard limit", "Hyperbolic Tangent","Linear",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_dropdown.config(width=20)
        self.activation_function_variable.set(self.activation_function)
        self.activation_function_dropdown.grid(row=5, column=0) #, sticky=tk.N + tk.E + tk.S + tk.W

        self.draw_button_generate = tk.Button(self.sliders_frame, text="Generate Random Data", fg="red", width=16, command=self.toolbar_draw_callback_generate)
        self.draw_button_generate.grid(row=6, column=0)

        self.draw_button_train = tk.Button(self.sliders_frame, text="Train", fg="red", width=16, command=self.toolbar_draw_callback_train)
        self.draw_button_train.grid(row=7, column=0)

        #self.display_activation_function()
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())




    def input_weight_slider_callback_w1(self):
        self.input_weight_w1 = self.input_weight_slider_w1.get()
        if self.training_started:
            self.toolbar_draw_callback_train()

    def input_weight_slider_callback_w2(self):
        self.input_weight_w2 = self.input_weight_slider_w2.get()
        if self.training_started:
            self.toolbar_draw_callback_train()


    def bias_slider_callback(self):
        self.bias = self.bias_slider.get()
        if self.training_started:
            self.toolbar_draw_callback_train()

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        if self.training_started:
            self.toolbar_draw_callback_train()

    def find_activation(self,net_value,isarray):
        if self.activation_function == "Sigmoid":
            activation = 1.0 / (1 + np.exp(-net_value))
        elif self.activation_function == "Linear":
            activation = net_value
        elif self.activation_function == "Symmetrical Hard limit":
            if isarray:
                activation = net_value
                activation[activation < 0] = -1
                activation[activation >= 0] = 1
            else:
                if net_value < 0:
                    activation = -1
                else:
                    activation = 1
        elif self.activation_function == "Hyperbolic Tangent":
            activation = (np.exp(net_value) - np.exp(-net_value)) / (np.exp(net_value) + np.exp(-net_value))

        return activation

    def reset_detail(self):
        self.activation_function = self.activation_function_variable.get()
        self.bias = self.bias_slider.get()
        self.input_weight_w2 = self.input_weight_slider_w2.get()
        self.input_weight_w1 = self.input_weight_slider_w1.get()

    def reset_points(self):
        self.root.status_bar.set('%s', " ")
        self.points = []
        self.ts = []
        self.plotpoint = False
        self.draw_color_boundary = False

    def toolbar_draw_callback_train(self):
        self.training_started = True
        self.root.status_bar.set('%s', "Training has started")
        self.reset_detail()

        if self.plotpoint:

            self.weight = np.array([self.input_weight_w1, self.input_weight_w2])  # [w1,w2]
            no_of_point = len(self.points)

            for i in range(100):

                for point in range(no_of_point):
                    p = np.array(self.points[point]) # [x,y]
                    net_value = np.sum(self.weight * p) + self.bias  # 1x2 * 2x1
                    activation = self.find_activation(net_value,False)
                    error = self.ts[point]-activation
                    self.weight = self.weight + error * p
                    self.bias = self.bias + error

                if i % 98 == 0:
                    self.x_values = np.linspace(-10,10,10)
                    self.y_values = ((- self.weight[0] * self.x_values) - self.bias ) / self.weight[1]

                    self.draw_color_boundary = True
                    self.plot_details()

            self.root.status_bar.set('%s', "Training Completed with New W1 : [" + str(self.weight[0]) + "],  W2 : [" + str(self.weight[1]) + "], Bias : [" + str(self.bias) + "]")
        else:
            self.root.status_bar.set('%s', "Provide / Click Generate Data before Training")

    def toolbar_draw_callback_generate(self):

        self.points = []
        self.ts = []
        for i in range(0,self.numberOfPoint_no):
            point = np.random.rand(2) * (self.xmax - self.xmin) + self.xmin
            if i%2==0:
                t = 1
            else:
                t = -1
            self.points.append(point)  # [[x1,y1],[x2,y2],[]]
            self.ts.append(t)
        self.plotpoint = True
        self.plot_details()
        self.root.status_bar.set('%s', "Random Points Generated, Ready for Training")


    def plot_details(self):
        self.axes.cla()
        self.axes.cla()

        if self.plotpoint:
            for i in range(self.numberOfPoint_no):
                if self.ts[i] == -1:
                    self.axes.plot(self.points[i][0], self.points[i][1], 's',color="black")
                else:
                    self.axes.plot(self.points[i][0], self.points[i][1], 'r^',color="blue")

        self.axes.plot(self.x_values, self.y_values)
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)


        if self.draw_color_boundary:

            green_boundary = 10
            red_boundary = -10
            number_of_neg_valid = []
            number_of_pos_valid = []

            a = []
            b = []

            a.append(self.x_values[0])
            a.append(self.y_values[0])
            b.append(self.x_values[1])
            b.append(self.y_values[1])

            for i in range(self.numberOfPoint_no):
                x = self.points[i][0]
                y = self.points[i][1]

                above_left = self.find_above_left_of_the_line(a,b,x,y)

                if self.ts[i] == -1:
                    number_of_neg_valid.append(above_left)
                else:
                    number_of_pos_valid.append(above_left)

            pos = np.array(number_of_pos_valid).all()
            neg = np.array(number_of_neg_valid).all()

            if (pos == True and neg == False):
                green_boundary = 10
                red_boundary = -10
            else:
                green_boundary = -10
                red_boundary = 10

            plt.fill_between(self.x_values, self.y_values, red_boundary, color="red")  # Gray 10
            plt.fill_between(self.x_values, self.y_values, green_boundary, color="green")  # Blue -10


        plt.title(self.activation_function)
        self.canvas.draw()

    def find_above_left_of_the_line(self,a, b, cx,cy):
        return ((b[0] - a[0]) * (cy - a[1]) - (b[1] - a[1]) * (cx - a[0])) > 0

    def plot_details_activation_function(self):
        self.x_values = np.linspace(-10, 10, 256, endpoint=True)
        self.y_values = np.linspace(-10, 10, 256, endpoint=True)
        net_value = self.input_weight_w1 * self.x_values + self.input_weight_w2 * self.y_values + self.bias
        activation = self.find_activation(net_value, True)

        self.y_values = activation

        self.axes.cla()
        self.axes.cla()

        self.axes.plot(self.x_values, self.y_values)
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)

        plt.title(self.activation_function)
        self.canvas.draw()

