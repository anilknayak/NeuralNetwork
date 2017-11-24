# Nayak, Anil Kumar
# 1001-396-015
# 2017-10-08
# Assignment_03_02

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

np.seterr(divide='ignore')


class Adaline:
    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root

        #########################################################################
        # Widrow-Huff learning and adaptive filters Variable declaration
        #########################################################################
        # Model Values
        self.delayed_elements = 10
        self.learning_rate = 0.1
        self.training_sample_percentage = 80
        self.batch_size = 100
        self.iterations = 10
        self.data = None
        self.weights = None
        self.number_of_batch_to_process = 0
        self.number_of_output = 2
        self.number_of_input = 2
        self.number_of_records = 0
        self.number_of_training_sample = 0

        # Sample Training and Testing data
        self.training_sample = None
        self.training_sample_size = 0
        self.testing_sample = None
        self.testing_sample_size = 0

        # Graph Details
        # The limits of the error axes should be set between 0 and 2
        self.xmin = 0
        self.xmax = self.batch_size
        self.ymin = 0
        self.ymax = 2
        self.x_values = None
        self.y_values = None
        self.x_values1 = None
        self.y_values1 = None
        self.x_values2 = None
        self.y_values2 = None
        self.x_values3 = None
        self.y_values3 = None

        # Widget Details
        self.length_widget = 300
        self.file_name = 'stock_data.csv'

        # Network Details
        self.input = None
        self.target = None
        self.batch_data = None
        self.net_value = None
        self.error_price = None
        self.error_volume = None
        self.mean_square_error_price = []
        self.mean_square_error_volume = []
        self.mean_absolute_error_price = []
        self.mean_absolute_error_volume = []
        self.add_bias = True
        #########################################################################
        #  Set up the plotting area
        #########################################################################
        figure_size = (5, 4)
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0)
        self.figure = plt.figure(figsize=figure_size)
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Iterations')
        self.axes.set_ylabel('MSE for Price')
        self.axes.set_title("Mean Squared Error (MSE) for price")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0)

        self.plot_frame1 = tk.Frame(self.master)
        self.plot_frame1.grid(row=0, column=1)
        self.figure1 = plt.figure(figsize=figure_size)
        self.axes1 = self.figure1.gca()
        self.axes1.set_xlabel('Iterations')
        self.axes1.set_ylabel('MSE for Volume')
        self.axes1.set_title("Mean Squared Error (MSE) for volume")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=self.plot_frame1)
        self.plot_widget1 = self.canvas1.get_tk_widget()
        self.plot_widget1.grid(row=0, column=1)

        self.plot_frame2 = tk.Frame(self.master)
        self.plot_frame2.grid(row=1, column=0)
        self.figure2 = plt.figure(figsize=figure_size)
        self.axes2 = self.figure2.gca()
        self.axes2.set_xlabel('Iterations')
        self.axes2.set_ylabel('MAE for Price')
        self.axes2.set_title("Maximum Absolute Error (MAE) for price")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=self.plot_frame2)
        self.plot_widget2 = self.canvas2.get_tk_widget()
        self.plot_widget2.grid(row=0, column=1)

        self.plot_frame3 = tk.Frame(self.master)
        self.plot_frame3.grid(row=1, column=1)
        self.figure3 = plt.figure(figsize=figure_size)
        self.axes3 = self.figure3.gca()
        self.axes3.set_xlabel('Iterations')
        self.axes3.set_ylabel('MAE for Volume')
        self.axes3.set_title("Maximum Absolute Error (MAE) for volume")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas3 = FigureCanvasTkAgg(self.figure3, master=self.plot_frame3)
        self.plot_widget3 = self.canvas3.get_tk_widget()
        self.plot_widget3.grid(row=0, column=1)

        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=0, column=3)  # , sticky=tk.N + tk.E + tk.S + tk.W
        self.sliders_frame.rowconfigure(0, weight=2)
        self.sliders_frame.columnconfigure(0, weight=2, uniform='xx')

        # set up the sliders Number of Delayed Elements
        self.delayed_element_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                               from_=0, to_=100, resolution=1, bg="#DDDDDD", length=self.length_widget,
                                               activebackground="#FF0000",
                                               highlightcolor="#00FFFF",
                                               label="Number of Delayed Elements",
                                               command=lambda event: self.delayed_element_slider_callback())
        self.delayed_element_slider.set(self.delayed_elements)
        self.delayed_element_slider.bind("<ButtonRelease-1>", lambda event: self.delayed_element_slider_callback())
        self.delayed_element_slider.grid(row=0, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        # set up the sliders Learning Rate
        self.learning_rate_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                             length=self.length_widget,
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF",
                                             label="Learning Rate",
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=1, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        # set up the sliders Training Sample Size (Percentage)
        self.training_sample_size_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                                    from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                                    length=self.length_widget,
                                                    activebackground="#FF0000",
                                                    highlightcolor="#00FFFF",
                                                    label="Training Sample Size (Percentage)",
                                                    command=lambda event: self.training_sample_size_slider_callback())
        self.training_sample_size_slider.set(self.training_sample_percentage)
        self.training_sample_size_slider.bind("<ButtonRelease-1>",
                                              lambda event: self.training_sample_size_slider_callback())
        self.training_sample_size_slider.grid(row=2, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        # set up the slider for Batch Size
        self.batch_size_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                          from_=0, to_=200, resolution=1, bg="#DDDDDD",
                                          length=self.length_widget,
                                          activebackground="#FF0000",
                                          highlightcolor="#00FFFF",
                                          label="Batch Size",
                                          command=lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=3, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        # set up the slider for Number of Iterations
        self.iteration_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                         from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                         length=self.length_widget,
                                         activebackground="#FF0000",
                                         highlightcolor="#00FFFF",
                                         label="Number of Iterations",
                                         command=lambda event: self.iterations_slider_callback())
        self.iteration_slider.set(self.iterations)
        self.iteration_slider.bind("<ButtonRelease-1>", lambda event: self.iterations_slider_callback())
        self.iteration_slider.grid(row=4, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        # for i in range(5,10,1):
        #     self.label_for_learning_method = tk.Label(self.sliders_frame, text="", justify="center",bg="white", width=40)
        #     self.label_for_learning_method.grid(row=i, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        self.setting_weights_zero_button = tk.Button(self.sliders_frame, text="Set Weights and biases to Zero",
                                                     fg="red", bg="#DDDDDD", width=25,
                                                     command=self.set_weights_biases_zero)
        self.setting_weights_zero_button.grid(row=15, column=0)

        self.adjust_weight_button = tk.Button(self.sliders_frame, text="Adjust Weights", fg="red", bg="#DDDDDD",
                                              width=16, command=self.train)
        self.adjust_weight_button.grid(row=16, column=0)

        # Initialization Starts
        self.read_data()
        self.set_weights_biases_zero()

    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()

    def delayed_element_slider_callback(self):
        self.delayed_elements = self.delayed_element_slider.get()

    def training_sample_size_slider_callback(self):
        self.training_sample_percentage = self.training_sample_size_slider.get()
        self.read_data()

    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()

    def iterations_slider_callback(self):
        self.iterations = self.iteration_slider.get()

    # Reading Data File
    def read_data(self):
        # When your program is started it should automatically read the input data file
        self.data = np.loadtxt(self.file_name, skiprows=1, delimiter=',', dtype=np.float32)
        self.number_of_records = len(self.data)
        # Normalize Data
        self.normalize_data()
        self.create_sample()

    def create_sample(self):
        number_of_sample = int((self.number_of_records * self.training_sample_percentage) / 100)
        self.training_sample = self.data[0:number_of_sample]
        self.testing_sample = self.data[number_of_sample:self.number_of_records]

        self.training_sample_size = len(self.training_sample)
        self.testing_sample_size = len(self.testing_sample)

        # print("Training Sample Size : ",self.training_sample_size)
        # print("Testing Sample Size : ",self.testing_sample_size)

    # This method can handle any number of input dimension for normalization
    def normalize_data(self):
        # Normalize the price and volume data by
        # dividing each of the values by the maximum value of the corresponding data and then subtracting 0.5 from each value.
        # Each row of data in the file becomes a row in the matrix
        # So the resulting matrix has dimension [num_samples x sample_dimension]
        self.number_of_input = len(self.data[0, :])

        for i in range(self.number_of_input):
            max_value = np.max(self.data[:, i])
            self.data[:, i] = ((self.data[:, i]) / max_value) - 0.5

            # max_price = np.max(self.data[:, 0])
            # max_volume = np.max(self.data[:, 1])
            # self.data[:, 0] = ((self.data[:, 0])/ max_price) - 0.5
            # self.data[:, 1] = ((self.data[:, 1]) / max_volume) - 0.5

    # This will initialize the weights and bias to zero
    def set_weights_biases_zero(self):
        self.reset_all()
        # When this button is pushed all the weights and biases should be set to zero
        # The weights should not be reset when the "Learn" button is pressed.
        if self.add_bias:
            self.weights = np.zeros(shape=(self.number_of_output, ((self.delayed_elements * 2) + self.number_of_input + 1)))
        else:
            self.weights = np.zeros(shape=(self.number_of_output, ((self.delayed_elements * 2) + self.number_of_input)))

    # This method will update the weights for the network
    def adjust_weights(self):
        # LMS algorithm should be applied and plots should be displayed
        # The weights should not be reset when the "Learn" button is pressed.
        # go through the current training batch and adjust the weight accordingly

        number_of_delayed_batch = self.batch_size - (self.delayed_elements + 1)
        # print("Batch is divided into number of delayed window : ",number_of_delayed_batch)
        for i in range(number_of_delayed_batch):
            self.prepare_input(i)
            self.calculate_output()
            error = self.target - self.net_value
            update_weights_by = 2 * self.learning_rate * error
            update_weights_by = np.outer(np.array(update_weights_by), np.array(self.input))
            self.weights = self.weights + update_weights_by

            # print("Delayed Window Data for Input : ",len(input_data)

    # This method will calculate the output of the network
    def calculate_output(self):
        self.net_value = np.array(np.dot(self.weights, self.input)).reshape(-1)

    # This method will prepare input to the network according to the each batch and delay elements
    def prepare_input(self, i):
        from_point = i
        to_point = i + (self.delayed_elements + 1)
        input = self.batch_data[from_point:to_point]
        self.input = np.array(np.array(input).reshape(-1), dtype=float).reshape(-1, 1)
        if self.add_bias:
            self.input = np.vstack((self.input, [1]))
        self.target = self.batch_data[to_point]

    # This method will help in preparation of batch for the training data
    def prepare_batch(self, i, data):
        from_point = i * self.batch_size
        to_point = (i + 1) * self.batch_size
        self.batch_data = data[from_point:to_point]

    # Following method will plot the graph
    def plot_graph(self):

        # Plot MSE Price
        value = len(self.mean_square_error_price)
        self.x_values = np.linspace(0, value, value, endpoint=True, dtype=int)
        self.y_values = self.mean_square_error_price
        self.axes.cla()
        self.axes.cla()
        self.axes.plot(self.x_values, self.y_values)
        self.axes.xaxis.set_visible(True)
        self.axes.set_title("Mean Squared Error (MSE) for price")
        plt.xlim(self.xmin, value)
        plt.ylim(self.ymin, self.ymax)
        self.canvas.draw()

        # Plot MSE Volume
        value1 = len(self.mean_square_error_volume)
        self.x_values1 = np.linspace(0, value1, value1, endpoint=True, dtype=int)
        self.y_values1 = self.mean_square_error_volume
        self.axes1.cla()
        self.axes1.cla()
        self.axes1.plot(self.x_values1, self.y_values1)
        self.axes1.xaxis.set_visible(True)
        self.axes1.set_title("Mean Squared Error (MSE) for volume")
        plt.xlim(self.xmin, value1)
        plt.ylim(self.ymin, self.ymax)
        self.canvas1.draw()

        # Plot MAE Price
        value2 = len(self.mean_absolute_error_price)
        self.x_values2 = np.linspace(0, value2, value2, endpoint=True, dtype=int)
        self.y_values2 = self.mean_absolute_error_price
        self.axes2.cla()
        self.axes2.cla()
        self.axes2.plot(self.x_values2, self.y_values2)
        self.axes2.xaxis.set_visible(True)
        self.axes2.set_title("Maximum Absolute Error (MAE) for price")
        plt.xlim(self.xmin, value2)
        plt.ylim(self.ymin, self.ymax)
        self.canvas2.draw()

        # Plot MAE Volume
        value3 = len(self.mean_absolute_error_volume)
        self.x_values3 = np.linspace(0, value3, value3, endpoint=True, dtype=int)
        self.y_values3 = self.mean_absolute_error_volume
        self.axes3.cla()
        self.axes3.cla()
        self.axes3.plot(self.x_values3, self.y_values3)
        self.axes3.xaxis.set_visible(True)
        self.axes3.set_title("Maximum Absolute Error (MAE) for volume")
        plt.xlim(self.xmin, value3)
        plt.ylim(self.ymin, self.ymax)
        self.canvas3.draw()

    # Following method will help and start the training of the data
    def train(self):
        number_of_batches = int(self.training_sample_size / self.batch_size)
        # count = 0
        for iteration in range(self.iterations):
            for i in range(number_of_batches):
                self.prepare_batch(i, self.training_sample)
                self.adjust_weights()
                # count = count + 1
                # print(count)
                # Once that batch is processed, freeze the weights and biases
                # run the test set through the network and display MSE and MAE for price and volume
                self.test()

                # Calculations of the error should be done after the "Batch Size"  samples have been processed
                # self.calculate_error()

                # four plots, Mean Squared Error (MSE) for price and volume and
                #  Maximum Absolute Error (MAE) for price and volume
                # self.plot_graph()

        print(self.mean_absolute_error_price)
        print(self.mean_absolute_error_volume)
        print(self.mean_square_error_price)
        print(self.mean_square_error_volume)

    # Testing method
    def test(self):
        self.batch_data = self.testing_sample
        self.calculate_error()
        self.plot_graph()

    # Calculate the error for the batch
    def calculate_error(self):
        # Calculations of the error should be done after the "Batch Size"
        self.error_price = []
        self.error_volume = []

        number_of_delayed_batch = self.batch_size - (self.delayed_elements + 1)

        for i in range(number_of_delayed_batch):
            self.prepare_input(i)
            self.calculate_output()

            # Following Error will be multiple for one batch
            error = self.target - self.net_value
            error_price = error[0]
            error_volume = error[1]
            self.error_price.append(error_price)
            self.error_volume.append(error_volume)

        # calculate Error for each batch size of 100
        self.calculate_maximum_absolute_error()
        self.calculate_mean_square_error()

    def calculate_mean_square_error(self):
        self.mean_square_error_price.append(np.mean(np.square(self.error_price)))
        self.mean_square_error_volume.append(np.mean(np.square(self.error_volume)))

    def calculate_maximum_absolute_error(self):
        self.mean_absolute_error_price.append(np.max(np.abs(self.error_price)))
        self.mean_absolute_error_volume.append(np.max(np.abs(self.error_volume)))

    def reset_all(self):
        self.mean_absolute_error_volume = []
        self.mean_absolute_error_price = []
        self.mean_square_error_volume = []
        self.mean_square_error_price = []
        self.weights = None
