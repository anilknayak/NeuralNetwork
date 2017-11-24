# Nayak, Anil Kumar


import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import os
import scipy.misc
from sklearn.utils import shuffle
np.seterr(divide='ignore')

class HebbianLearning:


    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root

        #########################################################################
        # Hebbian Variable declaration
        #########################################################################
        self.epochs = 0
        self.epochs_per_click = 100
        self.decay = 0.1
        self.xmin = 0
        self.xmax = 1
        self.ymin = 0
        self.ymax = 105
        self.x_values = []
        self.y_values = []
        self.data = []

        # Model Values
        self.learning_rate = 0.1
        self.weight = 0.001
        self.weight_min = -0.001
        self.weight_max = 0.001
        self.bias_min = -0.001
        self.bias_max = 0.001
        self.weights = []
        self.bias = []

        self.activation_function = "Symmetrical Hard limit"
        self.learning_method = "Filtered Learning"

        self.inputs_nodes = 784 # Input layer
        self.nodes_in_layer = [{'nodes':10}] # Hidden Layer
        self.output_nodes = 10 # Output layer

        self.error = []

        self.number_of_input_data = 1000

        self.isLayerPrepared = False
        self.isDataPrepared = False
        self.resetPlot = False
        self.layers = []

        self.isDataReadComplete = False
        self.training_images = None
        self.training_target = None
        self.testing_images = None
        self.testing_target = None

        self.targetMin = -1
        self.targetMax = 1

        self.length_widget = 300

        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=3, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=2)
        self.plot_frame.columnconfigure(0, weight=2)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel('Error rate (%)')
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

        # set up the sliders learning Rate
        self.learning_rate_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD",length=self.length_widget,
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Alpha (Learning Rate)",
                                            command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=0) #, sticky=tk.N + tk.E + tk.S + tk.W


        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################
        self.label_for_learning_method = tk.Label(self.sliders_frame, text="Learning Method", justify="center",bg="#DDDDDD",width=40)
        self.label_for_learning_method.grid(row=4, column=0) #, sticky=tk.N + tk.E + tk.S + tk.W

        self.learning_method_variable = tk.StringVar()
        self.learning_method_dropdown = tk.OptionMenu(self.sliders_frame, self.learning_method_variable,
                                                          "Filtered Learning", "Delta Rule","Unsupervised Hebb",
                                                          command=lambda
                                                              event: self.learning_method_callback())
        self.learning_method_dropdown.config(width=20)
        self.learning_method_variable.set(self.learning_method)
        self.learning_method_dropdown.grid(row=5, column=0) #, sticky=tk.N + tk.E + tk.S + tk.W

        self.label_for_activation_function = tk.Label(self.sliders_frame, text="Transfer Functions", justify="center",bg="#DDDDDD",width=40)
        self.label_for_activation_function.grid(row=6, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.sliders_frame, self.activation_function_variable,
                                                          "Symmetrical Hard limit", "Hyperbolic Tangent", "Linear",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_dropdown.config(width=20)
        self.activation_function_variable.set(self.activation_function)
        self.activation_function_dropdown.grid(row=7, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        self.draw_button_generate = tk.Button(self.sliders_frame, text="Rondomize Weights & Bias", fg="red",bg="#DDDDDD", width=20, command=self.generate_weight_and_bias)
        self.draw_button_generate.grid(row=8, column=0)

        self.draw_button_train = tk.Button(self.sliders_frame, text="Adjust Weights (Learn)", fg="red",bg="#DDDDDD", width=16, command=self.on_click_train)
        self.draw_button_train.grid(row=9, column=0)

        self.read_data()
        self.reset_plot()



    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        self.set_title_of_graph()

    def learning_method_callback(self):
        self.learning_method = self.learning_method_variable.get()
        self.set_title_of_graph()

    def set_title_of_graph(self):
        self.title = "Error Rate Graph while Learning Method : "+self.learning_method + " and Transfer Function : "+self.activation_function
        self.root.top_frame.set('%s', "Error Rate Graph while Learning Method :")

    def reset_plot(self):
        self.epochs = 0
        self.xmax = 1
        self.error = []
        self.root.status_bar.set('%s', "Error Graph has plotted")
        self.x_values = []
        self.y_values = []

        self.axes.cla()
        self.axes.cla()

        self.axes.plot(self.x_values, self.y_values)
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)

        plt.title("Error Rate Graph")
        self.canvas.draw()

    def find_activation(self,net_value,isarray):
        activation =[]

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
            net_value = np.array(net_value,dtype=np.float128)

            # neu = (np.exp(net_value) - np.exp(-net_value))
            # deno = (np.exp(net_value) + np.exp(-net_value))
            # neu = np.nan_to_num(neu)
            # deno = np.nan_to_num(deno)
            # activation = neu / deno

            activation = np.tanh(net_value)


        return activation

    def update_weights_per_input(self,input,target):
        if self.learning_method == 'Filtered Learning':
            self.adjust_weights_by_filtered_learning(input,target)
        elif self.learning_method == 'Delta Rule':
            self.adjust_weights_by_delta_rule(input,target)
        elif self.learning_method == 'Unsupervised Hebb':
            self.adjust_weights_by_unsupervised_hebb(input)

    def adjust_weights_by_filtered_learning(self,input,target):
        product = np.dot(target,np.transpose(input))
        weightToBeChanged = self.learning_rate * product
        for index in range(len(self.weights)):
            self.weights[index] = (1-self.decay) * (self.weights[index]) + weightToBeChanged[index]

        # newW  = (1-decay) oldW + LR * target * Input

    def adjust_weights_by_delta_rule(self, input, target):
        net = np.dot(self.weights,input) # 10x1
        activation = self.find_activation(net,True)
        product = np.dot((target-activation),np.transpose(input))
        weightToBeChanged = self.learning_rate * product
        for index in range(len(self.weights)):
            self.weights[index] = self.weights[index] + weightToBeChanged[index]

        # newW  = oldW + LR * ERROR * Input

    def adjust_weights_by_unsupervised_hebb(self, input):
        net = np.dot(self.weights, input)
        activation = self.find_activation(net, True)
        product = np.dot(activation,np.transpose(input))
        weightToBeChanged = self.learning_rate * product
        for index in range(len(self.weights)):
            self.weights[index] = self.weights[index] + weightToBeChanged[index]

        # newW  = oldW + LR * activation_output * Input

    def reconfigure_output(self,output):
        modified_output = []

        # or self.learning_method == 'Unsupervised Hebb'
        if self.learning_method == 'Delta Rule' or self.learning_method == 'Unsupervised Hebb':

            if self.activation_function == "Hyperbolic Tangent" or self.activation_function == "Sigmoid" or self.activation_function == "Linear":
                len_arr = len(output)
                max_index = np.argmax(output)

                modified_output = np.ones(len_arr) * self.targetMin
                modified_output[max_index] = self.targetMax
            else:
                activation = self.find_activation(output, True)
                modified_output = activation
        else:
            len_arr = len(output)
            max_index = np.argmax(output)

            modified_output = np.ones(len_arr) * self.targetMin
            modified_output[max_index] = self.targetMax

        return modified_output

    def on_click_train(self):
        if self.resetPlot:
            # print("First time")
            # Read Data
            if not self.isDataReadComplete:
                self.read_data()

            # Prepare Weights
            if not self.isDataPrepared:
                # Generating random Bias and Weights for each node in each layer
                self.generate_weight_and_bias()

            self.reset_plot()

        self.train()

    # Hebbian Functions starts
    def train(self):
        self.resetPlot = False
        self.root.status_bar.set('%s', "Training on 800 Mnist image data has commenced")
        lentr = len(self.training_images)

        for epoch in range(self.epochs_per_click):
            # This means that you train (adjust the weights) for one complete set of the training data (one epoch)

            for i in range(0,lentr):
                target1 = self.training_target[i, :]
                input1 = self.training_images[i, :]

                target = np.array(target1).reshape(-1,1)
                input = np.array(input1).reshape(-1,1)

                self.update_weights_per_input(input, target)

            self.test(epoch)

        # Plot Graph
        self.plot_error_graph()

        self.root.status_bar.set('%s', "Error Graph has plotted")
        return 0

    def test(self,epoch):
        self.root.status_bar.set('%s', "Testing on 200 Mnist image data has commenced for Epoch : " + str(epoch))
        #Then turn off the training (freeze the weights and biases) and run the test data through the network and calculate the error rate
        lentr = len(self.testing_images)

        # An epoch is one pass over all the training samples
        correct_classification = 0
        for i in range(0, lentr):
            target1 = self.testing_target[i, :]
            input1 = self.testing_images[i, :]

            target = np.array(target1,dtype=float).reshape(-1, 1)
            input = np.array(input1,dtype=float).reshape(-1, 1)

            net = np.dot(self.weights, input)
            output = self.reconfigure_output(net)

            output = np.array(output,dtype=float).reshape(-1,1)

            if (target == output).all():
                correct_classification = correct_classification + 1

        errorRate = 100*((lentr - correct_classification) / lentr)
        self.error.append(errorRate)


    def plot_error_graph(self):
        self.root.status_bar.set('%s', "Error Graph has plotted")
        #Plot the error rate, in percent, after each epoch on the error rate graph
        #The error-rate graph should be able to display up to 1000 epochs

        self.epochs = self.epochs + self.epochs_per_click
        self.xmax = self.epochs
        self.x_values = np.linspace(self.xmin, self.epochs, self.epochs, endpoint=True,dtype=int)
        self.y_values = self.error

        self.axes.cla()
        self.axes.cla()

        self.axes.plot(self.x_values, self.y_values)
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)

        plt.title("Error Rate Graph")
        self.canvas.draw()
        return 0


    def generate_weight_and_bias(self):
        # print("Generating Weight")
        self.resetPlot = True
        self.isDataPrepared = True

        self.root.status_bar.set('%s', "Randomized weights and biases has been generated, ranging from "+str(self.weight_min)+" to "+str(self.weight_max))
        # Create Weights and Bias for the layer
        # Automatically initialize all the weights and biases to be from -0.001 to 0.001

        self.weights = []
        self.weights = np.random.uniform(self.weight_min,self.weight_max,(10,785))

    def read_data(self):
        #print("Read Data")
        self.root.status_bar.set('%s', "Reading Mnist data has started")
        dir_path = os.getcwd()
        images_path = dir_path + "/Data"

        all_images = np.zeros((1000, 785))
        all_target = np.zeros((1000, 10))
        counter = 0
        for image_path in os.listdir(images_path):
            image_vector = self.read_one_image_and_convert_to_vector(images_path + "/" + image_path)
            all_images[counter,:] = image_vector
            target = np.ones(10)
            target = target * self.targetMin
            class_id = int(image_path[0])
            target[class_id] = self.targetMax
            all_target[counter,:] = target
            counter = counter + 1

        images, targets = shuffle(all_images,all_target)

        # Separate the input data into two sets
        # The training set
        # The first set should include 80% of your data set (randomly selected)
        #images (1000, 785)
        #targets (1000, 10)

        self.training_images = images[0:800,:]
        self.training_target = targets[0:800,:]

        # self.training_images (800, 785)
        # self.training_target (800, 10)

        # The test set
        # The second set (the other 20%) is the test set. The test set will be used for calculating the error rate
        self.testing_images = images[800:1000,:]
        self.testing_target = targets[800:1000,:]

        self.isDataReadComplete = True
        self.root.status_bar.set('%s', "Reading Mnist data has completed")


    def read_one_image_and_convert_to_vector(self,file_name):
        # Convert to vector

        # Image size in 28x28
        img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float

        # Flat the image to 784 = 28*28
        img_vector = img.reshape(-1)

        # Normalize the image
        img_vector_norm = img_vector/255

        # Divide the input numbers by 255 and subtract 0.5
        img_vector_norm_sub = (img_vector_norm-0.5)*2

        # Adding bias
        img_vector_norm_sub =  np.hstack((img_vector_norm_sub,[1]))

        # print("Shape of Image",np.shape(img_vector_norm_sub))
        return img_vector_norm_sub # reshape to column vector and return it

