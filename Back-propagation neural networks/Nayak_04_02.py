# Nayak, Anil Kumar


import matplotlib
import tensorflow as tf
import itertools
from sklearn.metrics import confusion_matrix

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tensorflow.examples.tutorials.mnist import input_data

np.seterr(divide='ignore')


class BackPropagation:

    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root

        #########################################################################
        # Hebbian Variable declaration
        #########################################################################
        # Model Values
        self.sess = tf.InteractiveSession()
        self.lambda_weight_regularization = 0.01
        self.learning_rate = 0.1
        self.training_sample_percentage = 10
        self.batch_size = 64
        self.nodes_in_hidden_layer = 100
        self.data = None
        self.weights = None
        self.epochs = 10
        self.number_of_output = 10
        self.hidden_layer_transfer_function = "Relu"
        self.output_layer_transfer_function = "SoftMax"
        self.cost_function_type = "Cross Entropy"
        self.cost_function = None
        self.optimizer = None
        self.hidden_layer_weights = None
        self.hidden_layer_biases = None
        self.output_layer_weights = None
        self.output_layer_biases = None
        self.epoch_loss = 0
        self.regularizer = None
        self.regularization = None
        self.reset_data = False
        # Sample Training and Testing data
        self.initial_weight_min = -0.0001
        self.initial_weight_max = 0.0001
        self.training_sample = None
        self.training_labels = None
        self.training_sample_size = 0
        self.testing_sample = None
        self.testing_labels = None
        self.testing_sample_size = 0

        # Graph Details
        # The limits of the error axes should be set between 0 and 2
        figure_size = (5, 4)
        self.xmin = 0
        self.xmax = self.batch_size
        self.ymin = 0
        self.ymax = 2
        self.x_values = None
        self.y_values = None
        self.reset = True
        self.zero_weight = False
        self.use_old_weight = False

        # Widget Details
        self.length_widget = 300

        # Network Details
        self.input = tf.placeholder('float', [None, 784])
        self.target = tf.placeholder('float', [None, 10])
        self.model = None
        self.epoch_loss = []
        self.error_rate = []
        self.accuracy_rate = []

        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame_parent = tk.Frame(self.master)
        self.plot_frame_parent.grid(row = 0, column = 0)  # , sticky=tk.N + tk.E + tk.S + tk.W
        self.plot_frame_parent.rowconfigure(0, weight = 2)
        self.plot_frame_parent.columnconfigure(0, weight = 2, uniform = 'xx')


        self.plot_frame = tk.Frame(self.plot_frame_parent)
        self.plot_frame.grid(row=0, column=0)
        self.figure = plt.figure(figsize=figure_size)
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Loss')
        self.axes.set_title("Loss Function")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0)

        self.plot_frame1 = tk.Frame(self.plot_frame_parent)
        self.plot_frame1.grid(row = 1, column = 0)
        self.figure1 = plt.figure(figsize = figure_size)
        self.axes1 = self.figure1.gca()
        self.axes1.set_xlabel('Epoch')
        self.axes1.set_ylabel('Error')
        self.axes1.set_title("Error Function")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master = self.plot_frame1)
        self.plot_widget1 = self.canvas1.get_tk_widget()
        self.plot_widget1.grid(row = 1, column = 0)

        self.plot_frame3 = tk.Frame(self.plot_frame_parent)
        self.plot_frame3.grid(row = 1, column = 1)
        self.figure3 = plt.figure(figsize = figure_size)
        self.axes3 = self.figure3.gca()
        self.axes3.set_xlabel('Epoch')
        self.axes3.set_ylabel('Accuracy')
        self.axes3.set_title("Accuracy Graph")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas3 = FigureCanvasTkAgg(self.figure3, master = self.plot_frame3)
        self.plot_widget3 = self.canvas3.get_tk_widget()
        self.plot_widget3.grid(row = 1, column = 1)

        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=0, column=3)  # , sticky=tk.N + tk.E + tk.S + tk.W
        self.sliders_frame.rowconfigure(0, weight=2)
        self.sliders_frame.columnconfigure(0, weight=2, uniform='xx')

        # set up the sliders Number of Delayed Elements
        self.lambda_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=0.0, to_=1.0, resolution=0.01, bg="#DDDDDD", length=self.length_widget,
                                      activebackground="#FF0000",
                                      highlightcolor="#00FFFF",
                                      label="Lambda (Weight regularization)",
                                      command=lambda event: self.lambda_slider_callback())
        self.lambda_slider.set(self.lambda_weight_regularization)
        self.lambda_slider.bind("<ButtonRelease-1>", lambda event: self.lambda_slider_callback())
        self.lambda_slider.grid(row=0, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        # set up the sliders Learning Rate
        self.learning_rate_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                             length=self.length_widget,
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF",
                                             label="Alpha (Learning Rate)",
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=1, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        # set up the sliders Training Sample Size (Percentage)
        self.training_sample_size_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                                    from_=1, to_=100, resolution=1, bg="#DDDDDD",
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
                                          from_=1, to_=256, resolution=1, bg="#DDDDDD",
                                          length=self.length_widget,
                                          activebackground="#FF0000",
                                          highlightcolor="#00FFFF",
                                          label="Batch Size",
                                          command=lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=3, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        # set up the slider for Number of Iterations
        self.hidden_layer_slider = tk.Scale(self.sliders_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL,
                                            from_=1, to_=500, resolution=1, bg="#DDDDDD",
                                            length=self.length_widget,
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Num. of Nodes (in hidden layer)",
                                            command=lambda event: self.hidden_layer_node_slider_callback())
        self.hidden_layer_slider.set(self.nodes_in_hidden_layer)
        self.hidden_layer_slider.bind("<ButtonRelease-1>", lambda event: self.hidden_layer_node_slider_callback())
        self.hidden_layer_slider.grid(row=4, column=0)  # , sticky=tk.N + tk.E + tk.S + tk.W

        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################
        self.label_for_hidden_layer_transfer_function = tk.Label(self.sliders_frame,
                                                                 text="Hidden Layer Transfer Function",
                                                                 justify="center",
                                                                 bg="#DDDDDD", width=40)
        self.label_for_hidden_layer_transfer_function.grid(row=5, column=0)

        self.hidden_layer_transfer_function_variable = tk.StringVar()
        self.hidden_layer_transfer_function_dropdown = tk.OptionMenu(self.sliders_frame,
                                                                     self.hidden_layer_transfer_function_variable,
                                                                     "Relu", "Sigmoid",
                                                                     command=lambda
                                                                         event: self.hidden_layer_transfer_function_callback())
        self.hidden_layer_transfer_function_dropdown.config(width=20)
        self.hidden_layer_transfer_function_variable.set(self.hidden_layer_transfer_function)
        self.hidden_layer_transfer_function_dropdown.grid(row=6, column=0)

        self.label_for_output_layer_transfer_function = tk.Label(self.sliders_frame,
                                                                 text="Output Layer Transfer Function",
                                                                 justify="center",
                                                                 bg="#DDDDDD", width=40)
        self.label_for_output_layer_transfer_function.grid(row=7, column=0)

        self.output_layer_transfer_function_variable = tk.StringVar()
        self.output_layer_transfer_function_dropdown = tk.OptionMenu(self.sliders_frame,
                                                                     self.output_layer_transfer_function_variable,
                                                                     "Softmax", "Sigmoid",
                                                                     command=lambda
                                                                         event: self.output_layer_transfer_function_callback())
        self.output_layer_transfer_function_dropdown.config(width=20)
        self.output_layer_transfer_function_variable.set(self.output_layer_transfer_function)
        self.output_layer_transfer_function_dropdown.grid(row=8, column=0)

        self.label_for_cost_function = tk.Label(self.sliders_frame,
                                                text="Cost Function",
                                                justify="center",
                                                bg="#DDDDDD", width=40)
        self.label_for_cost_function.grid(row=9, column=0)
        self.cost_function_variable = tk.StringVar()
        self.cost_function_dropdown = tk.OptionMenu(self.sliders_frame,
                                                    self.cost_function_variable,
                                                    "Cross Entropy", "Mean Square Error",
                                                    command=lambda
                                                        event: self.cost_function_callback())
        self.cost_function_dropdown.config(width=20)
        self.cost_function_variable.set(self.cost_function_type)
        self.cost_function_dropdown.grid(row=10, column=0)

        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################



        self.setting_weights_zero_button = tk.Button(self.sliders_frame, text="Set Weights and biases to Zero",
                                                     fg="red", bg="#DDDDDD", width=25,
                                                     command=self.set_weights_biases_zero)
        self.setting_weights_zero_button.grid(row=11, column=0)

        self.adjust_weight_button = tk.Button(self.sliders_frame, text="Adjust Weights", fg="red", bg="#DDDDDD",
                                              width=16, command=self.train)
        self.adjust_weight_button.grid(row=12, column=0)

        self.reset_button = tk.Button(self.sliders_frame, text = "Reset Network (As you have started)", fg = "red", bg = "#DDDDDD",
                                              width = 25, command = self.reset_begining)
        self.reset_button.grid(row = 14, column = 0)

        # Initialization Starts
        self.read_data()

    def hidden_layer_transfer_function_callback(self):
        self.hidden_layer_transfer_function = self.hidden_layer_transfer_function_variable.get()
        self.reset_all(['model'])

    def output_layer_transfer_function_callback(self):
        self.output_layer_transfer_function = self.output_layer_transfer_function_variable.get()
        self.reset_all(['model'])

    def cost_function_callback(self):
        self.cost_function_type = self.cost_function_variable.get()
        self.reset_all(['model'])

    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()
        self.reset_all(['model'])

    def lambda_slider_callback(self):
        self.lambda_weight_regularization = self.lambda_slider.get()
        self.reset_all(['model'])

    def training_sample_size_slider_callback(self):
        self.training_sample_percentage = self.training_sample_size_slider.get()
        self.reset_all(['data','model'])

    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()


    def hidden_layer_node_slider_callback(self):
        self.nodes_in_hidden_layer = self.hidden_layer_slider.get()
        self.reset_all(['model_create'])

    def set_weights_biases_zero(self) :
        self.reset_all(['model_weight_zero'])

    def reset_begining(self):
        self.reset = True
        self.epoch_loss = []
        self.error_rate = []
        self.accuracy_rate = []
        self.use_old_weight = False
        self.zero_weight = False
        self.reset_data = True
        self.lambda_weight_regularization = 0.01
        self.lambda_slider.set(self.lambda_weight_regularization)
        self.learning_rate = 0.1
        self.learning_rate_slider.set(self.learning_rate)
        self.training_sample_percentage = 10
        self.training_sample_size_slider.set(self.training_sample_percentage)
        self.batch_size = 64
        self.batch_size_slider.set(self.batch_size)
        self.nodes_in_hidden_layer = 100
        self.hidden_layer_slider.set(self.nodes_in_hidden_layer)
        self.hidden_layer_transfer_function = "Relu"
        self.hidden_layer_transfer_function_variable.set(self.hidden_layer_transfer_function)
        self.output_layer_transfer_function = "SoftMax"
        self.output_layer_transfer_function_variable.set(self.output_layer_transfer_function)
        self.cost_function_type = "Cross Entropy"
        self.cost_function_variable.set(self.cost_function_type)

    def reset_all(self,reset_configs):
        for reset_config in reset_configs:
            if reset_config == 'model':
                self.reset = True
                self.use_old_weight = True
                self.zero_weight = False
            elif reset_config == 'model_create':
                self.reset = True
                self.use_old_weight = False
                self.zero_weight = False
                self.epoch_loss = []
                self.error_rate = []
                self.accuracy_rate = []
            elif reset_config == 'model_weight_zero':
                self.reset = True
                self.use_old_weight = False
                self.zero_weight = True
                self.epoch_loss = []
                self.error_rate = []
                self.accuracy_rate = []
            elif reset_config == 'data':
                self.reset_data = True
                self.use_old_weight = True
                self.zero_weight = False
            elif reset_config == 'weight':
                self.use_old_weight = False
                self.zero_weight = False
                self.epoch_loss = []
                self.error_rate = []
                self.accuracy_rate = []

    # Reading Data File
    def read_data(self):
        self.data = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.data.train._images = np.vstack((self.data.train._images, self.data.test._images))
        self.data.train._labels = np.vstack((self.data.train._labels, self.data.test._labels))
        self.normalize_data()
        self.create_sample()

    # This method can handle any number of input dimension for normalization
    def normalize_data(self):
        self.data.train._images = (self.data.train._images-0.5) * 2

    # This method will create sample for the training and testing
    def create_sample(self):
        total = np.shape(self.data.train._images)[0]
        training_sample_size = int(( total* self.training_sample_percentage) / 100)
        self.data.test._images = self.data.train._images[training_sample_size :total, :]
        self.data.train._images = self.data.train._images[0:training_sample_size, :]
        self.data.test._labels = self.data.train._labels[training_sample_size :total, :]
        self.data.train._labels = self.data.train._labels[0 :training_sample_size, :]
        self.training_sample_size = np.shape(self.data.train._images)[0]
        self.data.train._num_examples = self.training_sample_size

    # prepare weights and biases for the model
    def prepare_weights(self):

        # if self.use_old_weight :
        #     print('reusing weights')
        #     tf.get_variable_scope().reuse_variables()
        #
        # with tf.variable_scope('weights') as scope:

        if self.use_old_weight :
            print('reusing weights')

        else:
            # print('new weights init')
            if not self.zero_weight:
                # print('preparing new random weights')
                self.hidden_layer_weights = tf.Variable(tf.random_uniform(shape=[784, self.nodes_in_hidden_layer],
                                                                          minval=self.initial_weight_min,
                                                                          maxval=self.initial_weight_max,
                                                                          dtype=tf.float32),
                                                        name="weights_hidden_layer_1")

                self.hidden_layer_biases = tf.Variable(tf.random_uniform(shape=[self.nodes_in_hidden_layer],
                                                                         minval=self.initial_weight_min,
                                                                         maxval=self.initial_weight_max,
                                                                         dtype=tf.float32),
                                                       name="biases_hidden_layer_1")

                self.output_layer_weights = tf.Variable(tf.random_uniform(shape=[self.nodes_in_hidden_layer, self.number_of_output],
                                                                          minval=self.initial_weight_min,
                                                                          maxval=self.initial_weight_max,
                                                                          dtype=tf.float32),
                                                        name="weights_output_layer")

                self.output_layer_biases = tf.Variable(tf.random_uniform(shape=[self.number_of_output],
                                                                         minval=self.initial_weight_min,
                                                                         maxval=self.initial_weight_max,
                                                                         dtype=tf.float32),
                                                   name="biases_output_layer")
            else:
                # print('preparing zero weight')
                self.hidden_layer_weights = tf.Variable(tf.zeros(shape = [784, self.nodes_in_hidden_layer],
                                                                 dtype = tf.float32),
                                                        name = "weights_hidden_layer_1")

                self.hidden_layer_biases = tf.Variable(tf.ones(shape = [self.nodes_in_hidden_layer],
                                                               dtype = tf.float32),
                                                       name = "biases_hidden_layer_1")

                self.output_layer_weights = tf.Variable(tf.zeros(shape = [self.nodes_in_hidden_layer, self.number_of_output],
                                                                 dtype = tf.float32),
                                                        name = "weights_output_layer")

                self.output_layer_biases = tf.Variable(tf.ones(shape = [self.number_of_output],
                                                               dtype = tf.float32),
                                                       name = "biases_output_layer")

                self.zero_weight = False

    # Prepare the regularization
    def prepare_regularization(self):
        # print('applying regularization')
        self.regularizer = tf.contrib.layers.l2_regularizer(self.lambda_weight_regularization)
        self.regularization =  tf.contrib.layers.apply_regularization(self.regularizer,[self.hidden_layer_weights,self.output_layer_weights])

    # prepare model
    def prepare_model(self):
        # print('preparing models')
        self.prepare_weights()

        hidden_layer = {'weights': self.hidden_layer_weights,
                        'biases': self.hidden_layer_biases}

        output_layer = {'weights': self.output_layer_weights,
                        'biases': self.output_layer_biases}

        l1 = tf.add(tf.matmul(self.input, hidden_layer['weights']), hidden_layer['biases'])

        if self.hidden_layer_transfer_function == 'Relu':
            l1 = tf.nn.relu(l1, name="relu_hidden_layer")
        elif self.hidden_layer_transfer_function == 'Sigmoid':
            l1 = tf.sigmoid(l1, name="sigmoid_hidden_layer")

        output_l = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

        if self.output_layer_transfer_function == 'Softmax':
            output_l = tf.nn.softmax(output_l, name="softmax_output_layer")
        elif self.output_layer_transfer_function == 'Sigmoid':
            output_l = tf.sigmoid(output_l, name="sigmoid_output_layer")

        self.model = output_l

    # Prepare cost function
    def prepare_cost_function(self):
        # print('preparing cost function')
        if self.cost_function_type == "Cross Entropy":
            self.cost_function = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.target))
        elif self.cost_function_type == "Mean Square Error":
            self.cost_function = tf.reduce_mean(
                tf.losses.mean_squared_error(predictions=self.model, labels=self.target))

        # Apply Regularization to cost function
        self.cost_function += self.regularization

    # Prepare Optimizer
    def prepare_optimizer(self):
        # print('preparing optimizer')
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
            self.cost_function)

    # Following method will help and start the training of the data
    def train(self):

        if self.reset_data:
            # print('reinitializing data')
            self.read_data()
            self.reset_data = False

        if self.reset:
            # print("preparing model in training")
            # Prepare Model
            self.prepare_model()

            # Prepare Regularization
            self.prepare_regularization()

            # Prepare Cost Function
            self.prepare_cost_function()

            # Prepare Optimizer
            self.prepare_optimizer()

            # Initialize the variables
            self.sess.run(tf.global_variables_initializer())

            self.reset = False
            self.use_old_weight = True
            self.zero_weight = False

        # print(self.hidden_layer_weights)
        # print('running session')
        with self.sess.as_default():
            # saver = tf.train.Saver()
            for epoch in range(self.epochs):
                epoch_loss_itr = 0
                for _ in range(int(self.training_sample_size / self.batch_size)):
                    epoch_x, epoch_y = self.data.train.next_batch(self.batch_size)
                    _, c = self.sess.run([self.optimizer, self.cost_function], feed_dict={self.input: epoch_x, self.target: epoch_y})
                    epoch_loss_itr += c
                self.epoch_loss.append(epoch_loss_itr)

                correct = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.target, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                accuracy_rate_epoch = accuracy.eval({self.input: self.data.test._images, self.target: self.data.test._labels})
                # print('Epoch ', epoch, ' completed out of ', self.epochs, ' loss is ', epoch_loss_itr)
                # print('Accuracy is ', accuracy_rate)
                self.accuracy_rate.append(accuracy_rate_epoch)
                self.error_rate.append(1-accuracy_rate_epoch)

                self.plot_graph()

            # Confusion Matrix
            target_labels = self.data.test._labels
            actual_labels = self.sess.run(self.model, feed_dict = {self.input: self.data.test._images})

            target_labels = self.sess.run(tf.argmax(target_labels,1))
            actual_labels = self.sess.run(tf.argmax(actual_labels, 1))

            self.plot_confusion_matrix(target_labels,actual_labels)

    # Following method will plot the graph
    def plot_graph(self):

        # Plot Loss Function
        value = len(self.epoch_loss)
        max = np.max(self.epoch_loss)
        self.x_values = np.linspace(0, value, value, endpoint=True, dtype=int)
        self.y_values = self.epoch_loss
        self.axes.cla()
        self.axes.cla()
        self.axes.plot(self.x_values, self.y_values)
        self.axes.xaxis.set_visible(True)
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Loss')
        self.axes.set_title("Loss Function")
        plt.xlim(self.xmin, value)
        plt.ylim(self.ymin, max)
        self.canvas.draw()

        # Plot Error Function
        value1 = len(self.error_rate)
        max1 = np.max(self.error_rate)
        x_values = np.linspace(0, value1, value1, endpoint = True, dtype = int)
        y_values = self.error_rate
        self.axes1.cla()
        self.axes1.cla()
        self.axes1.plot(x_values, y_values)
        self.axes1.xaxis.set_visible(True)
        self.axes1.set_xlabel('Epoch')
        self.axes1.set_ylabel('Error')
        self.axes1.set_title("Error Function")
        plt.xlim(self.xmin, value1)
        plt.ylim(self.ymin, max1)
        self.canvas1.draw()

        # Plot Accuracy Function
        value2 = len(self.accuracy_rate)
        max2 = np.max(self.accuracy_rate)
        x_values = np.linspace(0, value2, value2, endpoint = True, dtype = int)
        y_values = self.accuracy_rate
        self.axes3.cla()
        self.axes3.cla()
        self.axes3.plot(x_values, y_values)
        self.axes3.xaxis.set_visible(True)
        self.axes3.set_xlabel('Epoch')
        self.axes3.set_ylabel('Accuracy')
        self.axes3.set_title("Accuracy Graph")
        plt.xlim(self.xmin, value2)
        plt.ylim(self.ymin, max2)
        self.canvas3.draw()

    def plot_confusion_matrix(self,y_test,y_pred,class_names=[0,1,2,3,4,5,6,7,8,9]):
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        self.confusion_matrix_plot(cnf_matrix, classes=class_names, title='Confusion matrix')

    def confusion_matrix_plot(self,cm,classes,title,cmap=plt.cm.Blues,normalize=False):
        self.plot_frame2 = tk.Frame(self.plot_frame_parent)
        self.plot_frame2.grid(row = 0, column = 1)
        self.figure2 = plt.figure(figsize = (5,4))
        self.axes2 = self.figure2.gca()
        self.axes2.set_xlabel('Predicted label')
        self.axes2.set_ylabel('True label')
        self.axes2.set_title("Confusion Matrix")
        imshow_obj =self.axes2.imshow(cm, interpolation='nearest', cmap=cmap)
        self.figure2.colorbar(imshow_obj)
        tick_marks = np.arange(len(classes))
        self.axes2.set_xticks(tick_marks, classes) #, rotation = 45
        self.axes2.set_yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            self.axes2.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        self.figure2.tight_layout()
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master = self.plot_frame2)
        self.plot_widget2 = self.canvas2.get_tk_widget()
        self.plot_widget2.grid(row = 0, column = 1)
        self.canvas2.draw()

        self.plot_frame2 = tk.Frame(self.plot_frame_parent)

