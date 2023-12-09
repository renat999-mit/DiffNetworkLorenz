import tensorflow as tf
from tensorflow.keras import layers

import keras
import keras.backend as K

import tensorflow_probability as tfp

import sys
import numpy as np

import json

tfd = tfp.distributions

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')
NUMBER_TYPE = tf.float64  # or tf.float32

STD_MIN_VALUE = 1e-13  # the minimal number that the diffusivity models can have

class ModelBuilder:
    """
    Builder for neural network that approximates :math:'\\Sigma', and Normal Distribution using it.
    """
    DIFF_TYPES = ["diagonal", "triangular", "spd"]

    @staticmethod
    def diff_network(n_input_dimensions,
                     n_output_dimensions,
                     n_layers,
                     n_dim_per_layer,
                     name,
                     diffusivity_type="diagonal",
                     activation="tanh",
                     dtype=tf.float64,
                     ):
        """
        Constructs a neural network for approximating the :math:'\\Sigma' matrix.

        Parameters
        ----------
        n_input_dimensions : int
            Number of input dimensions.
        n_output_dimensions : int
            Number of output dimensions.
        n_layers : int
            Number of layers in the network.
        n_dim_per_layer : int
            Number of neurons in each layer.
        name : str
            Name of the model.
        diffusivity_type : str, optional
            Type of diffusivity matrix to use ('diagonal', 'triangular', 'spd'). Default is 'diagonal'.
        activation : str, optional
            Activation function to use in the layers. Default is 'tanh'.
        dtype : data-type, optional
            Data type of the layers. Default is tf.float64.

        Returns
        -------
        tf.keras.Model
            A TensorFlow Keras model representing the neural network.
        """
        def make_tri_matrix(z):
            # first, make all eigenvalues positive by changing the diagonal to positive values
            z = tfp.math.fill_triangular(z)
            z2 = tf.linalg.diag(tf.linalg.diag_part(z))
            z = z - z2 + tf.abs(z2)  # this ensures the values on the diagonal are positive
            return z

        def make_spd_matrix(z):
            z = make_tri_matrix(z)
            return tf.linalg.matmul(z, tf.linalg.matrix_transpose(z))

        # initialize with small (not zero!) values so that it does not dominate the drift
        # estimation at the beginning of training
        small_init = 1e-2
        # small_init = 0.3
        initializer = tf.keras.initializers.RandomUniform(minval=-small_init, maxval=small_init, seed=None)

        inputs = layers.Input((n_input_dimensions,), dtype=dtype, name=name + '_inputs')

        #Network for Sigma matrix
        gp_x = inputs
        for i in range(n_layers):
            gp_x = layers.Dense(n_dim_per_layer,
                                activation=activation,
                                dtype=dtype,
                                kernel_initializer=initializer,
                                bias_initializer=initializer,
                                name=name + "_std_hidden_{}".format(i))(gp_x)
        if diffusivity_type == "diagonal":
            gp_output_std = layers.Dense(n_output_dimensions,
                                         kernel_initializer=initializer,
                                         bias_initializer=initializer,
                                         activation=lambda x: tf.nn.softplus(x) + STD_MIN_VALUE,
                                         name=name + "_output_std", dtype=dtype)(gp_x)
        elif diffusivity_type == "triangular":
            # the dimension of std should be N*(N+1)//2, for one of the Cholesky factors L of the covariance,
            # so that we can create the lower triangular matrix with positive eigenvalues on the diagonal.
            gp_output_tril = layers.Dense((n_output_dimensions * (n_output_dimensions + 1) // 2),
                                          activation="linear",
                                          kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          name=name + "_output_cholesky", dtype=dtype)(gp_x)
            gp_output_std = layers.Lambda(make_tri_matrix)(gp_output_tril)
        elif diffusivity_type == "spd":
            # the dimension of std should be N*(N+1)//2, for one of the Cholesky factors L of the covariance,
            # so that we can create the SPD matrix C using C = L @ L.T to be used later.
            gp_output_tril = layers.Dense((n_output_dimensions * (n_output_dimensions + 1) // 2),
                                          activation="linear",
                                          kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          name=name + "_output_spd", dtype=dtype)(gp_x)
            gp_output_std = layers.Lambda(make_spd_matrix)(gp_output_tril)
            # gp_output_std = layers.Lambda(lambda L: tf.linalg.matmul(L, tf.transpose(L)))(gp_output_tril)
        else:
            raise ValueError(f"Diffusivity type {diffusivity_type} not supported. Use one of {ModelBuilder.DIFF_TYPES}.")

        gp = tf.keras.Model(inputs,
                            gp_output_std,
                            name=name + "_gaussian_process")
        return gp

    @staticmethod
    def define_normal_distribution(xn,
                                   tilde_xn,
                                   step_size_,
                                   jac_par,
                                   sigma_theta, 
                                   diffusivity_type):
        """
        Defines a normal distribution based on the provided parameters.

        Parameters
        ----------
        xn : tensor
            The current state.
        tilde_xn : tensor
            The modified current state.
        step_size_ : float
            The step size for the Euler-Maruyama method.
        jac_par : dict
            Parameters for the Jacobian computation.
        sigma_theta : tensor
            Sigma matrix for the distribution.
        diffusivity_type : str
            Type of diffusivity matrix to use ('diagonal', 'triangular', 'spd').

        Returns
        -------
        tfd.Distribution
            A TensorFlow Probability distribution object.
        """

        #Define constant tensors for using in Jacobian computation
        sigma = tf.constant([jac_par['sigma']], dtype=tf.float64)
        r = tf.constant([jac_par['r']], dtype=tf.float64)
        beta = tf.constant([jac_par['beta']], dtype=tf.float64)
        
        #Get individual coordinates of xn
        if len(xn.shape) == 1:
            x = tf.reshape(xn[0], [1])
            y = tf.reshape(xn[1], [1])
            z = tf.reshape(xn[2], [1])
        else:
            x, y, z = xn[:,0], xn[:,1], xn[:,2]
        
        # Compute Jacobian entries evaluated at xn
        df1dx = -sigma * tf.ones_like(x)
        df1dy = sigma * tf.ones_like(x)
        df1dz = tf.zeros_like(x)
        df2dx = r - z
        df2dy = -tf.ones_like(x)
        df2dz = -x
        df3dx = y
        df3dy = x
        df3dz = -beta * tf.ones_like(x)
                
        # Creates Jacobian tensor
        J = tf.stack([
            tf.stack([df1dx, df1dy, df1dz], axis=1),
            tf.stack([df2dx, df2dy, df2dz], axis=1),
            tf.stack([df3dx, df3dy, df3dz], axis=1)
        ], axis=1)
        
        #Compute drift term
        drift_ = tf.linalg.matvec(J, tilde_xn)
        
        #Compute mean
        mean = tilde_xn + drift_ * step_size_
        
        # #Compute sigma_theta*tilde_xn
        # sigma_tilde_xn = tf.linalg.matvec(sigma_theta, tilde_xn)
        
        #Compute diffusion matrix: (sigma_theta*tilde_xn)*(sigma_theta*tilde_xn)^T
        # diff_matrix = tf.matmul(sigma_tilde_xn[:, tf.newaxis], sigma_tilde_xn[:, tf.newaxis], transpose_b=True)
        diff_matrix = sigma_theta

        if diffusivity_type == "diagonal":
            approx_normal = tfd.MultivariateNormalDiag(
                loc=(mean),
                scale_diag=tf.math.sqrt(step_size_) * diff_matrix,
                name="approx_normal"
            )
        elif diffusivity_type == "triangular":
            # form the normal distribution with a lower triangular matrix
            approx_normal = ModelBuilder.define_normal_distribution_triangular(tilde_xn, step_size_, drift_, diff_matrix)
        elif diffusivity_type == "spd":
            # form the normal distribution with SPD matrix
            approx_normal = ModelBuilder.define_normal_distribution_spd(tilde_xn, step_size_, drift_, diff_matrix)
        else:
            raise ValueError(
                f"Diffusivity type <{diffusivity_type}> not supported. Use one of {ModelBuilder.DIFF_TYPES}.")
        return approx_normal

    @staticmethod
    def define_normal_distribution_triangular(yn_, step_size_, drift_, diffusivity_tril_):
        """
        Constructs a normal distribution using a triangular diffusivity matrix.

        Parameters
        ----------
        yn_ : tensor
            Current points (batch_size x dimension).
        step_size_ : float
            Step sizes per point (batch_size x 1).
        drift_ : tensor
            Estimated drift at yn_.
        diffusivity_tril_ : tensor
            Estimated diffusivity matrix at yn_ (batch_size x n_dim x n_dim).

        Returns
        -------
        tfd.MultivariateNormalTriL
            A TensorFlow Probability Multivariate Normal distribution with a lower triangular scale matrix.
        """
        # a cumbersome way to multiply the step size scalar with the batch of matrices...
        # better use tfp.bijectors.FillScaleTriL()
        tril_step_size = tf.math.sqrt(step_size_)
        n_dim = K.shape(yn_)[-1]
        full_shape = n_dim * n_dim
        step_size_matrix = tf.broadcast_to(tril_step_size, [K.shape(step_size_)[0], full_shape])
        step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

        # now form the normal distribution
        approx_normal = tfd.MultivariateNormalTriL(
            loc=(yn_ + step_size_ * drift_),
            scale_tril=tf.multiply(step_size_matrix, diffusivity_tril_),
            name="approx_normal"
        )
        return approx_normal

    @staticmethod
    def define_normal_distribution_spd(yn_, step_size_, drift_, diffusivity_spd_):
        """
        Constructs a normal distribution using a Symmetric Positive Definite (SPD) diffusivity matrix.

        Parameters
        ----------
        yn_ : tensor
            Current points (batch_size x dimension).
        step_size_ : float
            Step sizes per point (batch_size x 1).
        drift_ : tensor
            Estimated drift at yn_.
        diffusivity_spd_ : tensor
            Estimated diffusivity matrix at yn_ (batch_size x n_dim x n_dim).

        Returns
        -------
        tfd.MultivariateNormalTriL
            A TensorFlow Probability Multivariate Normal distribution with an SPD scale matrix.
        """
        # a cumbersome way to multiply the step size scalar with the batch of matrices...
        # TODO: REFACTOR with diffusivity_type=="triangular"
        spd_step_size = tf.math.sqrt(step_size_)  # NO square root because we use cholesky below?
        n_dim = K.shape(yn_)[-1]
        full_shape = n_dim * n_dim
        step_size_matrix = tf.broadcast_to(spd_step_size, [K.shape(step_size_)[0], full_shape])
        step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

        # multiply with the step size
        covariance_matrix = tf.multiply(step_size_matrix, diffusivity_spd_)
        # square the matrix so that the cholesky decomposition does not change the eienvalues
        covariance_matrix = tf.linalg.matmul(covariance_matrix, tf.linalg.matrix_transpose(covariance_matrix))
        # perform cholesky to get the lower trianular matrix needed for MultivariateNormalTriL
        covariance_matrix = tf.linalg.cholesky(covariance_matrix)

        # now form the normal distribution
        approx_normal = tfd.MultivariateNormalTriL(
            loc=(yn_ + step_size_ * drift_),
            scale_tril=covariance_matrix,
            name="approx_normal"
        )
        return approx_normal
    
class LossAndErrorPrintingCallback(keras.callbacks.Callback):

    """
    A custom callback for use with Keras models to print loss and error information.

    This callback provides a method to log messages during training and can be customized
    to output additional information at the end of each epoch, training batch, or test batch.
    """

    @staticmethod
    def __log(message, flush=True):
        """
        Logs a message to standard output.

        This is a static method used internally by the callback to output messages.

        Parameters
        ----------
        message : str
            The message to be logged.
        flush : bool, optional
            If True, the output buffer is flushed immediately. Default is True.
        """

        sys.stdout.write(message)
        if flush:
            sys.stdout.flush()

    def on_train_batch_end(self, batch, logs=None):
        """
        Override to take actions at the end of a training batch.

        Parameters
        ----------
        batch : int
            The index of the batch within the current epoch.
        logs : dict, optional
            Contains the return value of `model.train_on_batch`. Typically, the keys 
            are the names of the model's metrics and the values are the corresponding 
            values for the batch.
        """
        pass

    def on_test_batch_end(self, batch, logs=None):
        """
        Override to take actions at the end of a test batch.

        Parameters
        ----------
        batch : int
            The index of the batch within the current epoch.
        logs : dict, optional
            Contains the return value of `model.test_on_batch`. Typically, the keys 
            are the names of the model's metrics and the values are the corresponding 
            values for the batch.
        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch.

        Logs the average loss for the epoch.

        Parameters
        ----------
        epoch : int
            The index of the epoch that ended.
        logs : dict, optional
            Dictionary containing the metrics results for this epoch.
            Typically, the keys are the names of the model's metrics and the values are 
            the corresponding average values for the entire epoch.
        """
        LossAndErrorPrintingCallback.__log(
            "\rThe average loss for epoch {} is {:7.10f} ".format(
                epoch, logs["loss"]
            )
        )

class SDEIdentification:
    """
    Wrapper class for identifying Stochastic Differential Equations (SDEs) using neural networks.

    This class is designed to work with a SDEApproximationNetwork. It can load a pre-trained model
    or a new model can be passed to it. It offers functionalities to train the model, evaluate 
    :math:'\\Sigma' matrix, sample new data points, and save the model.

    Parameters
    ----------
    model : SDEApproximationNetwork, optional
        A pre-initialized SDEApproximationNetwork model. If None, the model will be loaded from the provided path.
    path : str, optional
        The path to a trained model. Required if no model is provided.

    Attributes
    ----------
    model : SDEApproximationNetwork
        The SDE approximation network model used for identification.
    """

    def __init__(self, model=None, path=None):
        """
        Initializes the SDEIdentification instance with either a provided model or loads a model from a given path.
        """
        
        if model == None:
            
            assert(path is not None), "Model not provided. Please, provide a fresh model or path to trained model"
            
            #Load SDEApp dictionary
            with open(path + 'SDEApp_data.json', 'r') as infile:
                SDEApp = json.load(infile)
                
            step_size = SDEApp['step_size'] 
            jac_par = SDEApp['jac_par']
            method = SDEApp['method']
            diffusivity_type = SDEApp['diffusivity_type']
            
            #Load inner model (diff_network)
            loaded_diff_network = tf.keras.models.load_model(path + 'diff_network/')
            
            #Construct outer model (SDEApproximationNetwork)
            SDEApp_model = SDEApproximationNetwork(loaded_diff_network, step_size, jac_par, method, diffusivity_type)
            
            # #Load outer model weights
            # SDEApp_model.load_weights(path + 'SDEApp_model/')
            
            self.model = SDEApp_model
            
        else:
            
         self.model = model

    def train_model(self, xn, tilde_xn, tilde_xn1, step_size, validation_split=0.1, n_epochs=100, batch_size=1000, callbacks=[]):
        """
        Trains the SDEApproximationNetwork model.

        Parameters
        ----------
        xn : array-like
            The current state.
        tilde_xn : array-like
            The modified current state.
        tilde_xn1 : array-like
            The next state.
        step_size : float
            The step size for the Euler-Maruyama method.
        validation_split : float, optional
            The fraction of the data to be used as validation data. Default is 0.1.
        n_epochs : int, optional
            The number of epochs to train the model. Default is 100.
        batch_size : int, optional
            The number of samples per batch. Default is 1000.
        callbacks : list, optional
            List of callbacks to be used during training. Default is an empty list.

        Returns
        -------
        History
            A Keras History object containing the training history metrics.
        """
        print(f"training for {n_epochs} epochs with {int(xn.shape[0] * (1 - validation_split))} data points"
              f", validating with {int(xn.shape[0] * validation_split)}")

        features = np.column_stack([xn, tilde_xn])

        full_dataset = np.column_stack([features, tilde_xn1])
        
        if step_size is not None:
            full_dataset = np.column_stack([full_dataset, step_size])

        if len(callbacks) == 0:
            callbacks.append(LossAndErrorPrintingCallback())

        hist = self.model.fit(x=full_dataset,
                              epochs=n_epochs,
                              batch_size=batch_size,
                              verbose=0,
                              validation_split=validation_split,
                              callbacks=callbacks)
        return hist
    
    def eval_sigma(self, xn, tilde_xn):
        """
        Evaluates the :math:'\\Sigma' value (diffusivity matrix) for given state inputs.

        Parameters
        ----------
        xn : array-like
            The current state.
        tilde_xn : array-like
            The perturbation to current state.

        Returns
        -------
        array-like
            Approximation to :math:'\\Sigma' matrix.
        """
        
        sigma_theta = self.model.call_xn(xn, tilde_xn)
        return K.eval(sigma_theta)
    
    def sample_tilde_xn1(self,
                         xn,
                         tilde_xn,
                         step_size,
                         jac_par,
                         diffusivity_type):
        """
        Samples a new state (tilde_xn1) using the Euler-Maruyama scheme.

        Parameters
        ----------
        xn : array-like
            The current state.
        tilde_xn : array-like
            The modified current state.
        step_size : float
            The step size for the Euler-Maruyama method.
        jac_par : dict
            Parameters for the Jacobian computation.
        diffusivity_type : str
            The type of diffusivity matrix to use.

        Returns
        -------
        array-like
            The sampled next state perturbation (tilde_xn1).
        """
        sigma_theta = self.model.call_xn(xn, tilde_xn)

        approx_normal = ModelBuilder.define_normal_distribution(xn,
                                                                tilde_xn,
                                                                step_size,
                                                                jac_par,
                                                                sigma_theta,
                                                                diffusivity_type)

        tilde_xn1 = approx_normal.sample()
        return keras.backend.eval(tilde_xn1)
    
    def save_model(self,path):
        """
        Saves the SDEApproximationNetwork model and its parameters.

        Parameters
        ----------
        path : str
            The path where the model and its parameters will be saved.
        """
        
        SDEApp = {}
        SDEApp['step_size'] = self.model.step_size
        SDEApp['jac_par'] =  self.model.jac_par
        SDEApp['method'] = self.model.method
        SDEApp['diffusivity_type'] = self.model.diffusivity_type
        
        #Save dictionary with model data
        with open(path + 'SDEApp_data.json', 'w') as outfile:
            json.dump(SDEApp, outfile)
        
        #Save weights of outer model (SDEApproximationNetwork)       
        self.model.save(path + 'SDEApp_model/SDEApp_model_weights.h5')
        
        #Save inner model (diff_network)
        self.model.sde_model.save(path + 'diff_network')
        
        

class SDEApproximationNetwork(tf.keras.Model):
    """
    A TensorFlow Keras model designed for Stochastic Differential Equation (SDE) approximation.

    This model uses a specified neural network model (sde_model) to predict the Sigma matrix
    for a linearized Lorenz system. It is trained using an Euler-Maruyama scheme-based loss function.

    Parameters
    ----------
    sde_model : tf.keras.Model
        The neural network model used for predicting the Sigma matrix.
    step_size : float
        The step size used in the Euler-Maruyama method for SDE approximation.
    jac_par : dict
        Parameters for the Jacobian computation.
    method : str, optional
        The numerical method used for SDE approximation, default is 'euler'.
    diffusivity_type : str, optional
        The type of diffusivity matrix to be used, default is 'diagonal'.
    scale_per_point : float, optional
        Scaling factor applied per point, default is 1e-3.
    **kwargs
        Additional keyword arguments for the TensorFlow Keras model.

    Attributes
    ----------
    VALID_METHODS : list
        A list of valid numerical methods for SDE approximation.
    """
    VALID_METHODS = ["euler"]

    def __init__(self,
                 sde_model: tf.keras.Model,
                 step_size,
                 jac_par,
                 method="euler",
                 diffusivity_type="diagonal",
                 scale_per_point=1e-3,
                 **kwargs):
        """
        Initializes the SDEApproximationNetwork with the specified parameters.
        """
        super().__init__(**kwargs)
        self.sde_model = sde_model
        self.step_size = step_size
        self.jac_par = jac_par
        self.method = method
        self.diffusivity_type = diffusivity_type

        SDEApproximationNetwork.verify(self.method)

    @staticmethod
    def verify(method):
        """
        Verifies if the provided method is valid for SDE approximation.

        Parameters
        ----------
        method : str
            The numerical method to be verified.

        Raises
        ------
        ValueError
            If the method is not among the valid methods for SDE approximation.
        """
        if not (method in SDEApproximationNetwork.VALID_METHODS):
            raise ValueError(method + " is not a valid method. Use any of" + SDEApproximationNetwork.VALID_METHODS)

    def get_config(self):
        """
        Retrieves the configuration of the SDEApproximationNetwork.

        Returns
        -------
        dict
            A dictionary containing the configuration parameters of the model.
        """
        return {
            "sde_model": self.sde_model,
            "step_size": self.step_size,
            "method": self.method,
            "diffusivity_type": self.diffusivity_type
        }

    @staticmethod
    def euler_maruyama_pdf(xn, tilde_xn, tilde_xn1, step_size, jac_par, model_, diffusivity_type="diagonal"):
        """
        Computes the log probability density function for the Euler-Maruyama scheme.

        Parameters
        ----------
        xn : array-like
            The current state.
        tilde_xn : array-like
            The modified current state.
        tilde_xn1 : array-like
            The next state.
        step_size : float
            The step size for the Euler-Maruyama method.
        jac_par : dict
            Parameters for the Jacobian computation.
        model_ : tf.keras.Model
            The neural network model used for SDE approximation.
        diffusivity_type : str, optional
            The type of diffusivity matrix to be used, default is 'diagonal'.

        Returns
        -------
        tensor
            The log probability of transitioning from the current state to the next state.
        """
        
        sigma_theta = model_(tf.concat([xn, tilde_xn], axis=1)) #Call to the model defined in ModelBuilder.diff_network

        approx_normal = ModelBuilder.define_normal_distribution(xn,
                                                                tilde_xn,
                                                                step_size,
                                                                jac_par,
                                                                sigma_theta,
                                                                diffusivity_type)
        return approx_normal.log_prob(tilde_xn1)
    
    @staticmethod
    def split_inputs(inputs, step_size=None):
        """
        Splits the input tensor into components.

        Parameters
        ----------
        inputs : tensor
            The input tensor containing state variables and step size.
        step_size : float, optional
            The step size for the Euler-Maruyama method. If None, it is extracted from the inputs.

        Returns
        -------
        tuple
            A tuple of tensors (xn, tilde_xn, tilde_xn1, step_size).
        """
        
        n_total = inputs.shape[1]
        
        if step_size is not None:
            # Subtract one for the step_size at the end
            n_each = (n_total - 1) // 3
            x_n, tilde_xn, tilde_xn1, step_size = tf.split(inputs, num_or_size_splits=[n_each, n_each, n_each, 1], axis=1)
        else:
            n_each = n_total // 3
            x_n, tilde_xn, tilde_xn1 = tf.split(inputs, num_or_size_splits=[n_each, n_each, n_each], axis=1)
    
        return x_n, tilde_xn, tilde_xn1, step_size

    def call_xn(self, xn, tilde_xn):
        """
        Evaluates the diffusivity of the SDE model for given state inputs.

        Parameters
        ----------
        xn : tensor
            The current state.
        tilde_xn : tensor
            The perturbation to the current state.

        Returns
        -------
        tensor
            The evaluated diffusivity of the SDE model.
        """
        assert(len(xn.shape) == len(tilde_xn.shape)), "Shape dimension mismatch between xn and tilde_xn"
        if len(xn.shape) == 1:
            xn = xn.reshape((1,3))
            tilde_xn = tilde_xn.reshape((1,3))
            arguments = tf.concat([xn, tilde_xn], axis=1)
            #print(arguments.shape)
        else:
            arguments = tf.concat([xn, tilde_xn], axis=1)
        return self.sde_model(arguments)

    def call(self, inputs):
        """
        Processes the inputs and computes the loss for the model.

        Parameters
        ----------
        inputs : tensor
            The input tensor containing all of (xn, tilde_xn, tilde_xn1, step_size).

        Returns
        -------
        tensor
            The output of the neural network model predicting the Sigma matrix.
        """
        xn, tilde_xn, tilde_xn1, step_size = SDEApproximationNetwork.split_inputs(inputs, self.step_size)

        if self.method == "euler":
            log_prob = SDEApproximationNetwork.euler_maruyama_pdf(xn, tilde_xn, tilde_xn1, step_size, self.jac_par, self.sde_model,
                                                                  diffusivity_type=self.diffusivity_type)

        else:
            raise ValueError(self.method + " not available")

        sample_distortion = -tf.reduce_mean(log_prob, axis=-1)
        distortion = tf.reduce_mean(sample_distortion)

        loss = distortion

        # correct the loss so that it converges to zero regardless of dimension
        loss = loss + 2 * np.log(2 * np.pi) / np.log(10) * xn.shape[1]

        self.add_loss(loss)
        self.add_metric(distortion, name="distortion", aggregation="mean")

        return self.sde_model(tf.concat([xn, tilde_xn], axis=1))
    