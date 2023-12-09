Network_DietrichBased
=======================

Here is the documentation for the Network based on Dietrich's Paper.

The first step is to use ModelBuilder to create the neural network (encoder) object for approximating 
:math:'\Sigma'. Next, a SDEApproximationNetwork object must be used to wrap the neural network. 
Finally, that must be wrapped in a SDEIdentification object, which is then used to train the network
and, later on, use it to sample a path.

.. automodule:: Network_DietrichBased
   :members: