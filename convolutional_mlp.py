

import os       
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from load import loadData
from mlp import HiddenLayer

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        #number of input feature maps is the same LOL its the same damn variable
        #for the first layer, the number of feature maps is equal to the number of channels in the image.
        #
        assert image_shape[1] == filter_shape[1] #filter_shape is used only for the first layer, I believe.
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps (aka num filters) * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

def evaluate_lenet5(learning_rate=0.1, n_epochs=1000,
                    path='/Users/Davis/Desktop/dataset', 
                    nkerns=[20, 30, 40], batch_size=10):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    trainPath = path + '/train/*jpg'    #built path to all 3 folders for all images included.
    testPath = path + '/test/*jpg'      #these are correctly written
    validPath = path + '/valid/*jpg'    # validPath = '/Users/Davis/Desktop/dataset/valid/*jpg'

    rng = numpy.random.RandomState(23455)   #seed your random number generator

    train_set_x, train_set_y = loadData(trainPath)  
    valid_set_x, valid_set_y = loadData(validPath)
    test_set_x, test_set_y = loadData(testPath)



    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model with ' + os.path.split(path)[1]

    # Reshape matrix of rasterized images of shape (batch_size, 200 * 200)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (200, 200) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 100, 100))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (100-5+1 , 100-5+1) = (96, 96)
    # maxpooling reduces this further to (96/2, 96/2) = (48, 48)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 48, 48)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 100, 100),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (48-5+1, 48-5+1) = (44, 44)
    # maxpooling reduces this further to (44/2, 44/2) = (22, 22)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 22, 22)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 48, 48),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # Construct the third convolutional pooling layer
    # filtering reduces the image size to (22-5+1, 22-5+1) = (18, 18)
    # maxpooling reduces this further to (18/2, 18/2) = (9, 9)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 9, 9)
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 22, 22),
        filter_shape=(nkerns[2], nkerns[1], 5, 5),
        poolsize=(2, 2)
    )


    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 9 * 9),
    # or (500, 30 * 9 * 9) = (500, 2430) with the default values.
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * 9 * 9,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(
        input=layer3.output, 
        n_in=500, 
        n_out=2
    )

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # the data above will do all the work required. However, here is what the following block of code means
    # we are defining a function. Given some lists of indexes, the ERRORS will be calculated. How the errors
    # are calculated are described in the code above OUTSIDE of the function definition. This is useful
    # such that we can separate batch splitting and inputs from the rest of the code... We can also define multiple
    # functions over the entire feed forward network and just pull variables from within it as it runs! cool!
    # this is all made possible by the theano.function().
    #
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf #nothing will be > inf, so the worst case is having infinite loss (infinite error)
    best_iter = 0   #the best iteration will update
    test_score = 0.     #so will the best test score (higher is better)
    start_time = time.clock()   #the starting time is set here, later endtime-starttime = elapsed_time

    epoch = 0   #we start at zero epochs. Remember that each epoch equals 1 iteration through the ENTIRE training set.
    done_looping = False    #we are NOT done looping!

    while (epoch < n_epochs) and (not done_looping): #if n_epochs has been reached we stop; if done_looping is triggered we stop.
        epoch = epoch + 1   #everytime we reach this line, ONE epoch has been completed 
                            #note that ITERATIONS mean each MINI-BATCH. For example, for 120 images, 
                            #there are 12 iterations and 1 epoch per run through.
        for minibatch_index in xrange(n_train_batches): #iterates through INDEXES for MINI-BATCHES

            iter = (epoch - 1) * n_train_batches + minibatch_index #current iteration count. We verified this before.

            if iter % 100 == 0: #every 100 mini-batches trained, we print how many mini-batches have been trained!
                print 'training @ iter = ', iter

            cost_ij = train_model(minibatch_index)  #the theano function outputs the cost from the last layer (logistic regression X-Ent Cost)
                                                    #note that all updates are calculated above along with backprop errors.

            if (iter + 1) % validation_frequency == 0:  #the rest of the code is for early stopping.

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
