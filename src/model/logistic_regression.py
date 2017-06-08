# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1]+1)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        from util.loss_functions import BinaryCrossEntropyError
        loss = BinaryCrossEntropyError()

        learned = False
        iteration = 0
        grad = np.zeros(len(self.weight))


        minError =0;
        minWeights = self.weight

        # Train for some epochs if the error is not 0
        while not learned:
            totalError = 0
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):


                #append the bias and set it to one
                input = np.append([1], input)
                input[0] = 1
                
                output = self.fire(input)

                error = loss.calculateError(label, output)

                deltaError = label - output

                grad = grad + input * deltaError

                #z = np.dot(np.array(input), self.weight)
                #grad = grad + deltaError * Activation.sigmoidPrime(z) * input

                totalError += error

            self.updateWeights(grad / len(self.trainingSet.input))

            #test for minimal error
            if iteration == 1:
                minError = totalError
                minWeights = self.weight

            if totalError < minError:
                #the current weights have a better fittness than all the previous ones
                minError = totalError
                minWeights = self.weight
            
            iteration += 1
            
            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, totalError)
            
            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True

        #use the weiths with the bests error
        self.weight = minWeights
        if verbose:
            logging.info("Min error: %i", minError)
        
        
    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        #append the bias and set it to one
        testInstance = np.append([1], testInstance)
        testInstance[0] = 1;

        x = self.fire(testInstance)
        if x > 0.5:
            return True
        else:
            return False

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.



        return list(map(self.classify, test))

    def updateWeights(self, grad):
        self.weight = self.weight + (grad * self.learningRate)


    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
