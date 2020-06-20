from numpy import exp, ndarray, random, dot


class NeuralNetwork:
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 4 input connections and 1 output connection.
        # We assign random weights to a 4 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((4, 1)) - 1

    @staticmethod
    def __sigmoid(x: ndarray) -> ndarray:
        """
        The Sigmoid function, which describes an S shaped curve.
        We pass the weighted sum of the inputs through this function to
        normalise them between 0 and 1.
        """
        return 1 / (1 + exp(-x))

    @staticmethod
    def __sigmoid_derivative(x: ndarray) -> ndarray:
        """
        The derivative of the Sigmoid function.
        This is the gradient of the Sigmoid curve.
        It indicates how confident we are about the existing weight.
        """
        return x * (1 - x)

    def train(self,
              training_set_inputs: ndarray,
              training_set_outputs: ndarray,
              number_of_training_iterations: int) -> None:
        """
        We train the neural network through a process of trial and error.
        Adjusting the synaptic weights each time.
        """
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    def think(self, inputs: ndarray) -> ndarray:
        """
        The neural network thinks
        """
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
