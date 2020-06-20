try:
    from .data import data_in, data_out, data_test
    from .modules import NeuralNetwork
except ImportError:
    from data import data_in, data_out, data_test
    from modules import NeuralNetwork

if __name__ == "__main__":
    # Initialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)
    print()

    # The training set. Each consisting of 4 input values
    # and 1 output value.
    training_set_inputs = data_in
    training_set_outputs = data_out

    # Train the neural network using a training set.
    # Do it 100,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10 ** 5)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)
    print()

    # Test the neural network with a new situation.
    test_kit = data_test
    for test_case in test_kit:
        predicted = neural_network.think(test_case)
        predicted_rounded = predicted.round()
        print(f"Predicting new situation {test_case} -> ?: {predicted} ({predicted_rounded})")
