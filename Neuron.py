# import packages
import random
import math


# ------------------------------- Class Neuron -------------------------------
class Neuron:

    def __init__(self, input_number: int, learning_rate = 0.001):
        self.weights = [random.random() for _ in range(input_number)]
        self.bias = random.random()
        self.learning_rate = learning_rate

    # ----------- Linear Function -----------
    def forward(self, inputs: list[float]) -> float:
        z = sum(x * w for x,w in zip(inputs, self.weights)) + self.bias
        self.last_inputs, self.last_output = inputs, self.activation(z)
        return self.last_output
    
    # ----------- Acceptance Function -----------
    def activation(self, z: float) -> float:
        return 1 / (1 + math.exp(-z))
    
    # ----------- Gradient Descendt -----------
    def backward(self, dLoss_df: float) -> list[float]:
        df_dz = self.last_output * (1 - self.last_output)
        dLoss_dz = dLoss_df * df_dz
        dLoss_dInputs = [dLoss_dz * w for w in self.weights]

        # update the weights
        for i in range(len(self.weights)):
            dLoss_dWeight = dLoss_dz * self.last_inputs[i]
            self.weights[i] -= self.learning_rate * dLoss_dWeight

        # update the bias
        self.bias -= self.learning_rate * dLoss_dz
        return dLoss_dInputs
    

# -------------------------------- Class Layer -------------------------------
class Layer:

    def __init__(self, Neuron_number: int, input_number: int, learning_rate = 0.001):
        self.neurons = [Neuron(input_number, learning_rate) for _ in range(Neuron_number)]

    # ----------- Linear Function -----------
    def forward(self, inputs: list[float]) -> float:
        return [neuron.forward(inputs) for neuron in self.neurons]
    
    # ----------- Gradient Descendt -----------
    def backward(self, gradients: list[float]) -> list[float]:
        prev_gradients = [0.0] * len(self.neurons[0].weights)

        # calculate previous Layer gradients
        for i, neuron in enumerate(self.neurons):
            grad = neuron.backward(gradients[i])
            for j in range(len(prev_gradients)):
                prev_gradients[j] += grad[j]

        # return previous Layer gradients
        return prev_gradients


# ------------------------------- Class Network ------------------------------
class Network:

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    # ----------- Linear Function -----------
    def forward(self, inputs: list[float]) -> list[float]:
        data = inputs
        for layer in self.layers:
            data = layer.forward(data)
        return data  
    
    # ----------- Gradient Descendt -----------
    def backward(self, loss_gradient: list[float]):
        gradients = loss_gradient
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)

    # ----------- Train the Network -----------
    def train(self, dataset: list[tuple[list[float], list[float]]], epochs: int) -> list[float]:
        total_loss = []
        for _ in range(epochs):
            loss = 0
            for inputs, target in dataset:
                
                # calculate the Loss
                result = self.forward(inputs)
                dLoss_dResult = [2 * (result[i] - target[i]) for i in range(len(result))]
                loss += sum((result[i] - target[i])**2 for i in range(len(result)))

                # upgrade the Network backwards
                self.backward(dLoss_dResult)

            # update the Loss-in-Time
            total_loss.append(loss)

        # end of training
        return total_loss