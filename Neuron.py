# import packages
import random
import math


# ------------------------------- Class Neuron -------------------------------
class Neuron:

    def __init__(self, input_number: int, learning_rate: float, activation: str):
        self.weights = [random.random() for _ in range(input_number)]
        self.bias = random.random()
        self.learning_rate = learning_rate
        self.activation_type = activation

    # ----------- Linear Function -----------
    def forward(self, inputs: list[float]) -> float:
        self.last_z = sum(x * w for x,w in zip(inputs, self.weights)) + self.bias
        self.last_inputs, self.last_output = inputs, self.activation(self.last_z)
        return self.last_output
    
    # ----------- Activation Function -----------
    def activation(self, z: float) -> float:
        if self.activation_type == "sigmoid":
            return 1 / (1 + math.exp(-z))
        elif self.activation_type == "relu":
            return max(0,z)
        else:
            raise ValueError(f"[Neuron Istantation] Unsupported activation function: {self.activation_type}!!!")
        
    def activation_derivative(self, z: float, output: float) -> float:
        if self.activation_type == "sigmoid":
            return output * (1 - output)
        elif self.activation_type == "relu":
            return 1.0 if z > 0 else 0.0
    
    # ----------- Gradient Descendt -----------
    def backward(self, dLoss_df: float) -> list[float]:
        df_dz = self.activation_derivative(self.last_z, self.last_output)
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

    def __init__(self, Neuron_number: int, input_number: int, learning_rate = 0.001, activation = "sigmoid"):
        self.neurons = [Neuron(input_number, learning_rate, activation) for _ in range(Neuron_number)]

    # ----------- Linear Function -----------
    def forward(self, inputs: list[float]) -> list[float]:
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

    def __init__(self, layers: list[Layer], loss_correction = "mse", output_activation = "none"):
        self.layers = layers
        self.output_activation = output_activation
        self.loss_correction = loss_correction

    # ----------- Linear Function -----------
    def forward(self, inputs: list[float]) -> list[float]:
        data = inputs
        for layer in self.layers:
            data = layer.forward(data)

        # softmax output-activation if selected
        if self.output_activation == "softmax":
            output_exponential = [math.exp(y) for y in data]
            summation = sum(output_exponential)
            data = [y / summation for y in output_exponential]
        
        # return output data
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
                if self.loss_correction == "mse":
                    loss += sum((result[i] - target[i])**2 for i in range(len(result)))
                    dLoss_dResult = [2 * (result[i] - target[i]) for i in range(len(result))]
                elif self.loss_correction == "binary_crossentropy":
                    p = min(max(result[0], 1e-15), 1 - 1e-15)
                    loss -= (target[0] * math.log(p)) + ((1 - target[0]) * math.log(1 - p))
                    dLoss_dResult = [result[0] - target[0]]
                elif self.loss_correction == "crossentropy":
                    loss -= sum((target[i] * math.log(result[i] + 1e-15)) for i in range(len(result)))
                    dLoss_dResult = [(result[i] - target[i]) for i in range(len(result))]
                else:
                    raise ValueError(f"[Network Istantation] Unsupported loss correction: {self.loss_correction}!!!")

                # upgrade the Network backwards
                self.backward(dLoss_dResult)

            # update the Loss-in-Time
            total_loss.append(loss)

        # end of training
        return total_loss