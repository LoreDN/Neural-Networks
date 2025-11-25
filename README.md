# 🧪 Neural Networks
This repository is a collection of all my projects about **Neural Netwroks**, starting from simple ones wich requires only one **Neuron** and not a real Network, going to more complex ones.<br>

## 🧠 Neuron Implementation
First of all I had to implement a general and portable structure for the **Neuron** and the various **Layers**, in order to use them for different Networks and don't have the need to start from scratch for each Project.<br>
<br>
In order to realized this, I implemented the **Neuron** using the following structure:
- each **Neuron** can take an arbitrary number of inputs ( set when creating the Neuron instantation ).
- there is only one output, which is calculated via a **Linear Function** $z(\vec{in},\vec{w})$ and then evaluated via an **Activation Function** $f(z)$.
- the *Linear function* is based on a *weighted average* --> $z(\vec{in},\vec{w}) = \sum_i (input_i \cdot weight_i) + bias$.
- the *Acceptance Function* is based on the $sigmoid(z)$ --> $f(z) = \frac{1}{1 + e^{-z}}$.
- at the end each **Neuron** will create a single output, passed to the next **Layer** or used as a result.

## ⚙️ Training the Network
The previous documented **Neuron** is the core for a **Neural Network**, which is composed by different **Layers** of **Neurons**.<br>
In order to train the Network efficiently, I implemented the *Gradient Descendt* Algorithm, which confronts the results of the Network with the ones given by the Dataset, calculates the **Gradient** of the Loss, then update the *weights* and the *bias* of each **Neuron** recursively.<br>
In this way the Network will train on the given Dataset for an arbitrary number of *epochs*, and will return a list which describes the evolution of the **Loss-in-Time**.

## ✔️ Currently Implemented Neural Networks
In this repository can be found the real implementation of the Network ( as the file **Neuron.py** ), and can be also found all the **Neural Networks** I have created during time ( as *Jupyter Notebooks* ) together with some plots which describes the *Training results*.<br>
<br>
Here are them all:
- **cluster/ :** folder containing Networks which classify different elements by *Classes*.
  - **point/ :** cluster of a Point in the Cartesian Plane following a Boundary.
    - Line_2D
    - Curve_2D
    - Circle_2D
    - Ring_2D
  - **figures/ :** cluster of different figures.
    - Moons_2D
    - Spirals_2D