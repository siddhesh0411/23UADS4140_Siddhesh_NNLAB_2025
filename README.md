Objective 4
Write a Python program to test how a simple three-layer neural network performs when we change things like the activation function, hidden layer size, learning rate, batch size, and number of training rounds (epochs).
Model Overview:

Training Combinations:
The model is trained with various batch sizes (1, 10, 100) and different numbers of epochs (10, 50, 100) to observe how performance changes.

Data Preparation:
Before training, the image data goes through a few steps:

The pixel values are scaled between 0 and 1.

Each 28x28 image is reshaped into a one-dimensional array of 784 values.

Labels (digits) are transformed into one-hot encoded vectors — for example, the digit 2 becomes [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].

Model Architecture:
Input Layer:
Contains 784 neurons, one for each pixel in the input image.

Hidden Layer:
Contains 256 neurons.

Uses ReLU as the activation function.

Weights are initialized using Xavier initialization for better convergence.

Output Layer:
Has 10 neurons, each representing one digit from 0 to 9.

Outputs raw scores (logits) without applying softmax yet.

Loss Function and Optimizer:
The loss function used is softmax cross-entropy, which compares the model’s predictions to the actual one-hot encoded labels.

Adam optimizer is chosen for training, with a learning rate set at 0.1.

Training Process:
The function train_step handles one pass through the data, calculates the loss, computes gradients using TensorFlow's GradientTape, and updates weights.

The model is trained over several epochs using mini-batches of a chosen size.

A progress bar is displayed to show training progress within each epoch.

Training can be paused using Ctrl+C and resumed by pressing Enter.

Model Evaluation:
After every epoch, the model’s accuracy is tested using the separate test dataset.

Predictions are made by passing logits through softmax and picking the class with the highest score.

The accuracy is stored for performance tracking.

Visualizations:
After training:

A loss curve is plotted to show how the model's error decreased over time.

An accuracy curve shows how performance improved during training.

A confusion matrix is created to highlight which digits the model confused with others, shown as a heatmap.

Personal Observations:
While testing, i saw some interesting patterns. The highest test accuracy (33.89%) was achieved with a batch size of 10 and 10 epochs. On the other hand, the lowest accuracy (9.58%) happened when the batch size was 1 and epochs were set to 100.

This shows that simply training the model using very small batch sizes doesn’t always lead to better performance. Instead, there’s an ideal balance where the model performs best — using more epochs or smaller batches without adjusting other settings can make the model perform worse.
