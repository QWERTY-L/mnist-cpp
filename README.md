# mnist-cpp
A C++ neural network to detect the values of the handwritten mnist digits

## Usage
Compile the code with gcc using `g++ .\MnistNN.cpp -O3`. (The -O3 flag is optional but it makes the program run significantly faster). Next run a.exe and the program will start training. The program runs for 20 epochs by default with 10 batches per epoch. Gradient descent is performed each batch on 2000 images by default. After it is done training, you can train some more for a custom amount of time using `train [epochs] [batches per epoch] [samples per batch]` or just typing `train` and following the prompts.

## Commands
Comming Soon

## Neural Network Structure

Input: 784 array of pixel intensities
Hidden Layer: 256 neurons with tanh activation
Output Layer: 10 neurons with softmax activation

Loss function: Categorical Cross-Entropy (MSE is also an option but it trains much slower).

This is highly customizable so feel free to mess around with the code to get explore results.
