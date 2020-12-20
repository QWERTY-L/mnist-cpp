# mnist-cpp
A C++ neural network to detect the values of the handwritten mnist digits

## Usage
Compile the code with gcc using `g++ .\MnistNN.cpp -O3`. (The -O3 flag is optional but it makes the program run significantly faster). Next run a.exe and the program will start training. The program runs for 20 epochs by default with 10 batches per epoch. Gradient descent is performed each batch on 2000 images by default. After it is done training, you can train some more for a custom amount of time using `train [epochs] [batches per epoch] [samples per batch]` or just typing `train` and following the prompts.

## Commands

Note that for all commands you can type their name without arguments and you will be prompted to enter the arguments. For example, instead of using `print [file] [index]` you could just type `print`. Make sure to take out the square brackets!

- `exit` - Ends the program (alias: `end`)
- `print [file name] [training image index]` - Prints image value from the training set and saves it to file (automatically appends .bmp to the end of the file name) (alias: `TrainPrint`)
- `t_print [file name] [test image index]` - Prints image value from the test set and saves it to file (automatically appends .bmp to the end of the file name) (alias: `TestPrint`)
- `eval [training image index]` - Predicts the value of an image from the training set (alias: `TrainEval`) - The accuracy of the prediction depends on the accuracy of the NN
- `t_eval [test image index]` - Predicts the value of an image from the test set (alias: `TestEval`)
- `train [number of epochs] [batches per epochs] [samples per batch]` - Trains based on the inputed parameters

More commands will be added to this list at a future time, for now check the "input output" section of MnistNN.cpp for more information.

## Neural Network Structure

- Input: 784 array of pixel intensities
- Hidden Layer: 256 neurons with tanh activation
- Output Layer: 10 neurons with softmax activation

Loss function: Categorical Cross-Entropy (MSE is also an option but it trains much slower).

This is highly customizable so feel free to mess around with the code and explore the results.
