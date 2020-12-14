/*
 * Created December 2020
 * Author: Luke Sequeira
 *
 * Copyright (c) 2020 Luke Sequeira
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <tuple>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include "MnistParser.hpp"
#include "BMPArrayConvert.hpp"
#include "NNfunctions.hpp"

typedef unsigned char ubyte;
typedef std::vector<double> d_vec;
typedef std::vector<d_vec> d_mat;
typedef unsigned int uint; //unsigned int
typedef std::vector<std::vector<uint>> i_mat; //integer matrix
typedef std::vector<uint> i_vec; //integer array

int main()
{
    
    /*
    ██████   █████  ████████  █████      ██████  ██████  ███████     ██████  ██████   ██████   ██████ ███████ ███████ ███████ ██ ███    ██  ██████  
    ██   ██ ██   ██    ██    ██   ██     ██   ██ ██   ██ ██          ██   ██ ██   ██ ██    ██ ██      ██      ██      ██      ██ ████   ██ ██       
    ██   ██ ███████    ██    ███████     ██████  ██████  █████ █████ ██████  ██████  ██    ██ ██      █████   ███████ ███████ ██ ██ ██  ██ ██   ███ 
    ██   ██ ██   ██    ██    ██   ██     ██      ██   ██ ██          ██      ██   ██ ██    ██ ██      ██           ██      ██ ██ ██  ██ ██ ██    ██ 
    ██████  ██   ██    ██    ██   ██     ██      ██   ██ ███████     ██      ██   ██  ██████   ██████ ███████ ███████ ███████ ██ ██   ████  ██████                                                                                                                                      
    */

    //get image data
    auto TrainImagesTuple = MnistParse::ParseMnistImages("Data/train-images.idx3-ubyte");
    auto PrintingTrainImages = std::get<0>(TrainImagesTuple); //Train Images - Vector of matrixes of images. Dimensions: N x 28 x 28
    auto FlatTrainImages = std::get<1>(TrainImagesTuple); //Train Images - Vector of flattened image data. Dimensions: N x 784

    auto rawLabels = MnistParse::ParseMnistLabels("Data/train-labels.idx1-ubyte"); //Raw train Labels

    auto TestImagesTuple = MnistParse::ParseMnistImages("Data/t10k-images.idx3-ubyte");
    auto PrintingTestImages = std::get<0>(TestImagesTuple); //Test Images - Vector of matrixes of images. Dimensions: N x 28 x 28
    auto FlatTestImages = std::get<1>(TestImagesTuple); //Test Images - Vector of flattened image data. Dimensions: N x 784

    auto rawTestLabels = MnistParse::ParseMnistLabels("Data/t10k-labels.idx1-ubyte"); //Raw test Labels

    //Make labels into one-hot encoded vector
    d_mat Labels(rawLabels.size(), d_vec(10)); //Train labels - One hot encoded vectors ranging from 0 to 9
    for (int i = 0; i < rawLabels.size(); i++) //Convert raw data into one hot encoded vectors
    {

        d_vec n0 = {1,0,0,0,0,0,0,0,0,0};
        d_vec n1 = {0,1,0,0,0,0,0,0,0,0};
        d_vec n2 = {0,0,1,0,0,0,0,0,0,0};
        d_vec n3 = {0,0,0,1,0,0,0,0,0,0};
        d_vec n4 = {0,0,0,0,1,0,0,0,0,0};
        d_vec n5 = {0,0,0,0,0,1,0,0,0,0};
        d_vec n6 = {0,0,0,0,0,0,1,0,0,0};
        d_vec n7 = {0,0,0,0,0,0,0,1,0,0};
        d_vec n8 = {0,0,0,0,0,0,0,0,1,0};
        d_vec n9 = {0,0,0,0,0,0,0,0,0,1};

        if(rawLabels[i] == 0){

            Labels[i] = n0;

        }
        if(rawLabels[i] == 1){

            Labels[i] = n1;

        }
        if(rawLabels[i] == 2){

            Labels[i] = n2;

        }
        if(rawLabels[i] == 3){

            Labels[i] = n3;

        }
        if(rawLabels[i] == 4){

            Labels[i] = n4;

        }
        if(rawLabels[i] == 5){

            Labels[i] = n5;

        }
        if(rawLabels[i] == 6){

            Labels[i] = n6;

        }
        if(rawLabels[i] == 7){

            Labels[i] = n7;

        }
        if(rawLabels[i] == 8){

            Labels[i] = n8;

        }
        if(rawLabels[i] == 9){

            Labels[i] = n9;

        }
        

    }
    
    //normalize Image Data
    FlatTrainImages = smult(1.0/255, FlatTrainImages);
    FlatTestImages = smult(1.0/255, FlatTestImages);

    /*
    ███    ██ ███████ ██    ██ ██████   █████  ██          ███    ██ ███████ ████████ ██     ██  ██████  ██████  ██   ██ 
    ████   ██ ██      ██    ██ ██   ██ ██   ██ ██          ████   ██ ██         ██    ██     ██ ██    ██ ██   ██ ██  ██  
    ██ ██  ██ █████   ██    ██ ██████  ███████ ██          ██ ██  ██ █████      ██    ██  █  ██ ██    ██ ██████  █████   
    ██  ██ ██ ██      ██    ██ ██   ██ ██   ██ ██          ██  ██ ██ ██         ██    ██ ███ ██ ██    ██ ██   ██ ██  ██  
    ██   ████ ███████  ██████  ██   ██ ██   ██ ███████     ██   ████ ███████    ██     ███ ███   ██████  ██   ██ ██   ██                                                                                                                  
    */

    std::cout.precision(17);
	srand((unsigned int)time(NULL));

    /*
    *
    * NN format:
    * Input (784 Neurons)
    * Hidden Layer (256 Neurons) -> Activation: Tanh
    * Output (10 Neurons) -> Activation: Softmax
    * 
    */
    
   	//initialise weights and biases
	std::vector<std::vector<double>> w1(256, std::vector<double>(784, 0)); //format (L_i, (L_i-1, 0)) where L_i is the wieghts in the current layer and L_i-1 is from the previous layer
	std::vector<double> b1(256, 0); //format (L_i, 0)
	std::vector<std::vector<double>> w2(10, std::vector<double>(256, 0)); //format (L_i, (L_i-1, 0))
	std::vector<double> b2(10, 0); //format (L_i, 0)

    //Randomly initialize
    for(int i=0;i<w1.size();i++){

		for(int j=0;j<w1[0].size();j++){

			w1[i][j] = 2 * ((float) rand()/RAND_MAX) - 1;

		}

	}

	for(int j=0;j<b1.size();j++){

		b1[j] = 2 * ((float) rand()/RAND_MAX) - 1;

	}

	for(int i=0;i<w2.size();i++){

		for(int j=0;j<w2[0].size();j++){

			w2[i][j] = 2 * ((float) rand()/RAND_MAX) - 1;

		}

	}

	for(int j=0;j<b2.size();j++){

		b2[j] = 2 * ((float) rand()/RAND_MAX) - 1;

	}

    //Choose activation functions
	std::vector<std::vector<double>>(*act1)(std::vector<std::vector<double>>); //Unfortunatly, auto doesn't work here :(
	std::vector<std::vector<double>>(*act2)(std::vector<std::vector<double>>);
    std::vector<std::vector<double>>(*dact1)(std::vector<std::vector<double>>);
	std::vector<std::vector<double>>(*dact2)(std::vector<std::vector<double>>);
    d_mat(*loss_f)(d_mat, d_mat);
    d_mat(*Dloss_f)(d_mat, d_mat);
	act1 = tanh;
	act2 = softmax;
    dact1 = dtanh;
    dact2 = f_dsoftmax; //working version of dsoftmax
    loss_f = categorical_cross_entropy;
    Dloss_f = Dcategorical_cross_entropy; //switch to dcategorical-cross-entropy

    //Set gd variables
	double alpha = 0.5;
    int epochs = 20;
	int batches = 10;
    int batchSize = 2000;
    int begin;
	std::vector<std::vector<double>> loss;
    std::vector<std::vector<double>> Batch(batchSize, d_vec(FlatTrainImages[0].size())); //Data for the batch

	//intialize derivatives
	std::vector<std::vector<double>> dw1 = w1;
	std::vector<double> db1 = b1;
	std::vector<std::vector<double>> dw2 = w2;
	std::vector<double> db2 = b2;

    for (int j = 0; j < epochs; j++)
    {
        
        std::cout << std::endl << "EPOCH " << j + 1 << std::endl;
        std::cout << "doing backprop..." << std::endl;

        for (int i = 0; i < batches; i++) {
            begin = (rand() % (FlatTrainImages.size() - batchSize));
            auto X = d_mat(FlatTrainImages.begin() + begin, FlatTrainImages.begin() + begin + batchSize); //Choose batchSize random elements from X (drawback: elements are consecutive; use some form of shuffling for future projects)
            auto y = d_mat(Labels.begin()  + begin, Labels.begin() + begin + batchSize);
            auto r_vals = GradientDescent(batchSize, alpha, X, y, w1, b1, w2, b2, act1, act2, dw1, db1, dw2, db2, dact1 , dact2, loss_f, Dloss_f);
            w1 = std::get<0>(r_vals);
            w2 = std::get<1>(r_vals);
            b1 = std::get<2>(r_vals);
            b2 = std::get<3>(r_vals);
            loss = std::get<4>(r_vals);

        }

        std::cout << "Loss: ";
        printList(transpose(loss));

        double ac = 0;
        //evaluate accuracy
        std::cout << "Accuracy (on 10,000 test samples): ";
        int acc_samples = 10000;
        for (int k = 0; k < acc_samples; k++)
        {

            auto p1 = max(transpose(h(h(d_mat{FlatTestImages[k]}, w1, b1, act1), w2, b2, act2)));
            if(p1 == ((int) rawTestLabels[k])){ac++;}
            
        }
        ac /= acc_samples;
        std::cout << ac;

    }


    /*
    ██ ███    ██ ██████  ██    ██ ████████      ██████  ██    ██ ████████ ██████  ██    ██ ████████     ███████ ██    ██ ███████ ████████ ███████ ███    ███ 
    ██ ████   ██ ██   ██ ██    ██    ██        ██    ██ ██    ██    ██    ██   ██ ██    ██    ██        ██       ██  ██  ██         ██    ██      ████  ████ 
    ██ ██ ██  ██ ██████  ██    ██    ██        ██    ██ ██    ██    ██    ██████  ██    ██    ██        ███████   ████   ███████    ██    █████   ██ ████ ██ 
    ██ ██  ██ ██ ██      ██    ██    ██        ██    ██ ██    ██    ██    ██      ██    ██    ██             ██    ██         ██    ██    ██      ██  ██  ██ 
    ██ ██   ████ ██       ██████     ██         ██████   ██████     ██    ██       ██████     ██        ███████    ██    ███████    ██    ███████ ██      ██                                                                                                                                                
    */

   std::cout << std::endl << "Entering Input-Output mode: " << std::endl;

   while(true){

       std::cout << std::endl;
       std::string command; //command to parse
       std::cout << ">>> ";

       //take input
       std::cin >> command; //remember the 'arrows' in cin go in the opposite direction!
       std::cout << std::endl;

       //PARSE COMMANDS

       //exit (alias: end)
       if(command == "exit" || command == "end"){

           std::cout << "Exiting the program";
           return 0;

       }

       //print (alias: TrainPrint)
       if (command == "print" || command == "TrainPrint")
       {
           std::string name;
           int index;
           std::cout << "Printing a training set image to file. To print a test set image, use 't_print' OR 'TestPrint'";
           std::cout << std::endl << "What should the filename be" << std::endl << "Input > ";
           std::cin >> name;
           std::cout << std::endl << "What index in the training images do you want to print (from 0 to 59999)" << std::endl << "Input > ";
           std::cin >> index;
           BMP::arrayToBMP((name + ".bmp"), PrintingTrainImages[index], PrintingTrainImages[index], PrintingTrainImages[index]);
           std::cout << std::endl << "Image is a " << rawLabels[index] << std::endl << "Image Saved to " << name << ".bmp";

       }

       //t_print (alias: TestPrint)
       if (command == "t_print" || command == "TestPrint")
       {
           std::string name;
           int index;
           std::cout << "Printing a test set image to file. To print a training set image, use 'print' OR 'TrainPrint'";
           std::cout << std::endl << "What should the filename be" << std::endl << "Input > ";
           std::cin >> name;
           std::cout << std::endl << "What index in the training images do you want to print (from 0 to 9999)" << std::endl << "Input > ";
           std::cin >> index;
           BMP::arrayToBMP((name + ".bmp"), PrintingTestImages[index], PrintingTestImages[index], PrintingTestImages[index]);
           std::cout << std::endl << "Image is a " << rawTestLabels[index] << std::endl << "Image Saved to " << name << ".bmp";

       }
       
       //eval (alias: TrainEval)
       if (command == "eval" || command == "TrainEval")
       {
            int index;
            std::cout << "Evaluating the value of a training set image to file. To evaluate a test set image, use 't_eval' OR 'TestEval'";
            std::cout << std::endl << "What index in the training images do you want to evaluate (from 0 to 59999)" << std::endl << "Input > ";
            std::cin >> index;
            std::cout << "Predicted Value: " << max(transpose(h(h(d_mat{FlatTrainImages[0]}, w1, b1, act1), w2, b2, act2)));

       }

       //t_eval (alias: TestEval)
       if (command == "t_eval" || command == "TestEval")
       {
            int index;
            std::cout << "Evaluating the value of a test set image to file. To evaluate a training set image, use 'eval' OR 'TrainEval'";
            std::cout << std::endl << "What index in the test images do you want to evaluate (from 0 to 9999)" << std::endl << "Input > ";
            std::cin >> index;
            std::cout << "Predicted Value: " << max(transpose(h(h(d_mat{FlatTestImages[0]}, w1, b1, act1), w2, b2, act2)));

       }

       //train (alias: NONE)
       if(command == "train"){

           std::cout << "How many epochs would you like to train the program for? \nInput > ";
           std::cin >> epochs;
           std::cout << "How many batches per epoch would you like? \nInput > ";
           std::cin >> batches;
           std::cout << "How many training samples would you like? (The original program uses 2000; using too many could crash the program. Unless you have a supercomputer, don't input 60000!)\nInput > ";
           std::cin >> batchSize;

            for (int j = 0; j < epochs; j++)
            {
                
                std::cout << std::endl << "EPOCH " << j + 1 << std::endl;
                std::cout << "doing backprop..." << std::endl;

                for (int i = 0; i < batches; i++) {
                    begin = (rand() % (FlatTrainImages.size() - batchSize));
                    auto X = d_mat(FlatTrainImages.begin() + begin, FlatTrainImages.begin() + begin + batchSize); //Choose batchSize random elements from X (drawback: elements are consecutive; use some form of shuffling for future projects)
                    auto y = d_mat(Labels.begin()  + begin, Labels.begin() + begin + batchSize);
                    auto r_vals = GradientDescent(batchSize, alpha, X, y, w1, b1, w2, b2, act1, act2, dw1, db1, dw2, db2, dact1 , dact2, loss_f, Dloss_f);
                    w1 = std::get<0>(r_vals);
                    w2 = std::get<1>(r_vals);
                    b1 = std::get<2>(r_vals);
                    b2 = std::get<3>(r_vals);
                    loss = std::get<4>(r_vals);

                }

                std::cout << "Loss: ";
                printList(transpose(loss));

                double ac = 0;
                //evaluate accuracy
                std::cout << "Accuracy (on 10,000 test samples): ";
                int acc_samples = 10000;
                for (int k = 0; k < acc_samples; k++)
                {

                    auto p1 = max(transpose(h(h(d_mat{FlatTestImages[k]}, w1, b1, act1), w2, b2, act2)));
                    if(p1 == ((int) rawTestLabels[k])){ac++;}
                    
                }
                ac /= acc_samples;
                std::cout << ac;

            }

            }


   }

    return 0;

}

