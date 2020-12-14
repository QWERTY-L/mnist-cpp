/*
 * Created December 2020
 * Author: Luke Sequeira
 * 
 * Copyright (c) 2020 Luke Sequeira
 * 
 * Data Source: http://yann.lecun.com/exdb/mnist/
 * 
 */


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <tuple>

typedef unsigned char ubyte;
typedef std::vector<double> d_vec;
typedef std::vector<d_vec> d_mat;
typedef unsigned int uint; //unsigned int
typedef std::vector<std::vector<uint>> i_mat; //integer matrix
typedef std::vector<uint> i_vec; //integer array

#ifndef MNISTPARSER_HPP
#define MNISTPARSER_HPP
namespace MnistParse{

    void printList(d_mat lst) {
        std::cout << "\n";
        for (int i = 0; i < lst.size(); i++) {
            std::cout << "\n";
            for (int j = 0; j < lst[i].size(); j++) {
                std::cout << lst[i][j] << ", ";
            }
        }
        std::cout << "\n";
    }

    void printList(i_mat lst) {
        std::cout << "\n";
        for (int i = 0; i < lst.size(); i++) {
            std::cout << "\n";
            for (int j = 0; j < lst[i].size(); j++) {
                std::cout << lst[i][j] << ", ";
            }
        }
        std::cout << "\n";
    }

    std::tuple<std::vector<i_mat>, d_mat> ParseMnistImages(std::string path){ //returns tuple containing 3d array of 2d images and 2d array of flattened images

        std::ifstream get_data;
        get_data.open(path, std::ios::binary | std::ios::ate); //open file
        int file_size = get_data.tellg(); //get file size
        std::cout << "File Size: " << file_size << std::endl;

        if(file_size == -1){ //file doesn't exist

            std::cout << "File not found" << std::endl;
            throw "File not found";

        }

        ubyte *data = new ubyte[file_size]; //define unsigned char array for data
        char *td = new char[file_size]; //cast this char array to ubyte later

        //read data
        get_data.seekg(0, std::ios::beg);
        get_data.read(td, file_size); //read data to char array
        //data = (ubyte*) td; //cast signed char data to unsigned char data  DON'T DO THIS, IT CAUSES PROGRAM TO CRASH WHEN MEMORY IS FREED
        for (int i = 0; i < file_size; i++)
        {
            data[i] = (ubyte) td[i];
        }
        

        //Check if file type is correct

        /*
        * 
        * File Header Format:
        * 
        * Magic Number - 32 bit int 0x803
        * Number of Images - 32 bit int (ex. 60000 or 10000)
        * Number of Rows - 32 bit int 28
        * Number of Columns - 32 bit int 28
        * 
        */

        if(data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x08 && data[3] == 0x03){ //File header for images in hex is 0x00000803 (0x803 is 2051 in decimal)

            std::cout << "Image filetype correct";

        } else if(data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x08 && data[3] == 0x01){ //File header for labels in hex is 0x00000801 (0x801 is 2049 in decimal)

            std::cout << "You inputed the label file instead of the images file. Change the file extention to the images file (ex. 'train-images.idx3-ubyte')";
            throw "Error: You inputed the label file instead of the images file. Change the file extention to the images file (ex. 'train-images.idx3-ubyte')";

        } else{

            std::cout << "The file you inserted had an incorrect format. Did you forget to unzip the files?";
            throw "Error: The file you inserted had an incorrect format. Did you forget to unzip the files?";

        }

        //Calculate number of images
        int num_images = (data[7]) + (data[6] * 0x100) + (data[5] * 0x10000) +(data[4] * 0x1000000);
        std::cout << std::endl << num_images << " images have been detected" << std::endl;

        //read images

        int data_start = 16; //pixel data starts on bit 17 (set to 16 because array index starts at 0)
        std::vector<i_mat> displayIms(num_images, i_mat(28, i_vec(28))); //Images to be printed (not inputed into NN)
        d_mat flattenedImages(num_images, d_vec(28*28)); //vector of flat images 
        for (int k = 0; k < num_images; k++)
        {
                for(int i = 0; i<28; i++){

            for(int j = 0; j<28; j++){

                displayIms[k][i][j] = data[data_start + (i * 28) + j + (k*784)];

            }

        }
        }

        for (int k = 0; k < num_images; k++)
        {
            for (int j = 0; j < 784; j++)
            {
                flattenedImages[k][j] = data[data_start + j + (k*784)];
            }
            
        }
        

        get_data.close(); //close file

        //delete data to prevent memory leaks
        delete[] data;
        delete[] td;
        data = nullptr;
        td = nullptr;

        return std::make_tuple(displayIms, flattenedImages); //return array for printing followed by flattened double array

    }


    i_vec ParseMnistLabels(std::string path){ //returns vector of unsigned ints of labels

        std::ifstream get_data;
        get_data.open(path, std::ios::binary | std::ios::ate); //open file
        int file_size = get_data.tellg(); //get file size
        std::cout << "File Size: " << file_size << std::endl;
        if(file_size == -1){ //file doesn't exist

            std::cout << "File not found" << std::endl;
            throw "File not found";

        }

        ubyte *data = new ubyte[file_size]; //define unsigned char array for data
        char *td = new char[file_size]; //cast this char array to ubyte later

        //read data
        get_data.seekg(0, std::ios::beg);
        get_data.read(td, file_size); //read data to char array
        //data = (ubyte*) td; //cast signed char data to unsigned char data

        for (int i = 0; i < file_size; i++)
        {
            data[i] = (ubyte) td[i];
        }

        //Check if file type is correct

        /*
        * 
        * File Header Format:
        * 
        * Magic Number - 32 bit int 0x801
        * Number of Images - 32 bit int (ex. 60000 or 10000)
        * 
        */

        if(data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x08 && data[3] == 0x01){ //File header for labels in hex is 0x00000801 (0x801 is 2049 in decimal)

            std::cout << "Label filetype correct";

        } else if(data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x08 && data[3] == 0x03){ //File header for images in hex is 0x00000803 (0x803 is 2051 in decimal)

            std::cout << "You inputed the image file instead of the labels file. Change the file extention to the images file (ex. 'train-labels.idx1-ubyte')";
            throw "You inputed the image file instead of the labels file. Change the file extention to the images file (ex. 'train-labels.idx1-ubyte')";

        } else{

            std::cout << "The file you inserted had an incorrect format. Did you forget to unzip the files?";
            throw "Error: The file you inserted had an incorrect format. Did you forget to unzip the files?";

        }

        //Calculate number of images
        int num_labels = (data[7]) + (data[6] * 0x100) + (data[5] * 0x10000) +(data[4] * 0x1000000);
        std::cout << std::endl << num_labels << " labels have been detected" << std::endl;

        //read images

        int data_start = 8; //pixel data starts on bit 9 (set to 8 because array index starts at 0)
        i_vec Labels(num_labels); //Images to be printed (not inputed into NN)

        for (int k = 0; k < num_labels; k++)
        {

                Labels[k] = data[data_start + k];
            
        }

        get_data.close(); //close file

        //delete data to prevent memory leaks
        delete[] data;
        delete[] td;
        data = nullptr;
        td = nullptr;

        return Labels;

    }

}
#endif // !MNISTPARSER_HPP