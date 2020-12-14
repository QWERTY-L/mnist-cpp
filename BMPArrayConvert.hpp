/*
 * Created December 2020
 * Author: Luke Sequeira
 *
 * Copyright (c) 2020 Luke Sequeira
 */


#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#ifndef BMPArrayConvert_HPP
#define BMPArrayConvert_HPP

typedef unsigned char ubyte; //unsigned char
typedef unsigned int uint; //unsigned int
typedef std::vector<std::vector<uint>> i_mat; //integer matrix
typedef std::vector<uint> i_vec; //integer array

namespace BMP{


    void printList(i_mat arrs){

        for(int i = 0; i < arrs.size(); i++){

            for(int j = 0; j < (arrs[i]).size(); j++){

                std::cout << arrs[i][j] << ", ";

            }

            std::cout << std::endl;

        }
        return;

    }

    void intTOChar(ubyte* out, size_t in, int bytes){

        for (int i = 0; i < bytes; i++){
                out[i] = in;
                in = in >> 8; //move a over 8 bits
            }

    }

    bool arrayToBMP(std::string filePath, i_mat r, i_mat g, i_mat b){ //takes pixel arrays and converts it to a BMP file.

        std::ofstream writeBMP;
        writeBMP.open(filePath); //Open file

        //write file header (BITMAPFILEHEADER) -> all data is little endian except 'BM' header (14 bytes)

        writeBMP << 'B' << 'M'; //write BM header to file (indicating this is a bitmap file) (2 bytes)

        //write file size (4 bytes)
        uint32_t file_size = 54 + (3 * r.size() * r[0].size()); //there are 54 bytes in the header section and and 3 * len * width bytes in raw data section
        ubyte* cfs = new ubyte[4];
        intTOChar(cfs, file_size, 4);
        for (int i = 0; i < 4; i++)
        {
            writeBMP << cfs[i];
        }
        delete[] cfs;

        //reserved bytes dependant on the program. Initialized to zero (2 * 2 bytes = 4 bytes)
        for (int i = 0; i < 4; i++)
        {
            writeBMP << (ubyte) 0; //write 4 zeros
        }

        //offset (distance from first byte to data. Equal to 54 == 0x36 by default) (4 bytes)
        writeBMP << (ubyte) 54;
        for (int i = 0; i < 3; i++)
        {
            writeBMP << (ubyte) 0; //write 3 zeros
        }
        
        //DIB header (BITMAPINFOHEADER) - Information header (40 bytes)

        //size of Info header (4 bytes)
        writeBMP << (ubyte) 40; //info header is 40 bytes
        for (int i = 0; i < 3; i++)
        {
            writeBMP << (ubyte) 0; //write 3 zeros
        }

        //width of file in pixels (4 bytes).
        //Followed by height of file in pixels (4 bytes)
        uint32_t width = r.size();
        uint32_t height = r[0].size();
        ubyte *cwd = new ubyte[4]; //width char array
        ubyte *chd = new ubyte[4]; //height char array
        intTOChar(cwd, width, 4);
        intTOChar(chd, height, 4);
        for (int i = 0; i < 4; i++)
        {
            writeBMP << cwd[i];
        }
        for (int i = 0; i < 4; i++)
        {
            writeBMP << chd[i];
        }

        //Planes == 1 (2 bytes)
        writeBMP << (ubyte) 1;
        writeBMP << (ubyte) 0;

        //Bits per Pixel. We set it to 24 for bit RGB color (2 bytes)
        writeBMP << (ubyte) 24; //24 bit RGB color (~16M possible colors)
        writeBMP << (ubyte) 0;

        //Compression Used. Usually set to 0 (no compression) (4 bytes)
        for (int i = 0; i < 4; i++)
        {
            writeBMP << (ubyte) 0; //write 4 zeros
        }

        //Compressed Image Size (can be set to 0 if compression == 0) (4 bytes)
        for (int i = 0; i < 4; i++)
        {
            writeBMP << (ubyte) 0; //write 4 zeros
        }

        //The following; Pixels per meter X, Pixels per meter Y, Num Colors Used, and num important colors will all be initialized to 0. Each is 4 bytes (4 * 4 bytes = 16 bytes)
        for (int i = 0; i < 16; i++)
        {
            writeBMP << (ubyte) 0; //write 4 zeros
        }

        //Pixel Raw Data (This must be padded to make data a multiple of 4 bytes) (bottom to top; left to right)
        //Data is in BGR format

        //Calculate padding
        int padding = (4 - ((r[0].size() * 3) % 4)) % 4;

        //write data
        for (int i = r.size() - 1; i >= 0; i--) //bottom to top
        {
            for (int j = 0; j < r[0].size(); j++) //left to right
            {
                writeBMP << (ubyte) (b[i][j]); //write blue
                writeBMP << (ubyte) (g[i][j]); //write green
                writeBMP << (ubyte) (r[i][j]); //write red
            }
        for (int k = 0; k < padding; k++)
        {
            writeBMP << (ubyte) 0; //pad with appropriate amount of zeros
        }
        }
        
        

        return true; //successful

    }

}

#endif