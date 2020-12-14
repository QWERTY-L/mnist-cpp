/*
 * Created December 2020
 * Author: Luke Sequeira
 *
 * Copyright (c) 2020 Luke Sequeira
 */

#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <chrono>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#ifndef NNFUNCTIONS_HPP
#define NNFUNCTIONS_HPP

const double ln_2 = log(2);

template<typename T>
void printList(std::vector<std::vector<T>> lst) {
	std::cout << "\n";
	for (int i = 0; i < lst.size(); i++) {
		std::cout << "\n";
		for (int j = 0; j < lst[i].size(); j++) {
			std::cout << lst[i][j] << ", ";
		}
	}
	std::cout << "\n";
}

template<typename T>
void shape(std::vector<std::vector<T>> lst){

    std::cout << std::endl << "Shape: (" << lst.size() << ", " << lst[0].size() << ")" << std::endl;

}

double relu(double x) {

	if (x > 0) {

		return x;

	}
	else {

		return 0;

	}

}

std::vector<double> relu(std::vector<double> x) {
	std::vector<double> c(x.size());
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = relu(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> relu(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

			c[i] = relu(x[i]);

	}
	return c;

}

/*double crossEntropy(double yp, double y) {

	if(y == 1){

		return -log(yp);

	} else{

		return -log(1-yp);

	}

}

std::vector<double> crossEntropy(std::vector<double> yp, std::vector<double> y) {
	std::vector<double> c(y.size());
	#pragma omp parallel for
	for (int i = 0; i < y.size(); i++) {

		c[i] = crossEntropy(yp[i], y[i]);

	}
	return c;

}

std::vector<std::vector<double>> crossEntropy(std::vector<std::vector<double>> yp, std::vector<std::vector<double>> y) {
	std::vector<std::vector<double>> c(y.size(), std::vector<double>(1));
	#pragma omp parallel for
	for (int i = 0; i < y.size(); i++) {

			c[i] = crossEntropy(yp[i], y[i]);

	}
	return c;

}*/

double drelu(double x) {

	if (x > 0) {

		return 1;

	}
	else {

		return 0;

	}

}

std::vector<double> drelu(std::vector<double> x) {
	std::vector<double> c(x.size());
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = drelu(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> drelu(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = drelu(x[i]);

	}
	return c;

}

double sig(double x) {

	return 1.0 / (1 + exp(-x));

}

std::vector<double> sig(std::vector<double> x) {
	std::vector<double> c(x.size());
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = sig(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> sig(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = sig(x[i]);

	}
	return c;

}

double dsig(double a) { //takes input sig(a) and returns dsig(a)

	//double a = sig(x);
	return (a * (1 - a));

}

std::vector<double> dsig(std::vector<double> x) {
	std::vector<double> c(x.size());
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = dsig(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> dsig(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = dsig(x[i]);

	}
	return c;

}

d_vec softmax(d_vec x){

    //Calculate e^X and sum
    float div = 0; //sum
    #pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = exp(x[i]);
        div += x[i]; //Calculate SUM of exps
    }
    #pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = x[i] / div;
    }

    return x;

}

d_mat softmax(d_mat x){

    d_mat out_c(x.size(), d_vec(x[0].size()));
    #pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        out_c[i] = softmax(x[i]);
    }

    return out_c;

}

double dsoftmax(double a) { //takes input softmax(a) and returns dsoftmax(a)

	return (a * (1 - a));

}

std::vector<double> dsoftmax(std::vector<double> x) {
	std::vector<double> c(x.size());
    #pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = dsoftmax(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> dsoftmax(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
    #pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = dsoftmax(x[i]);

	}
	return c;

}

/*double tanh(double x) {

	return 1.0 / (1 + exp(-x));

}*/ //this is a builtin function of math.h

std::vector<double> tanh(std::vector<double> x) {
	std::vector<double> c(x.size());
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = tanh(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> tanh(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = tanh(x[i]);

	}
	return c;

}

double dtanh(double a) { //takes input tanh(a) and returns dtanh(a)

	//double a = sig(x);
	return (1 - (a*a));

}

std::vector<double> dtanh(std::vector<double> x) {
	std::vector<double> c(x.size());
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = dtanh(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> dtanh(std::vector<std::vector<double>> x) {
	std::vector<std::vector<double>> c(x.size(), std::vector<double>(x[0].size()));
#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) {

		c[i] = dtanh(x[i]);

	}
	return c;

}

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> a) {
	
	std::vector<std::vector<double>> c(a[0].size(), std::vector<double>(a.size()));
	#pragma omp parallel for
	for (int i = 0; i < a[0].size(); i++) {
	#pragma omp parallel for
		for (int j = 0; j < a.size(); j++) {

			c[i][j] = a[j][i];
		}

	}
	return c;
}

std::vector<double> add(std::vector<double> a, std::vector<double> b) {

	std::vector<double> c(a.size());
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = a[i] + b[i];
		//std::cout << omp_get_thread_num() << "   ";

	}
	return c;

}

std::vector<std::vector<double>> add(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) {

	std::vector<std::vector<double>> c(a.size(), std::vector<double>(a[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

			c[i] = add(a[i], b[i]);

	}
	return c;

}

std::vector<double> sum(std::vector<double> a) {

	std::vector<double> c = { 0 };
	for (int i = 0; i < a.size(); i++) {

		c[0] += a[i];

	}
	return c;

}

std::vector<std::vector<double>> sum(std::vector<std::vector<double>> a) {

	std::vector<std::vector<double>> c(a.size(), std::vector < double>(1));
	for (int i = 0; i < a.size(); i++) {

		c[i] = sum(a[i]);

	}
	return c;

}

std::vector<double> sub(std::vector<double> a, std::vector<double> b) {

	std::vector<double> c(a.size());
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = a[i] - b[i];
		//std::cout << omp_get_thread_num() << "   ";

	}
	return c;

}

std::vector<std::vector<double>> sub(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) {

	std::vector<std::vector<double>> c(a.size(), std::vector<double>(a[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = sub(a[i], b[i]);

	}
	return c;

}

std::vector<double> emult(std::vector<double> a, std::vector<double> b) {

	std::vector<double> c(a.size());
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = a[i] * b[i];
		//std::cout << omp_get_thread_num() << "   ";

	}
	return c;

}

std::vector<std::vector<double>> emult(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) {

	std::vector<std::vector<double>> c(a.size(), std::vector<double>(a[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = emult(a[i], b[i]);

	}
	return c;

}

std::vector<double> smult(double a, std::vector<double> b) {

	std::vector<double> c(b.size());
	#pragma omp parallel for
	for (int i = 0; i < b.size(); i++) {

		c[i] = a * b[i];
		//std::cout << omp_get_thread_num() << "   ";

	}
	return c;

}

std::vector<std::vector<double>> smult(double a, std::vector<std::vector<double>> b) {

	std::vector<std::vector<double>> c(b.size(), std::vector<double>(b[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < b.size(); i++) {

		c[i] = smult(a, b[i]);

	}
	return c;

}

std::vector<std::vector<double>> dadd(std::vector<std::vector<double>> a, std::vector<double> b) {

	std::vector<std::vector<double>> c(a.size(), std::vector<double>(a[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

			c[i] = add(a[i], b);

	}
	return c;

}

void printList(std::vector<double> lst) {

	std::cout << "\n";
	for (int i = 0; i < lst.size(); i++) {

		std::cout << lst[i] << ", ";

	}
	std::cout << "\n";
}

double dot(std::vector<double> a, std::vector<double> b) {

	std::vector<double> c(a.size());
	double out = 0;
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = a[i] * b[i];
	}
	for (int i = 0; i < a.size(); i++) {

		out += c[i];

	}
	return out;

}
std::vector<double> dot(std::vector<double> a, std::vector<std::vector<double>> b) {

	std::vector<double> c(b.size());
	#pragma omp parallel for
	for (int i = 0; i < b.size(); i++) {

		c[i] = dot(a, b[i]);

	}
	return c;

}

std::vector<std::vector<double>> dot(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) {

	std::vector<std::vector<double>> c(a.size());
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++) {

		c[i] = dot(a[i], b);

	}
	return c;
}

std::vector<std::vector<double>> h(std::vector<std::vector<double>> x, std::vector<std::vector<double>> w, std::vector<double> b, std::vector<std::vector<double>>(*act)(std::vector<std::vector<double>>)) {

	return act(dadd(transpose(dot(w, x)), b));

}

double MSE(double yh, double yp) {

	double c = (yh + yp);
	return (c * c);

}

std::vector<double> MSE(std::vector<double> yh, std::vector<double> yp) {

	std::vector<double> c = sub(yh, yp);
	return emult(c, c);

}

std::vector<std::vector<double>> MSE(std::vector<std::vector<double>> yh, std::vector<std::vector<double>> yp) {

	std::vector<std::vector<double>> c = sub(yh, yp);
	return emult(c, c);

}

std::vector<std::vector<double>> dMSE(std::vector<std::vector<double>> yh, std::vector<std::vector<double>> yp) {

	return transpose(smult(2, sub(transpose(yh), yp))); //by the power rule, dx^2/dx = 2x (the 2 can technically be removed to give the effect of a twice as small learning rate. This may negligibly improve the performance)

}

std::vector<double> categorical_cross_entropy(std::vector<double> yh, std::vector<double> yp) {

    d_vec c = {0};
    #pragma omp parallel for
    for (int i = 0; i < yp.size(); i++)
    {
        if(yh[i] != 0){
            c[0] -= log2(yp[i]);
        }
    }
    
    return c;

}

d_mat categorical_cross_entropy(d_mat yh, d_mat yp){

    d_mat c(yh.size(), d_vec(1));
    #pragma omp parallel for
    for (int i = 0; i < yh.size(); i++)
    {
        c[i] = categorical_cross_entropy(yh[i], yp[i]);
    }
    return c;

}

d_mat f_dsoftmax(d_mat input){ //use this instead of dsoftmax

	d_mat a(input.size(), d_vec(input[0].size()));
	#pragma omp parallel for
	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < a[0].size(); j++)
		{
			a[i][j] = 1; //set to an array of ones; the magic happens in dcategorical_cross_entropy
		}	
	}
	return a;

}

d_mat Dcategorical_cross_entropy(d_mat yh, d_mat yp){

    #pragma omp parallel for
    for (int i = 0; i < yh.size(); i++)
    {
		#pragma omp parallel for
        for (int j = 0; j < yh[0].size(); j++)
        {
            yh[i][j] -= yp[i][j];
        }
    }
	return yh;

}

int max(d_mat vec){ //takes a d_mat vector and returns the position of the greatest element

    int max_element = 0;
    for (int i = 0; i < vec.size(); i++)
    {
        if(vec[max_element] < vec[i]){max_element = i;}
    }
    
    return max_element;

}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<double>, std::vector<double>, std::vector<std::vector<double>>> GradientDescent(int samples, double alpha, std::vector<std::vector<double>> X, std::vector<std::vector<double>> y, std::vector<std::vector<double>> w1, std::vector<double> b1, std::vector<std::vector<double>> w2, std::vector<double> b2, std::vector<std::vector<double>>(*act1)(std::vector<std::vector<double>>), std::vector<std::vector<double>>(*act2)(std::vector<std::vector<double>>), std::vector<std::vector<double>> dw1, std::vector<double> db1, std::vector<std::vector<double>> dw2, std::vector<double> db2, std::vector<std::vector<double>>(*dact1)(std::vector<std::vector<double>>), std::vector<std::vector<double>>(*dact2)(std::vector<std::vector<double>>), d_mat(*loss_f)(d_mat, d_mat), d_mat(*Dloss_f)(d_mat, d_mat)) {

	//returns w1, w2, b1, b2 so they can be stored in main()

	//evaluation
	std::vector<std::vector<double>> L1 = h(X, w1, b1, act1);
	std::vector<std::vector<double>> yp = h(L1, w2, b2, act2);

	//DERIVATIVES

	//Layer 2
	std::vector<std::vector<double>> dLoss = Dloss_f(y, yp); 
	std::vector<std::vector<double>> dACT2 = dact2(yp);
	std::vector<std::vector<double>> dACT1 = dact1(L1);
	std::vector<std::vector<double>> dz2 = emult(dLoss, dACT2);
	dw2 = dot(transpose(dz2), transpose(L1)); //technically, dw2 is the negative of this, but we omit the negative and later add rather than subtract
	db2 = dot(transpose(dLoss), transpose(dACT2))[0]; //this is also the negative of db2
	
	//Layer 1
	std::vector<std::vector<double>> dz1 = emult(dot(dz2, transpose(w2)), dACT1);
	dw1 = dot(transpose(dz1), transpose(X)); //again, these are the negatives of dw1 and db1 respectivly
	db1 = transpose(sum(transpose(dz1)))[0];

	//update weights and biases

	w1 = add(w1, (smult(alpha, smult(1.0 / samples, dw1)))); //note, remove 1 smult and change to smult(alpha/samples, vector);
	w2 = add(w2, (smult(alpha, smult(1.0 / samples, dw2))));
	b1 = add(b1, (smult(alpha, smult(1.0 / samples, db1))));
	b2 = add(b2, (smult(alpha, smult(1.0 / samples, db2))));

    //calculate loss
	std::vector<std::vector<double>> losss = smult(1.0 / samples, sum(transpose(loss_f(y, yp))));

	return std::make_tuple(w1, w2, b1, b2, losss);

}
#endif