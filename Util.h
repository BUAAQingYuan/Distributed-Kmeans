#ifndef Kmeans_util_h
#define Kmeans_util_h

#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <string>
#include <fstream>
#include <iostream>

using Eigen::MatrixXf;
using Eigen::VectorXf;

void LoadData(Eigen::MatrixXf& matrix,std::string train_data,const int number,const int dims);

void InitRandom(Eigen::MatrixXf& cents,Eigen::MatrixXf matrix,const int n,const int k,int seed);

void InitMCMC(Eigen::MatrixXf& cents,Eigen::MatrixXf matrix,const int k);

void computeDistance(Eigen::MatrixXf& distance,Eigen::MatrixXf& matrix,Eigen::MatrixXf& cents);

double computeSSE(Eigen::MatrixXf& distance);

void updateCluster(Eigen::MatrixXf& matrix,Eigen::MatrixXf& distance,Eigen::MatrixXf& cents);

void printResult(bool flag,Eigen::MatrixXf& cents);



#endif
