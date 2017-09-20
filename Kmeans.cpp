#include "Util.h"
#include <cmath>

using namespace std;

void train(string train_data,const int train_size,const int dims,const int k,const int iter)
{
	Eigen::MatrixXf matrix(train_size,dims);
	cout << "loading data ..." << endl;
	LoadData(matrix,train_data,train_size,dims);
	// init centers
	Eigen::MatrixXf cents(k,dims);
	cout << "init cluster centers ..." << endl;
	InitRandom(cents,matrix,train_size,k,100);
	//iterator
	cout << "running Iteration ..." << endl;
	
	double epolon = 1e-3;
	Eigen::MatrixXf distance(train_size,k);
	double prev_sse = 0;
	for(int i=1;i<=iter;i++)
	{
		computeDistance(distance,matrix,cents);
		double sse = computeSSE(distance);
		cout << "iter " << i << "," << "SSE=" << sse << endl;
		if(abs(prev_sse - sse)<epolon)
			break;
		updateCluster(matrix,distance,cents);						
	}

	cout << "Iteration end." << endl;
	printResult(true,cents);
}

int main()
{
	string path = "/home/qqqing/EigenTest/mcmc_kmeans/points.txt";
	train(path,10,2,2,5);
}

