#include "Util.h"
#include "Twister_random.h"
#include <set>

#define MAX_K 100

void LoadData(Eigen::MatrixXf& matrix,std::string train_data,const int number,const int dims)
{
	matrix = Eigen::MatrixXf(number,dims);
	std::ifstream ontf;
	ontf.open(train_data.c_str(),std::ifstream::in);
	std::string line;
	
	size_t firComma = 0;
	size_t secComma = 0;

	int col = 0;
	int row = 0;
	
	while(!ontf.eof())
	{
		getline(ontf,line);
		firComma = line.find(',',0);
		if(row==number)
		{
			break;
		}
		matrix(row,col++) = atof(line.substr(0,firComma).c_str());
		while(firComma < line.size()&& col<=dims-1)
		{
			secComma = line.find(',',firComma+1);
			matrix(row,col++) = atof(line.substr(firComma+1,secComma - firComma-1).c_str());
			firComma = secComma;
		}
		if(col == dims)
		{
			col = 0;
			++row;
		}

		
	}
	ontf.close();
}

// [low,high)
int randint(int low,int high)
{
	double r = (high-low)*genrand_double();
	int rint = (int)r;
	return rint + low;
}

void select_without_replacement(std::set<int> &selected,int n,int k)
{
	while(selected.size()<k)
	{
		int temp = randint(0,n);
		selected.insert(temp);
	}
}

// select without replacement
void InitRandom(Eigen::MatrixXf& cents,Eigen::MatrixXf matrix,const int n, const int k,int seed)
{
	init_genrand(seed);
	std::set<int> selected;
	select_without_replacement(selected,n,k);
	std::set<int>::iterator ite1 = selected.begin();
	std::set<int>::iterator ite2 = selected.end();
	int count =0;
	for(;ite1!=ite2;ite1++)
	{
		cents.row(count++) = matrix.row(*ite1); 	
	}
}

void computeDistance(Eigen::MatrixXf& distance,Eigen::MatrixXf& matrix,Eigen::MatrixXf& cents)
{
	distance = -2*(matrix*cents.transpose());
	distance.rowwise() += cents.rowwise().squaredNorm().transpose().row(0);
	distance.colwise() += matrix.rowwise().squaredNorm().col(0);	
}

double computeSSE(Eigen::MatrixXf& distance)
{
	return distance.rowwise().minCoeff().sum();	
}

void updateCluster(Eigen::MatrixXf& matrix,Eigen::MatrixXf& distance,Eigen::MatrixXf& cents)
{
	//clear
	cents.setZero(cents.rows(),cents.cols());
	int counts[MAX_K] = {0};
	MatrixXf::Index minCols[distance.rows()];
	for(int i=0;i<distance.rows();i++)
	{
		distance.row(i).minCoeff(&minCols[i]);
		cents.row(minCols[i]) += matrix.row(i);
		counts[minCols[i]]++;	
	}
	
	for(int i=0;i<cents.rows();i++)
	{
		cents.row(i) = cents.row(i)*1.0/counts[i];
	}
}


void printResult(bool flag,Eigen::MatrixXf& cents)
{
	if(flag)
	{
		std::cout << "The K cluster center are:" << std::endl;
		for(int i=0;i<cents.rows();i++)
		{
			std::cout << cents.row(i) << std::endl;	
		}
	}else{
		std::ofstream file("centers.txt");
		if(file.is_open())
		{
			file << cents << std::endl;
			file.close();
		}
	}			
}


