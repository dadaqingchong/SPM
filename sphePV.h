// Copyright (c) 2016 Ruqing Zhang. All Rights Reserved.
// For more information, bug reports, fixes, contact:
// Ruqing Zhang (daqingchongzrq@gmail.com && zhangruqing@software.ict.ac.cn) 

#pragma once
#include <vector>
#include <list>
#include <string>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <cstdint>
#include <cmath>
#include <limits>
#include <Eigen/SparseCore>
#include <Eigen/Dense>

#include "Word.h"

using namespace std;
using namespace Eigen;

#define EIGEN_NO_DEBUG
typedef Matrix<float, Dynamic, Dynamic, RowMajor> RMatrixXf;


class sphePV
{
public:
	int iter;
	int min_count;
	int dim;
	float min_alpha;
	long doc_num;
	long long total_words;

	vector<Word *> vocab;
	vector<Word *> vocab_read;
	unordered_map<string, WordP> vocab_hash;
	unordered_map<string, WordP> vocab_read_hash;
	float kappa_0;
	float init_kappa0;
	float init_kappa;
	RMatrixXf W, W_pre, D, mu_0, kappa, kappa_, X;

	float dis_left;
	float dis_right;
	float kappa_left;
	float kappa_right;
	std::random_device rd;
	std::mt19937 generator;

public:
	sphePV(int iter=100, int min_count=2, int dim=400, float init_kappa0 = 0.5, float init_kappa = 0.5, float dis_left=1.5, float dis_right=2.5, float kappa_left=10, float kappa_right=15);
	~sphePV(void);

	void read_word(string v_file);
	void build_vocab(string filename);
	void init_weights(string w_file);
	vector<vector<Word *>> build_docs(string filename);
	
	void variaest(vector<vector<Word *>>& docs);
	void train(string filename, string w_file, string v_file);
	void save_doc2vec(string filename, const RMatrixXf& data);
	void save_mu0vec(string filename, const RMatrixXf& data);
};

