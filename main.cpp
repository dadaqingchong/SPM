//  Copyright (c) 2016 Ruqing Zhang. All Rights Reserved.
//  For more information, bug reports, fixes, contact:
//  Ruqing Zhang (daqingchongzrq@gmail.com && zhangruqing@software.ict.ac.cn) 

#include "sphePV.h"

void help()
{

	cout << "Spheerical Paragraoh Vector toolkit" << endl << endl;
	cout << "Options:" << endl;
	cout << "Parameters for training:" << endl;
	cout << "\t-train <file>" << endl;
	cout << "\t\tUse text data from <file> to train the model" << endl;
	cout << "\t-doc_output <file>" << endl;
	cout << "\t\tUse <file> to save the resulting doc vectors" << endl;
	cout << "\t-dim <int>"<< endl;
	cout << "\t\tSet size of vectors; default is 400"<< endl;
	cout << "\t-iter <int>" << endl;
	cout << "\t\tRun more training iterations; default is 100;" << endl;
	cout << "\t-min-count <int>" << endl;
	cout << "\t\tThis will discard words that appear less than <int> times; default is 2" << endl;
	cout << "\t-w_file <file>" << endl;
	cout << "\t\tread the vocabulary file from <file>" << endl;
	cout << "\t-v_file <file>" << endl;
	cout << "\t\tread the word vectors file from <file>" << endl;
	cout << "\t-init_kappa0 <float>" << endl;
	cout << "\t\tthe initial kappa_0" << endl;
}

int ArgPos(char *str, int argc, char **argv)
{
	for (int i = 1; i < argc; ++i)
		if (!strcmp(str, argv[i])) {
			if (i == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return i;
		}
		return -1;
}

int main(int argc, char* argv[])
{
	Eigen::initParallel();

	int i = 0;
	if (argc == 1)
	{
		help();
		return 0;
	}

	string input_file = "";
	string d_output_file = "";
	string save_vocab_file = "";
	string w_file = "";
	string v_file = "";
	string mu0_output_file = "";
	int dim = 50;
	int iter = 15;
	int min_count = 2;
	float init_kappa0 = 1500;
	float dis_left = -0.5;
	float dis_right = 0.5;
	float kappa_left = 1000;
	float kappa_right = 1500;

	if ((i = ArgPos((char *)"-dim", argc, argv)) > 0)
		dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-init_kappa0", argc, argv)) > 0)
		init_kappa0 = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0)
		input_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-w_file", argc, argv)) > 0)
		w_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-v_file", argc, argv)) > 0)
		v_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-doc_output", argc, argv)) > 0)
		d_output_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-mu0_output", argc, argv)) > 0)
        mu0_output_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0)
		iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0)
		min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-dis_left", argc, argv)) > 0)
        dis_left = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-dis_right", argc, argv)) > 0)
        dis_right = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-kappa_left", argc, argv)) > 0)
        kappa_left = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-kappa_right", argc, argv)) > 0)
		kappa_right = atof(argv[i + 1]);

	sphePV p2v(iter, min_count, dim, init_kappa0, dis_left, dis_right, kappa_left, kappa_right);

	p2v.train(input_file, w_file, v_file);

	if(d_output_file != "")
		p2v.save_doc2vec(d_output_file, p2v.D);
	if(mu0_output_file != "")
		p2v.save_mu0vec(mu0_output_file,p2v.mu_0);
}
