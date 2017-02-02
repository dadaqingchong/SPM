//  Copyright (c) 2016 Ruqing Zhang. All Rights Reserved.
//  For more information, bug reports, fixes, contact:
//  Ruqing Zhang (daqingchongzrq@gmail.com && zhangruqing@software.ict.ac.cn) 

#include "sphePV.h"

sphePV::sphePV(int iter, int min_count, int dim, float init_kappa0, float init_kappa, float dis_left, float dis_right, float kappa_left, float kappa_right):
iter(iter), min_count(min_count), dim(dim), init_kappa0(init_kappa0), init_kappa(init_kappa), dis_left(dis_left), dis_right(dis_right), kappa_left(kappa_left), kappa_right(kappa_right), generator(rd())
{
	doc_num = 0;
	total_words = 0;
}

sphePV::~sphePV(void)
{
}

void sphePV::build_vocab(string filename)
{
	ifstream in(filename);
	string s, w;
	unordered_map<string, size_t> word_cn;

	while (std::getline(in, s))
	{
		doc_num++;
		istringstream iss(s);
		while (iss >> w)
		{
			auto it = vocab_read_hash.find(w);
			if(it == vocab_read_hash.end()) continue;

			if(word_cn.count(w) > 0)
				word_cn[w]++;
			else
				word_cn[w] = 1;
		}
	}
	in.close();
	cout << word_cn.size() << endl;
    int i = 0;

    for(auto kv: word_cn)
	{
		if(kv.second < min_count)
			continue;

		auto it = vocab_read_hash.find(kv.first);
		Word *word = it->second.get();
		
		Word *w = new Word(i, kv.second, kv.first, word->index);
		vocab.push_back(w);
		
		vocab_hash[w->text] = WordP(w);
		total_words += kv.second;
		i++;
	}
	cout << vocab_hash.size() << endl;
}


void sphePV::read_word(string v_file)
{
	ifstream in(v_file);
	string s, w;
	int i = 0;
	while (std::getline(in, s))
	{
		istringstream iss(s);
		while (iss >> w)
		{
			Word *w_read = new Word(i, 1, w, 0);
			vocab_read.push_back(w_read);
			vocab_read_hash[w_read->text] = WordP(w_read);
			i++;
		}
	}
}




void sphePV::init_weights(string w_file)
{	
	cout << dis_left << "    " << dis_right << endl;
	std::uniform_real_distribution<float> distribution(dis_left, dis_right);
	auto uniform = [&] (int) {return distribution(generator);};
	std::uniform_real_distribution<float> distribution_(kappa_left, kappa_right);
	auto uniform_ = [&] (int) {return distribution_(generator);};

	//mu_n'
	D = RMatrixXf::NullaryExpr(doc_num, dim, uniform);
		
	//the observed variable w_n, fixed, word2vec preprocessing, ||w|| = 1
	W = RMatrixXf::NullaryExpr(vocab.size(), dim, uniform) / (float)dim;
	//read the word vector from <file>

	W_pre = RMatrixXf::NullaryExpr(vocab_read.size(), dim, uniform) / (float)dim;
	ifstream in(w_file);
	string s, w;
	int i = 0;
	while (std::getline(in, s))
	{
		istringstream iss(s);
		int j = 0;
		while (iss >> w)
		{
			W_pre(i,j) = atof(w.c_str());
			j++;
		}
		i++;
	}
	in.close();
    for(auto& v: vocab)    
    {  
    	int position = v->index_prevocab;
        W.row(v->index) = W_pre.row(position) / W_pre.row(position).norm();
    }

	//the mean direction for the document generation mu_0, ||mu_0|| = 1
	mu_0 = RMatrixXf::NullaryExpr(1, dim, uniform);
	mu_0 = mu_0 / mu_0.norm();

	//the concentration parameter for the document genetation kappa_0
	kappa_0 = init_kappa0;
	//the concentration parameter for the words generation kappa_n
	kappa = RMatrixXf::NullaryExpr(doc_num, 1, uniform_);
	//kappa'
	kappa_ = RMatrixXf::NullaryExpr(doc_num, 1, uniform_);

}


vector<vector<Word *>> sphePV::build_docs(string filename)
{
	ifstream in(filename);
	string s, w;
	vector<vector<Word *>> docs;

	while (std::getline(in, s))
	{
		vector<Word *> doc;
		istringstream iss(s);

		while (iss >> w)
		{
			auto it = vocab_hash.find(w);
			if (it == vocab_hash.end()) continue;
			Word *word = it->second.get();
			doc.push_back(word);
		}
		docs.push_back(std::move(doc));
	}
	in.close();

	return std::move(docs);
}


void sphePV::variaest(vector<vector<Word *>>& docs)
{
	for(int k = 0; k < iter; k++)
	{	
		cout << "iter: " << k << endl;
		
        //E
		for(int i = 0; i < doc_num; ++i)
		{
			vector<Word *>& doc = docs[i];
			float kappa_current = kappa(i,0);
			int doc_len = doc.size();
			RowVectorXf word_sum = RowVectorXf::Zero(dim);
			for(int j = 0; j < doc_len; ++j)
			{
				Word* current_word = doc[j];
				word_sum += kappa_current * W.row(current_word->index);
			}
			RowVectorXf c = mu_0 * kappa_0 + word_sum;
			kappa_(i,0) = c.norm() / 2;
			D.row(i) = c / c.norm();
		}

		RMatrixXf E_d = RMatrixXf::Zero(doc_num, dim);
        for(int i = 0; i < doc_num; ++i)
        {
            float a =  sqrt(dim * dim + 4 * kappa_(i,0) * kappa_(i,0)) - dim ;
            float b = 2 * kappa_(i,0);
            float A = a/b;
            E_d.row(i) = D.row(i) * A;
        }

        //M
		RowVectorXf c = RMatrixXf::Zero(1, dim);
		for(int i = 0; i < doc_num; ++i)
		{
			c += E_d.row(i);
		}
		mu_0 = c / c.norm();

		float r1 = c.norm() / doc_num;
		kappa_0 = ( r1 * dim - r1 * r1 * r1 ) / ( 1 - r1 * r1);
		
		for(int m = 0; m < doc_num; ++m)
		{
			vector<Word *>& doc = docs[m];
			float word_sum = 0;
			int doc_len = doc.size();
			for(int n = 0; n < doc_len; ++n)
			{
				Word* current_word = doc[n];
				word_sum += E_d.row(m) * W.row(current_word->index).transpose();		
			}
			float r2 = word_sum / doc_len;
			kappa(m,0) = ( r2 * dim -r2 * r2 * r2 ) / (1 - r2 * r2);
		}
	}
}




void sphePV::train(string filename, string w_file, string v_file)
{	
	read_word(v_file);
	build_vocab(filename);
	init_weights(w_file);
	vector<vector<Word *>> docs = build_docs(filename);
	variaest(docs);
}
 


void sphePV::save_doc2vec(string filename, const RMatrixXf& data)
{
	IOFormat CommaInitFmt(StreamPrecision, DontAlignCols);
	ofstream out(filename, std::ofstream::out);
	out << data.format(CommaInitFmt) << endl;
	out.close();
}


void sphePV::save_mu0vec(string filename, const RMatrixXf& data)
{
	IOFormat CommaInitFmt(StreamPrecision, DontAlignCols);
    ofstream out(filename, std::ofstream::out);
    out << data.format(CommaInitFmt) << endl;
    out.close();

}
