// Copyright (c) 2016 Ruqing Zhang. All Rights Reserved.
//
// For more information, bug reports, fixes, contact:
// Ruqing Zhang (daqingchongzrq@gmail.com && zhangruqing@software.ict.ac.cn)


#include <vector>
#include <string>
#include <cstdint>
#include <memory>

using namespace std;

class Word
{
public:
	size_t index;
	size_t count;
	size_t index_prevocab;
	float sample_probability;
	string text;
	Word *left, *right;

	std::vector<size_t> codes;
	std::vector<size_t> points;

public:
	Word(void){};
	Word(size_t index, size_t count, string text, size_t index_prevocab, Word *left = nullptr, Word *right = nullptr):
	    index(index), count(count), text(text), index_prevocab(index_prevocab), left(left), right(right) {}

	~Word(void){};
};

typedef std::shared_ptr<Word> WordP;

