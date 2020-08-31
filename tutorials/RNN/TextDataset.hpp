#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string.h>

#include "../../WICWIU_src/Tensor.hpp"

using namespace std;

enum OPTION
{
    ONEHOT,
    CBOW
};

void MakeOneHotVector(int* onehotvector, int vocab_size, int index)
{

    for (int i = 0; i < vocab_size; i++)
    {
        if (i == index)
            onehotvector[i] = 1;
        else
            onehotvector[i] = 0;
    }
}

template <typename DTYPE>
class TextDataset
{
private:
    char* vocab;
    char* TextData;

    int vocab_size;
    int text_length;

    Tensor<DTYPE>* input;
    Tensor<DTYPE>* label;

    OPTION option;

    int VOCAB_LENGTH;

public:
    TextDataset(string File_Path, int vocab_length, OPTION pOption)
    {
        vocab = NULL;
        TextData = NULL;

        vocab_size = 0;
        text_length = 0;

        input = NULL;
        label = NULL;

        option = pOption;

        VOCAB_LENGTH = vocab_length;

        Alloc(File_Path);
    }

    virtual ~TextDataset() { Delete(); }

    void Alloc(string File_Path);

    void Delete();

    void FileReader(string pFile_Path);
    void MakeVocab();

    void MakeInputData();
    void MakeLabelData();

    int char2index(char c);

    char index2char(int index);

    Tensor<DTYPE>* GetInputData();

    Tensor<DTYPE>* GetLabelData();

    int GetTextLength();

    int GetVocabLength();
};

template <typename DTYPE>
void TextDataset<DTYPE>::Alloc(string File_Path)
{

    vocab = new char[VOCAB_LENGTH];
    // File_Reader
    FileReader(File_Path);

    // make_vocab
    MakeVocab();

    // make_Input_data
    MakeInputData();

    // make_label_data
    MakeLabelData();
}

template <typename DTYPE>
void TextDataset<DTYPE>::Delete()
{
    delete[] vocab;
    delete[] TextData;
}

template <typename DTYPE>
void TextDataset<DTYPE>::FileReader(string pFile_Path)
{
    ifstream fin;
    fin.open(pFile_Path);

    if (fin.is_open())
    {

        fin.seekg(0, ios::end);
        text_length = fin.tellg();
        fin.seekg(0, ios::beg);

        TextData = new char[text_length];

        fin.read(TextData, text_length);

        for (int i = 0; i < text_length; i++)
            TextData[i] = tolower(TextData[i]);
    }
    fin.close();
}

template <typename DTYPE>
void TextDataset<DTYPE>::MakeVocab()
{

    int flag = 0;
    for (int i = 0; i < text_length; i++)
    {

        flag = 0;
        vocab_size = (int)strlen(vocab);

        for (int j = 0; j < vocab_size; j++)
        {
            if (vocab[j] == TextData[i])
                flag = 1;
        }

        if (flag == 0)
        {
            vocab[vocab_size] = TextData[i];
        }
    }

    vocab_size = (int)strlen(vocab) + 1;
    sort(vocab, vocab + vocab_size - 1);
}

template <typename DTYPE>
void TextDataset<DTYPE>::MakeInputData()
{

    if (option == ONEHOT)
    {
        int* onehotvector = new int[vocab_size];

        input = new Tensor<DTYPE>(text_length, 1, 1, 1, vocab_size);

        for (int i = 0; i < text_length; i++)
        {
            MakeOneHotVector(onehotvector, vocab_size, char2index(TextData[i]));
            for (int j = 0; j < vocab_size; j++)
            {
                (*input)[Index5D(input->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
            }
        }
    }
}

template <typename DTYPE>
void TextDataset<DTYPE>::MakeLabelData()
{

    if (option == ONEHOT)
    {
        int* onehotvector = new int[vocab_size];

        label = new Tensor<float>(text_length, 1, 1, 1, vocab_size);

        for (int i = 0; i < text_length; i++)
        {

            if (i == text_length - 1)
            {
                MakeOneHotVector(onehotvector, vocab_size, vocab_size - 1);
                for (int j = 0; j < vocab_size; j++)
                {
                    (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
                }
                continue;
            }

            MakeOneHotVector(onehotvector, vocab_size, char2index(TextData[i + 1]));
            for (int j = 0; j < vocab_size; j++)
            {
                (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
            }
        }
    }
}

template <typename DTYPE>
int TextDataset<DTYPE>::char2index(char c)
{

    for (int index = 0; index < vocab_size; index++)
    {
        if (vocab[index] == c)
            return index;
    }
    return -1;
}

template <typename DTYPE>
char TextDataset<DTYPE>::index2char(int index)
{

    return vocab[index];
}

template <typename DTYPE>
Tensor<DTYPE>* TextDataset<DTYPE>::GetInputData()
{

    return input;
}

template <typename DTYPE>
Tensor<DTYPE>* TextDataset<DTYPE>::GetLabelData()
{
    return label;
}

template <typename DTYPE>
int TextDataset<DTYPE>::GetTextLength()
{
    return text_length;
}

template <typename DTYPE>
int TextDataset<DTYPE>::GetVocabLength()
{
    return vocab_size;
}
