#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string.h>

#include "../../WICWIU_src/Tensor.hpp"
#include "../../WICWIU_src/DataLoader.hpp"

#define DATATIME        20

using namespace std;

enum OPTION {
    ONEHOT,
    //CBOW
};


void MakeOneHotVector(int* onehotvector, int vocab_size, int index){

    for(int i=0; i<vocab_size; i++){
        if(i==index)
            onehotvector[i] = 1;
        else
            onehotvector[i] = 0;
    }
}

//: public Dataset<DTYPE>{

template<typename DTYPE>
class TextDataset : public Dataset<DTYPE> {
private:

    char* vocab ;
    char* TextData;

    int vocab_size;
    int text_length;

    Tensor<DTYPE>* m_input;
    Tensor<DTYPE>* m_label;

    OPTION option;

    int VOCAB_LENGTH;

    //dataloader랑 같이 하기
    DTYPE **m_aaInput;
    DTYPE **m_aaLabel;

    int m_numOfInput;             //input data의 개수!!!

    int m_dimOfInput;         //이거는 필요없지 않을까...
    int m_dimOfLabel;         //이것도 필요 없을듯?

public:
    TextDataset(string File_Path, int vocab_length, OPTION pOption) {
        vocab = NULL;
        TextData = NULL;

        vocab_size = 0;
        text_length = 0;

        m_input = NULL;
        m_label = NULL;

        option = pOption;

        VOCAB_LENGTH = vocab_length;


        //Dataloader랑 같이 하기
        m_aaInput = NULL;
        m_aaLabel = NULL;

        m_numOfInput = 0;

        m_dimOfInput = 0;
        m_dimOfLabel = 0;


        Alloc(File_Path);
    }

    virtual ~TextDataset() {
        Delete();
    }

    //왜 굳이 virtual인거지?
    void                                  Alloc(string File_Path);

    void                                  Delete();

    void                                  FileReader(string pFile_Path);
    void                                  MakeVocab();

    void                                  MakeInputData();
    void                                  MakeLabelData();

    int                                   char2index(char c);

    char                                  index2char(int index);

    Tensor<DTYPE>*                        GetInputData();

    Tensor<DTYPE>*                        GetLabelData();

    int                                   GetTextLength();

    int                                   GetVocabSize();

    char*                                 GetVocab();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    virtual int                           GetLength();

};

template<typename DTYPE> void TextDataset<DTYPE>::Alloc(string File_Path) {

    vocab = new char[VOCAB_LENGTH];

    //File_Reader
    FileReader(File_Path);

    //Dataloader랑 같이 하기
    m_dimOfInput = DATATIME;

    m_numOfInput = text_length/DATATIME;         //drop remainder                                   //여기를 data에 따라서 내가 바꿔줘야됨....;;;;
    m_aaInput = new DTYPE *[m_numOfInput];
    m_aaLabel = new DTYPE *[m_numOfInput];

    //make_vocab
    MakeVocab();

    m_dimOfLabel = vocab_size;

    //make_Input_data
    MakeInputData();

    //make_label_data
    MakeLabelData();
}


template<typename DTYPE> void TextDataset<DTYPE>::Delete() {
    delete []vocab;
    delete []TextData;
}

template<typename DTYPE> void TextDataset<DTYPE>::FileReader(string pFile_Path) {
    ifstream fin;
    fin.open(pFile_Path);

    if(fin.is_open()){

      //파일 사이즈 구하기
      fin.seekg(0, ios::end);
      text_length = fin.tellg();
      fin.seekg(0, ios::beg);        //포인터를 다시 시작위치로 바꿈

      //파일 길이만큼 할당
      TextData = new char[text_length];

      //파일 읽기
      fin.read(TextData, text_length);
      //여기에 NULL 추가


      //소문자로 변환
      //  for(int i=0; i<text_length; i++)
      //      TextData[i] = tolower(TextData[i]);

      // std::cout<<"---------filereader------------"<<'\n';
      // std::cout<<"e"<<TextData[text_length-3]<<"d";

    }
    fin.close();
}

template<typename DTYPE> void TextDataset<DTYPE>::MakeVocab(){

    int flag = 0;
    for(int i=0; i<text_length; i++){

        flag = 0;
        vocab_size = (int)strlen(vocab);

        for(int j=0; j<vocab_size; j++){
            if(vocab[j]==TextData[i])
              flag = 1;
            }

        if(flag==0){
          vocab[vocab_size] = TextData[i];
        }
    }

    vocab_size = (int)strlen(vocab)+1;
    //for(int i=0; i<vocab_size; i++)
    //    std::cout<<i<<"번째 vocab :"<<int(vocab[i])<<'\n';
    sort(vocab, vocab+vocab_size-1);


}

template<typename DTYPE> void TextDataset<DTYPE>::MakeInputData(){

    /*
    if(option == ONEHOT){
        int* onehotvector = new int[vocab_size];

        m_input = new Tensor<DTYPE>(text_length, 1, 1, 1, vocab_size);

        for(int i=0; i<text_length; i++){
            MakeOneHotVector(onehotvector, vocab_size, char2index(TextData[i]));
            for(int j=0; j<vocab_size; j++){
                (*m_input)[Index5D(m_input->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
            }
        }
    }
    */

/*
    //char generation에서 사용하려고 만든거!
    m_input = new Tensor<DTYPE>(TIME, 1, 1, 1, 1);

    for(int i=0; i<text_length; i++){
        (*m_input)[Index5D(m_input->GetShape(), i, 0, 0, 0, 0)] = char2index(TextData[i]);
    }
*/

    std::cout<<"------------MakeInputData------------"<<'\n';

    //dataloader랑 같이!
    int index=0;

    for (int i = 0; i < m_numOfInput; i++) {

        //std::cout<<'\n'<<i<<"번째 dataset"<<'\n';

        m_aaInput[i] = new DTYPE[m_dimOfInput];

        for(int j=0; j<DATATIME; j++){
            //std::cout<<TextData[index];
            m_aaInput[i][j] = char2index(TextData[index]);
            index ++;
        }
    }

}

template<typename DTYPE> void TextDataset<DTYPE>::MakeLabelData(){

/*
    if(option == ONEHOT){
        int* onehotvector = new int[vocab_size];

        m_label = new Tensor<float>(text_length, 1, 1, 1, vocab_size);

        for(int i=0; i<text_length; i++){

            //마지막 data
            if(i==text_length-1){
                MakeOneHotVector(onehotvector, vocab_size, vocab_size-1);
                for(int j=0; j<vocab_size; j++){
                    (*m_label)[Index5D(m_label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
              }
              continue;
            }

            MakeOneHotVector(onehotvector, vocab_size, char2index(TextData[i+1]));
            for(int j=0; j<vocab_size; j++){
                (*m_label)[Index5D(m_label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
            }
        }
    }
*/

    std::cout<<"------------MakelabelData()------------"<<'\n';
    std::cout<<"text_length : "<<text_length<<'\n';
    //dataloader랑 같이!
    //여기서는 그냥 index 넣어주고 아래에서 one-hot으로 만들어주기!
    int index=1;

    for (int i = 0; i < m_numOfInput; i++) {

        m_aaLabel[i] = new DTYPE[m_dimOfInput];

        for(int j=0; j<DATATIME; j++){
              //맨 마지막에 걸리는 경우!
              if( (i+1)*index == text_length){
                  m_aaLabel[i][j] = vocab_size-1;
                  //std::cout<<"eos"<<'\n';
              } else{
                //std::cout<<index<<" : "<<TextData[index]<<'\n';
                m_aaLabel[i][j] = char2index(TextData[index]);
                index ++;
            }
        }
    }

    std::cout<<"------------MakelabelData()------------"<<'\n';


}

template<typename DTYPE> int TextDataset<DTYPE>::char2index(char c){

    for(int index=0; index<vocab_size; index++){
        if(vocab[index]==c)
          return index;
    }
    std::cout<<"error!!!!!!!!!!!!!!!!!!!!!!"<<'\n';
    return -1;
}

template<typename DTYPE> char TextDataset<DTYPE>::index2char(int index){

    if(index == vocab_size-1)
        return 'E';
    else
        return vocab[index];
}

template<typename DTYPE> char* TextDataset<DTYPE>::GetVocab(){

    return vocab;
}

template<typename DTYPE> Tensor<DTYPE>* TextDataset<DTYPE>::GetInputData(){

    return m_input;
}

template<typename DTYPE> Tensor<DTYPE>* TextDataset<DTYPE>::GetLabelData(){
    return m_label;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetTextLength(){
    return text_length;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetVocabSize(){
    return vocab_size;
}


template<typename DTYPE> std::vector<Tensor<DTYPE> *> *TextDataset<DTYPE>::GetData(int idx) {

      //std::cout<<"----------------------GetData--------------"<<'\n';

      std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

      Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(DATATIME, 1, 1, 1, 1);                        //입력은 index
      Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(DATATIME, 1, 1, 1, m_dimOfLabel);             //label은 onthot!

      for (int i = 0; i < DATATIME; i++) {
          (*input)[i] = m_aaInput[idx][i];
      }

      //여기서 one-hot으로 변환해서 넣어주자 그러면!!!

      int* onehotvector = new int[vocab_size];
      for (int i=0; i<DATATIME; i++){
          MakeOneHotVector(onehotvector, vocab_size, m_aaLabel[idx][i]);
          //std::cout<<index2char(m_aaLabel[idx][i]);
          for(int j=0; j<vocab_size; j++){
              (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
          }
      }


      result->push_back(input);
      result->push_back(label);

      return result;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetLength() {
        std::cout<<"GetLength function"<<'\n';
        std::cout<<m_numOfInput<<'\n';
        return m_numOfInput;
}
