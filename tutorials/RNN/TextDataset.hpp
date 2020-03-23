#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>    //strlen 때문에 추가한 해더

#include "../../WICWIU_src/Tensor.hpp"
#include "../../WICWIU_src/DataLoader.hpp"        // 왜 추가한거지?   Dataset때문에 추가한거 같음... 추측임


using namespace std;


/*
int char2index(char* vocab, char c){

    for(int index=0; index<strlen(vocab); index++){
        if(vocab[index]==c)
          return index;
    }
    std::cout<<"해당 문자는 vocab에 없습니다."<<'\n';
    exit(0);
}

char index2char(char* vocab, int index){

    return vocab[index];
}
*/
void makeonehotvector(int* onehotvector, int vocab_size, int index){

    for(int i=0; i<vocab_size; i++){
        if(i==index)
            onehotvector[i] = 1;
        else
            onehotvector[i] = 0;
    }
}


template<typename DTYPE>
class TextDataset : public Dataset<DTYPE>{
private:

    char* vocab ;
    char* TextData;

    int vocab_length;
    int text_length;

    Tensor<DTYPE>* input;
    Tensor<DTYPE>* label;

public:
    TextDataset(string File_Path) {
        vocab = new char[100];
        TextData = NULL;

        vocab_length = 0;
        text_length = 0;

        input = NULL;               // NULL 맞는지 모르겠음
        label = NULL;

        Alloc(File_Path);
    }

    virtual ~TextDataset() {
        Delete();
    }

    virtual void                          Alloc(string File_Path);

    virtual void                          Delete();

    void                                  File_Reader(string pFile_Path);
    void                                  make_vocab();

    void                                  make_Input_data();
    void                                  make_Label_data();

    int                                   char2index(char c);

    char                                  index2char(int index);

    Tensor<DTYPE>*                        GetInputData();

    Tensor<DTYPE>*                        GetLabelData();

    int                                   GetTextLength();

    int                                   GetVocabLength();

    //virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    //virtual int                           GetLength();

};

template<typename DTYPE> void TextDataset<DTYPE>::Alloc(string File_Path) {

    //File_Reader
    File_Reader(File_Path);

    //make_vocab
    make_vocab();
    //make_Input_data
    make_Input_data();
    //make_label_data
    make_Label_data();
}

template<typename DTYPE> void TextDataset<DTYPE>::Delete() {

}

template<typename DTYPE> void TextDataset<DTYPE>::File_Reader(string pFile_Path) {
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
    }
    fin.close();
}

template<typename DTYPE> void TextDataset<DTYPE>::make_vocab(){

    int flag = 0;
    for(int i=0; i<text_length; i++){

        flag = 0;
        vocab_length = (int)strlen(vocab);

        for(int j=0; j<vocab_length; j++){
            if(vocab[j]==TextData[i])
              flag = 1;
            }

        if(flag==0){
          vocab[vocab_length] = TextData[i];
        }
    }
    sort(vocab, vocab+vocab_length);
}

template<typename DTYPE> void TextDataset<DTYPE>::make_Input_data(){

    int* onehotvector = new int[vocab_length];

    input = new Tensor<DTYPE>(text_length, 1, 1, 1, vocab_length);

    for(int i=0; i<text_length; i++){
        for(int j=0; j<vocab_length; j++){
            makeonehotvector(onehotvector, vocab_length, char2index(TextData[i]));
            (*input)[Index5D(input->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
        }
    }

}

template<typename DTYPE> void TextDataset<DTYPE>::make_Label_data(){

    int* onehotvector = new int[vocab_length];

    label = new Tensor<float>(text_length, 1, 1, 1, vocab_length);

    for(int i=0; i<text_length; i++){

        //맨 마지막 label 처리
        if(i==text_length-1){
          for(int j=0; j<vocab_length; j++){
              (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = 0;
          }
          continue;
        }

        for(int j=0; j<vocab_length; j++){
            makeonehotvector(onehotvector, vocab_length, char2index(TextData[i+1]));
            (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
        }
    }
}

template<typename DTYPE> int TextDataset<DTYPE>::char2index(char c){

    for(int index=0; index<strlen(vocab); index++){
        if(vocab[index]==c)
          return index;
    }
    std::cout<<"해당 문자는 vocab에 없습니다."<<'\n';
    exit(0);
}

template<typename DTYPE> char TextDataset<DTYPE>::index2char(int index){

    return vocab[index];
}

template<typename DTYPE> Tensor<DTYPE>* TextDataset<DTYPE>::GetInputData(){

    return input;
}

template<typename DTYPE> Tensor<DTYPE>* TextDataset<DTYPE>::GetLabelData(){
    return label;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetTextLength(){
    return text_length;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetVocabLength(){
    return vocab_length;
}
