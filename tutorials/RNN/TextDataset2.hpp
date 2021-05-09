#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string.h>

#include "../../WICWIU_src/Tensor.hpp"
//#include "../../WICWIU_src/DataLoader.hpp"        // 왜 추가한거지?   Dataset때문에 추가한거 같음... 추측임

using namespace std;

enum OPTION {
    ONEHOT,
    CBOWMODE
};


void MakeOneHotVector(int* onehotvector, int vocab_size, int index){

    for(int i=0; i<vocab_size; i++){
        if(i==index)
            onehotvector[i] = 1;
        else
            onehotvector[i] = 0;
    }
}


void Eliminate(char *str, char ch){

    int length = strlen(str);
    for(int i=0; i<length; i++){

        if(str[i] == ch)
        {
            for(int j=i; j<length; j++)
                str[j] = str[j+1];
        }
    }
}

void replace(char *str, char tar, char repl){

    int length = strlen(str);
    for(int i=0; i<length; i++){
        if(str[i] == tar)
           str[i] = repl;
    }
}

//: public Dataset<DTYPE>{

template<typename DTYPE>
class TextDataset2 {
private:

    string* vocab;    //이제 단어들을 갖고 있어야 하니깐!!!, 중복을 제거한 단어!
    char* TextData;

    string* wordTextData;   //

    int vocab_size;     //반복없는 단어의 개수
    int text_length;    // 이거는 char 개수...
    int word_num;       //단어의 개수

    Tensor<DTYPE>* input;
    Tensor<DTYPE>* label;

    OPTION option;

    int VOCAB_LENGTH;

public:
    TextDataset2(string File_Path, int vocab_length, OPTION pOption) {
        vocab = NULL;
        TextData = NULL;
        wordTextData = NULL;

        vocab_size = 0;
        text_length = 0;
        word_num=0;

        input = NULL;
        label = NULL;

        option = pOption;

        VOCAB_LENGTH = vocab_length;

        Alloc(File_Path);
    }

    virtual ~TextDataset2() {
        Delete();
    }

    //왜 굳이 virtual인거지?
    void                                  Alloc(string File_Path);

    void                                  Delete();

    void                                  FileReader(string pFile_Path);
    void                                  MakeVocab();

    //이거 2개는 그냥 다음 단어 예측하도록 만든거
    void                                  MakeInputData();
    void                                  MakeLabelData();

    //CBOW형태로 만든거
    void                                  MakeCBOWInputData();
    void                                  MakeCBOWLabelData();

    int                                   char2index(string str);

    char                                  index2char(int index);

    Tensor<DTYPE>*                        GetInputData();

    Tensor<DTYPE>*                        GetLabelData();

    int                                   GetTextLength();

    int                                   GetWordNum();

    int                                   GetVocabLength();

    //virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    //virtual int                           GetLength();

};

template<typename DTYPE> void TextDataset2<DTYPE>::Alloc(string File_Path) {

    vocab = new string[VOCAB_LENGTH];
    wordTextData = new string[VOCAB_LENGTH];

    //File_Reader
    FileReader(File_Path);

    //make_vocab
    MakeVocab();

    if(option==ONEHOT){
        //make_Input_data
        MakeInputData();

        //make_label_data
        MakeLabelData();
    }
    else if(option == CBOWMODE){
        MakeCBOWInputData();
        MakeCBOWLabelData();
    }
}


template<typename DTYPE> void TextDataset2<DTYPE>::Delete() {
    delete []vocab;
    delete []TextData;
}

template<typename DTYPE> void TextDataset2<DTYPE>::FileReader(string pFile_Path) {
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
      for(int i=0; i<text_length; i++)
          TextData[i] = tolower(TextData[i]);

      //빈공간 다 없애고 sp만 남기기
      Eliminate(TextData, '\n');
      Eliminate(TextData, ':');
      Eliminate(TextData, ',');
      Eliminate(TextData, '.');

      replace(TextData, '\r', ' ');

      //하나하나 ascii 값 찍어뵉
      // for(int i=0; i<text_length; i++)
      //     std::cout<<i<<"번째 = "<<(int)TextData[i]<<'\n';

      std::cout<<TextData<<'\n';
    }


    //제거했으니깐 길이 다시 설정해주기
    text_length = strlen(TextData);

    //단어의 개수!

    fin.close();
}

template<typename DTYPE> void TextDataset2<DTYPE>::MakeVocab(){

    int flag = 0;
    char* token = strtok(TextData, " ");    //단어 하나씩 가져오기 위한 변수

    int word_count = 0;

    while(token != NULL){

        wordTextData[word_count] = token;

        //중복확인하기
        for(int i=0; i<vocab_size; i++){
            if(vocab[i] == token)   //이거 비교 가능한지 모르겠음!!!
                flag = 1;
        }

        //중복이 아니라면
        if(flag == 0){
            vocab[vocab_size] = token;    //이거도 가능한지 모르겠음...
            vocab_size++;
        }

        token = strtok(NULL, " ");
        //단어 개수
        word_count++;
    }

    sort(vocab, vocab+vocab_size-1);

    //출력해보기
    for(int i=0; i<vocab_size; i++)
        std::cout<<i<<"번째 vocab : "<<vocab[i]<<'\n';

    word_num = word_count;

    std::cout<<"단어 개수 : "<<word_num<<'\n';

}

template<typename DTYPE> void TextDataset2<DTYPE>::MakeInputData(){


         int* onehotvector = new int[vocab_size];

         input = new Tensor<DTYPE>(word_num, 1, 1, 1, vocab_size);

         for(int i=0; i<word_num; i++){

              MakeOneHotVector(onehotvector, vocab_size, char2index(wordTextData[i]));
              for(int j=0; j<vocab_size; j++){
                  (*input)[Index5D(input->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
              }
         }


}

template<typename DTYPE> void TextDataset2<DTYPE>::MakeLabelData(){

        int* onehotvector = new int[vocab_size];

        label = new Tensor<float>(word_num, 1, 1, 1, vocab_size);

        for(int i=0; i<word_num; i++){

            //마지막 data
            if(i==word_num-1){
                  MakeOneHotVector(onehotvector, vocab_size, vocab_size-1);
                  for(int j=0; j<vocab_size; j++){
                      (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
                  }
              continue;
            }

            MakeOneHotVector(onehotvector, vocab_size, char2index(wordTextData[i+1]));
            for(int j=0; j<vocab_size; j++){
                (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
            }
        }

}







template<typename DTYPE> void TextDataset2<DTYPE>::MakeCBOWInputData(){


         //
         int* onehotvector = new int[vocab_size];

         input = new Tensor<DTYPE>(word_num, 1, 1, 1, vocab_size);

         for(int i=0; i<word_num-2; i++){

              //앞쪽에 해당하는 context input1
              MakeOneHotVector(onehotvector, vocab_size, char2index(wordTextData[i]));
              for(int j=0; j<vocab_size; j++){
                  (*input)[Index5D(input->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
              }

              //앞쪽에 해당하는 context input1
              MakeOneHotVector(onehotvector, vocab_size, char2index(wordTextData[i+2]));
              for(int j=0; j<vocab_size; j++){
                  (*input)[Index5D(input->GetShape(), i, 0, 0, 0, vocab_size+j)] = onehotvector[j];
              }
         }


}

template<typename DTYPE> void TextDataset2<DTYPE>::MakeCBOWLabelData(){


        int* onehotvector = new int[vocab_size];

        label = new Tensor<float>(word_num, 1, 1, 1, vocab_size);

        for(int i=1; i<word_num-1; i++){

            MakeOneHotVector(onehotvector, vocab_size, char2index(wordTextData[i]));
            for(int j=0; j<vocab_size; j++){
                (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
            }
        }

}



template<typename DTYPE> int TextDataset2<DTYPE>::char2index(string str){

    for(int index=0; index<vocab_size; index++){
        if(vocab[index]==str){
            return index;
        }
    }
    return -1;
}

template<typename DTYPE> char TextDataset2<DTYPE>::index2char(int index){

    return vocab[index];
}

template<typename DTYPE> Tensor<DTYPE>* TextDataset2<DTYPE>::GetInputData(){

    return input;
}

template<typename DTYPE> Tensor<DTYPE>* TextDataset2<DTYPE>::GetLabelData(){
    return label;
}

template<typename DTYPE> int TextDataset2<DTYPE>::GetTextLength(){
    return text_length;
}

template<typename DTYPE> int TextDataset2<DTYPE>::GetWordNum(){
    return word_num;
}

template<typename DTYPE> int TextDataset2<DTYPE>::GetVocabLength(){
    return vocab_size;
}
