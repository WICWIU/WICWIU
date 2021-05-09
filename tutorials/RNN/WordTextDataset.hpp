#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string.h>


#include "../../WICWIU_src/Tensor.hpp"
#include "../../WICWIU_src/DataLoader.hpp"

#define DATATIME        12

#define NUMOFVOCAB        12            //Subtext8-2.txt

#define NUMOFWORD         12          //subtext8-2.txt에서 단어의 개수!



using namespace std;

enum OPTION {
    ONEHOT,
    CBOWMODE,
    SKIPGRAM,
    ACCURACY
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
                str[j] = str[j+1];            //+1로 처리해주기 때문에 NULL까지 옮겨줌!!!
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


template<typename DTYPE>
class WordTextDataset : public Dataset<DTYPE> {
private:

    //textData에 있던 변수들!!!
    string* vocab;          //이제 단어들을 갖고 있어야 하니깐!!!, 중복을 제거한 단어!
    char* TextData;         //파일에서 읽어오기!

    string* wordTextData;   //strtok를 사용하면 원래 data가 바뀌어서 추가한거!

    int vocab_size;         //반복없는 단어의 개수
    int text_length;        // 이거는 char 개수...     //나중에 fastText에서 필요할 수도 있을거 같아서 남겨둠!!!
    int word_num;           //단어의 개수

    //각 단어가 몇 번 나왔는지는 없음...! 이거 배열을 하나 더 만들어서 가능할듯!!!   -> sampling 할때 필요!


    OPTION option;

    //word2vec.hpp에 있던 거!!!
    DTYPE **m_aaInput;
    DTYPE **m_aaLabel;

    int m_numOfInput;             //input data의 개수!!!

    int m_dimOfInput;
    int m_dimOfLabel;


public:
    WordTextDataset(string File_Path, OPTION pOption) {
          vocab = NULL;
          TextData = NULL;
          wordTextData = NULL;

          vocab_size = 0;
          text_length = 0;
          word_num=0;

          option = pOption;

          //word2vec.hpp에 있던거!
          m_aaInput = NULL;
          m_aaLabel = NULL;

          m_numOfInput = 0;

          m_dimOfInput = 0;
          m_dimOfLabel = 0;

          Alloc(File_Path);
    }

    virtual ~WordTextDataset() {
        Delete();
    }

    //왜 굳이 virtual인거지?
    void                                  Alloc(string pTextPath);

    void                                  Delete();

    void                                  FileReader(string pFile_Path);
    void                                  MakeVocab();

    void                                  MakeInputData();

    void                                  MakeLabelData();

    int                                   word2index(string str);

    string                                index2word(int index);

    int                                   GetTextLength();

    int                                   GetWordNum();

    string*                               GetVocab();

    int                                   GetVocabSize();

    int                                   GetInputDim();
    int                                   GetLabelDim();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    virtual int                           GetLength();

};

template<typename DTYPE> void WordTextDataset<DTYPE>::Alloc(string File_Path) {


    vocab        = new string[NUMOFVOCAB];      //원래는 NUMOFWORD+2 였음!  이거 문제가 아님!!!
    wordTextData = new string[NUMOFWORD];

    m_numOfInput = NUMOFWORD/DATATIME;     //sos eos 추가
    m_dimOfInput = 1;
    m_dimOfLabel = NUMOFVOCAB;                 //positive sample 추가


    m_aaInput = new DTYPE *[m_numOfInput];
    m_aaLabel = new DTYPE *[m_numOfInput];


    FileReader(File_Path);


    MakeVocab();

    MakeInputData();

    std::cout<<"input"<<'\n';

    MakeLabelData();

}


template<typename DTYPE> void WordTextDataset<DTYPE>::Delete() {
    delete []vocab;
    delete []TextData;
}

template<typename DTYPE> void WordTextDataset<DTYPE>::FileReader(string pFile_Path) {

    std::cout<<"FileReader"<<'\n';

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

      //소문자로 변환
      // for(int i=0; i<text_length; i++){
      //     TextData[i] = tolower(TextData[i]);
      // }

      // std::cout<<"제거 전 text 길이 : "<<text_length<<'\n';
      // std::cout<< strlen(TextData)<<'\n';

      //빈공간 다 없애고 sp만 남기기
      // Eliminate(TextData, '\n');
      // Eliminate(TextData, ':');
      // Eliminate(TextData, ',');
      // Eliminate(TextData, '.');
      // Eliminate(TextData, ';');
      // Eliminate(TextData, '?');
      // Eliminate(TextData, '!');
      // replace(TextData, '\r', ' ');

    }

    text_length = strlen(TextData);     //strlen원리가 NULL를 찾을 때 까지여서 마지막에 NULL이 자동으로 추가된거 같음!

    fin.close();
}



template<typename DTYPE> void WordTextDataset<DTYPE>::MakeVocab(){

    std::cout<<"---------------------Makevocab------------------"<<'\n';

    int flag = 0;
    char* token = strtok(TextData, " ");    //단어 하나씩 가져오기 위한 변수

    int word_count =0;

    while(token != NULL){

          //wordTextData[NUMOFWORD-4] = token;
          wordTextData[word_count] = token;         //301291

           std::cout<<"word count : "<<word_count<<" "<<token<<'\n'<<"끝"<<'\n';
           //std::cout<<word_count<<'\n';

           // if(word_count%100000==0)
           //     std::cout<<word_count<<'\n';

          //중복확인하기
          for(int i=0; i<vocab_size; i++){
              if(vocab[i] == token){
                  flag = 1;
                  break;
                  // std::cout<<"중복된 단어 : "<<token<<'\n';
              }
          }

          //중복이 아니라면
          if(flag == 0){
              vocab[vocab_size] = token;
              vocab_size++;
          }

          token = strtok(NULL, " ");
          //단어 개수
          word_count++;
          flag = 0;
    }

    std::cout<<"dddd"<<'\n';

    //SOS하고 EOS 추가하려면 여기서!!! 그리고 vocab수하고 파일에 있는 단어수도 증가시키기???
    //sos : "<s>"    eos : "<\s>"

    // std::cout<<index2word(vocab_size-1)<<" "<<index2word(vocab_size)<<"???"<<'\n';

    vocab[vocab_size++] = "<s>";
    //매우 중요!!! 여기서 ++해주는 이유는!!! 이제 모든 index 접근을.... <=가 아닌 <로 for문을 만들어서... 문제가 생김!! 이 문제는 단순히 text8에서만의 문제가 아니라!!! 다른 operator에서도 이제 접근할 때 for문에서 <로 해서 이제 문제가 생김!!! 마지막 <e>이거 때문에!!!
    vocab[vocab_size++] = "<e>";

    //count를 사용해서 subsampling을 하기위해!!! 굳이 sort를 할 필요가 없다고 생각함!!! 결과에 영향 없을 듯!
    //sort(vocab, vocab+vocab_size-1);

    word_num = word_count;

    std::cout<<"파일에 있는 단어의 개수(중복 포함) : "<<word_num<<'\n';
    std::cout<<"파일에 있는 단어의 개수(중복 미포함) : "<<vocab_size<<'\n';
    // std::cout<<index2word(vocab_size-2)<<" : "<<wordFrequency[vocab_size-2]<<" "<<index2word(vocab_size-1)<<" : "<<wordFrequency[vocab_size-1]<<'\n';
    // std::cout<<word2index("<s>")<<" "<<word2index("<e>")<<'\n';
    // std::cout<<"work : "<<wordFrequency[word2index("work")]<<'\n';

}

template<typename DTYPE> void WordTextDataset<DTYPE>::MakeInputData(){


    //dataloader랑 같이!
    int index=0;

    for (int i = 0; i < m_numOfInput; i++) {

        m_aaInput[i] = new DTYPE[m_dimOfInput];

        for(int j=0; j<DATATIME; j++){
            m_aaInput[i][j] = word2index(wordTextData[index]);
            index ++;
        }
    }

}

template<typename DTYPE> void WordTextDataset<DTYPE>::MakeLabelData(){

    //dataloader랑 같이!
    //여기서는 그냥 index 넣어주고 아래에서 one-hot으로 만들어주기!
    int index=1;

    for (int i = 0; i < m_numOfInput; i++) {

        m_aaLabel[i] = new DTYPE[m_dimOfInput];

        for(int j=0; j<DATATIME; j++){

            //맨 마지막에 걸리는 경우!
            if(index == text_length-1){
                m_aaLabel[i][j] = vocab_size-1;
            } else{

              m_aaLabel[i][j] = word2index(wordTextData[index]);
              index ++;
          }
        }
    }


}





template<typename DTYPE> int WordTextDataset<DTYPE>::word2index(string str){

    for(int index=0; index<vocab_size; index++){
        if(vocab[index]==str){
            return index;
        }
    }
    return -1;
}

template<typename DTYPE> string WordTextDataset<DTYPE>::index2word(int index){

    return vocab[index];
}

//이거는 필요한지 모르겠음....
template<typename DTYPE> int WordTextDataset<DTYPE>::GetTextLength(){
    return text_length;
}

template<typename DTYPE> int WordTextDataset<DTYPE>::GetWordNum(){
    return word_num;
}

template<typename DTYPE> string* WordTextDataset<DTYPE>::GetVocab(){
    return vocab;
}

template<typename DTYPE> int WordTextDataset<DTYPE>::GetVocabSize(){
    return vocab_size;
}

template<typename DTYPE> int WordTextDataset<DTYPE>::GetInputDim(){
    return m_dimOfInput;
}

template<typename DTYPE> int WordTextDataset<DTYPE>::GetLabelDim(){
    return m_dimOfLabel;
}


template<typename DTYPE> std::vector<Tensor<DTYPE> *> *WordTextDataset<DTYPE>::GetData(int idx) {


      std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

      Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(DATATIME, 1, 1, 1, 1);                        //입력은 index
      Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(DATATIME, 1, 1, 1, m_dimOfLabel);             //label은 onthot!

      for (int i = 0; i < DATATIME; i++) {
          (*input)[i] = m_aaInput[idx][i];
      }

      //여기서 one-hot으로 변환해서 넣어주자 그러면!!!

      int* onehotvector = new int[vocab_size];
      for (int i=0; i<DATATIME; i++){
          MakeOneHotVector(onehotvector, vocab_size, m_aaInput[idx][i]);
          for(int j=0; j<vocab_size; j++){
              (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
          }
      }


      result->push_back(input);
      result->push_back(label);

      return result;
}

template<typename DTYPE> int WordTextDataset<DTYPE>::GetLength() {
        return m_numOfInput;
}
