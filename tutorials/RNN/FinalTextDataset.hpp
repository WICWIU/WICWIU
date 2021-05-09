#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <sstream>

#include "../../WICWIU_src/Tensor.hpp"
#include "../../WICWIU_src/DataLoader.hpp"
#include "../../WICWIU_src/Dataset.hpp"

using namespace std;


/*

  sentence generator 부분도 수정해야됨!!!
  getvocab이 바뀌면서....
  넘겨주는게 배열에서.... map으로 변경됨!

    //seqLength * batchsize 가 word의 개수보다 많은 경우는?... 이 경우는 error가 나는거 같아...


*/

template<typename DTYPE> class TextDataset : public Dataset<DTYPE>{ //전처리 옵션 관리하고
private:
  string path;
  char* TextData;
  int text_length;
  int line_length;
  int max_sentence_length;
  //-----Field 클래스에서 차용------//
  //옵션들
  bool sequential = true;
  bool lower = true;
  bool padding = true;
  bool unk = true;
  //-----Vocab 클래스에서 차용------//
  map<int, string>* m_pIndex2Vocab;
  map<string, int>* m_pVocab2Frequency;
  map<string, int>* m_pVocab2Index;
  int n_vocabs;
  //-----넘겨주는 Data 관련!------//
  DTYPE **m_aaInput;
  DTYPE **m_aaLabel;

  int m_numOfInput;             //input data의 개수!!!                 //여기서 사용 X
  int m_window;                 //window size -> 홀수가 기본이겠지!
  int m_negative;

  int m_dimOfInput;
  int m_dimOfLabel;

public:
  TextDataset();

  void                         ReadFile(string path);

  void                         Pad(); //아직!!!!

  void                         AddSentence(string sentence);

  void                         AddWord(string word);

  vector<string>               SplitBy(string input, char delimiter);

  string                       Preprocess(string sentence);

  string                       Preprocess(char* sentence);

  string                       Remove(string sentence, string delimiters);

  virtual void                 BuildVocab();

  virtual                      ~TextDataset();

  int                          GetTextLength();

  void                         SetLineLength(int n);
  int                          GetLineLength();

  map<int, string>*            GetpIndex2Vocab();

  map<string, int>*            GetpVocab2Frequency();

  map<string, int>*            GetpVocab2Index();

  int                          GetNumberofVocabs();

  int                          GetNumberofWords();

  int                          GetMaxSentenceLength();

  char*                        GetTextData();

  // virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

  virtual int                           GetLength();            //추가!         //이거 virtual로 해서 상속하는 class에서 만들어 주도록 설정!!!

};

template<typename DTYPE> map<int, string>* TextDataset<DTYPE>::GetpIndex2Vocab(){
  return m_pIndex2Vocab;
}

template<typename DTYPE> map<string, int>* TextDataset<DTYPE>::GetpVocab2Frequency(){
  return m_pVocab2Frequency;
}

template<typename DTYPE> map<string, int>* TextDataset<DTYPE>::GetpVocab2Index(){
  return m_pVocab2Index;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetNumberofWords(){
  map<string, int>::iterator it;
  int result = 0;

  for(it=m_pVocab2Frequency->begin(); it!=m_pVocab2Frequency->end(); it++){
    result += it->second;
  }
  return result;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetNumberofVocabs(){
  return n_vocabs;                                                              // -1 수정함!
}

template<typename DTYPE> char* TextDataset<DTYPE>::GetTextData(){
  return TextData;
}

template<typename DTYPE> TextDataset<DTYPE>::TextDataset() {
  path="";
  text_length = 0;
  line_length = 0;
  max_sentence_length = 0;
  m_pIndex2Vocab = new map<int, string>();
  m_pVocab2Frequency = new map<string, int>();
  m_pVocab2Index = new map<string, int>();
  n_vocabs = 0;
  m_aaInput = NULL;
  m_aaLabel = NULL;
  m_numOfInput = 0;
  m_window     = 0;
  m_negative   = 0;
  m_dimOfInput = 0;
  m_dimOfLabel = 0;

  AddWord("<PAD>");
  AddWord("<SOS>");
  AddWord("<EOS>");

}

template<typename DTYPE> TextDataset<DTYPE>::~TextDataset() {
  cout << "TextDataset 소멸자 호출" << endl;
  delete[] TextData;
  delete m_pIndex2Vocab;
  delete m_pVocab2Frequency;
  delete m_pVocab2Index;
}


template<typename DTYPE> void TextDataset<DTYPE>::ReadFile(string path) {
  cout<<"<<<<<<<<<<<<<<<<  FileReader  >>>>>>>>>>>>>>>>>>>>"<<endl;
    this->path = path;
    cout << this->path << endl;
    ifstream fin;
    fin.open(path);

    if(fin.is_open()) {

      fin.seekg(0, ios::end);
      text_length = fin.tellg();
      fin.tellg();
      fin.seekg(0, ios::beg);

      TextData = new char[text_length];
      //파일 읽기
      fin.read(TextData, text_length);

      text_length = strlen(TextData);
      fin.close();
    }
    //cout<<text_length<<endl;
}

template<typename DTYPE> void TextDataset<DTYPE>::AddSentence(string sentence){
  //cout<<"<<<<<<<<<<<<<<<<  AddSentence  >>>>>>>>>>>>>>>>>>>>"<<endl;
  vector<string> words = SplitBy(sentence, ' ');
  if(words.size() > max_sentence_length) max_sentence_length = words.size();
  for(string word: words){
    AddWord(word);
  }
  vector<string>().swap(words);
}

template<typename DTYPE> void TextDataset<DTYPE>::AddWord(string word){
  if(m_pVocab2Index->find(word)==m_pVocab2Index->end()){
    m_pVocab2Index->insert(make_pair(word, n_vocabs));
    m_pVocab2Frequency->insert(make_pair(word, 1));
    m_pIndex2Vocab->insert(make_pair(n_vocabs, word));
    n_vocabs ++;
  }
  else{
    m_pVocab2Frequency->at(word)++;
  }
}

template<typename DTYPE> vector<string> TextDataset<DTYPE>::SplitBy(string input, char delimiter) {
  vector<string> answer;
  stringstream ss(input);
  string temp;

  while (getline(ss, temp, delimiter)) {
      answer.push_back(temp);
  }
  return answer;
}



template<typename DTYPE> string TextDataset<DTYPE>::Preprocess(string sentence) {
  if(lower){
    transform(sentence.begin(), sentence.end(), sentence.begin(), [](unsigned char c){ return std::tolower(c); });
  }
  sentence = Remove(sentence, ",.?!\"\'><:-");
  return sentence;
}

template<typename DTYPE> string TextDataset<DTYPE>::Preprocess(char* sentence){
  string new_sentence(sentence);
  return Preprocess(new_sentence);
}

template<typename DTYPE> string TextDataset<DTYPE>:: Remove(string str, string delimiters){
  vector<string> splited_delimiters;
  for(int i=0; i<delimiters.length(); i++){
    splited_delimiters.push_back(delimiters.substr(i,1));
  }
  for(string delimiter : splited_delimiters){
    int k = str.find(delimiter);
    while(k>=0){
      string k_afterStr = str.substr(k+1, str.length()-k);
      str = str.erase(k) + k_afterStr;
      k = str.find(delimiter);
    }
  }
    return str;
}

template<typename DTYPE> void TextDataset<DTYPE>:: BuildVocab(){
    cout<<"<<<<<<<<<<<<<<<<  BuildVocab 호출 >>>>>>>>>>>>>>>>>>>>"<<endl;
};

template<typename DTYPE> int TextDataset<DTYPE>:: GetTextLength(){
  return text_length;
}
template<typename DTYPE> int TextDataset<DTYPE>:: GetLineLength(){
  return line_length;
}
template<typename DTYPE> void TextDataset<DTYPE>:: SetLineLength(int n){
  line_length = n;
}
template<typename DTYPE> int TextDataset<DTYPE>::GetMaxSentenceLength(){
  return max_sentence_length;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetLength(){
  //return (int)line_length/2;                                                    //여기 찬효랑 이야기 해야됨! 이렇게 하기로 !!! 짝이 되어 있어서                                                                                //즉 한줄에 문장이 2개 있어서!!!
}


// template<typename DTYPE> std::vector<Tensor<DTYPE> *>* TextDataset<DTYPE>:: GetData(int idx){
//   std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

//   Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfInput);
//   Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfLabel);

//   for (int i = 0; i < m_dimOfInput; i++) {
//       //이거는 전체 단어의 개수 안 맞춰주면 이렇게 됨!!!
//       if(m_aaInput[idx][i]==-1)
//           std::cout<<'\n'<<"****************************************************************************************음수존재..."<<'\n';
//       (*input)[i] = m_aaInput[idx][i];
//   }

//   //(*label)[ (int)m_aaLabel[idx][0] ] = 1.f;
//   (*label)[0] = 1.f;

//   result->push_back(input);
//   result->push_back(label);

//   return result;
// }


//--------------------------------------------------병렬 코퍼스 데이터--------------------------------------------------//
template<typename DTYPE>
class ParalleledCorpusDataset : public TextDataset<DTYPE>{ //파일 경로 받아서 실제 보캡, Paired문장 등 보관
private:
  pair<string, string> m_languageName;
  vector< pair<string, string> >* m_pairedSentences;          // paired data
  vector< pair< int*, int* > >* m_pairedIndexedSentences;
public:
  ParalleledCorpusDataset(string path, string srcName, string dstName);

  void                                   Alloc(string path);

  void                                   MakeLineData();

  virtual void                           BuildVocab();

  virtual                                ~ParalleledCorpusDataset();

  //virtual std::vector<Tensor<DTYPE>*>*   GetData(int idx);

  vector< pair< int*, int* > >*         GetPairedIndexedSentences();
};


template<typename DTYPE> ParalleledCorpusDataset<DTYPE>::ParalleledCorpusDataset(string path, string srcName, string dstName) : TextDataset<DTYPE>::TextDataset() {
  m_languageName = make_pair(srcName, dstName);
  m_pairedIndexedSentences = NULL;
  m_pairedSentences = NULL;
  Alloc(path);
}
template<typename DTYPE> ParalleledCorpusDataset<DTYPE>::~ParalleledCorpusDataset() {
    cout << "ParalleledCorpusDataset 소멸자 호출" << endl;
    delete m_pairedSentences;
    delete m_pairedIndexedSentences;
}


template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::Alloc(string path) {
  m_pairedSentences = new vector< pair<string, string> >();
  m_pairedIndexedSentences = new vector< pair< int*, int* > >();
  this->ReadFile(path);
}
template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::BuildVocab() {
  MakeLineData();
  cout<<"<<<<<<<<<<<<<<<<  BuildVocab 호출 >>>>>>>>>>>>>>>>>>>>"<<endl;
  for(int i=0; i<m_pairedSentences->size(); i++){
    vector<string> temp_words = this->SplitBy(m_pairedSentences->at(i).first, ' ');
    vector<int> temp_first_indexed_words;
    for(string word: temp_words){
      //cout << word << endl;
      temp_first_indexed_words.push_back(this->GetpVocab2Index()->at(word));
    }
    while(temp_first_indexed_words.size() > this->GetMaxSentenceLength()){
      temp_first_indexed_words.push_back(this->GetpVocab2Index()->at("<PAD>"));
    }
    temp_words = this->SplitBy(m_pairedSentences->at(i).second, ' ');
    vector<int> temp_second_indexed_words;
    for(string word: temp_words){
      temp_second_indexed_words.push_back(this->GetpVocab2Index()->at(word));
    }
    while(temp_second_indexed_words.size() > this->GetMaxSentenceLength()){
      temp_second_indexed_words.push_back(this->GetpVocab2Index()->at("<PAD>"));
    }
    int* left = new int[this->GetMaxSentenceLength()]{0, };
    int* right = new int[this->GetMaxSentenceLength()]{0, };
    std::copy(temp_first_indexed_words.begin(), temp_first_indexed_words.end(), left);
    std::copy(temp_second_indexed_words.begin(), temp_second_indexed_words.end(), right);
    m_pairedIndexedSentences->push_back(make_pair(left, right));
  }
  m_pairedIndexedSentences->shrink_to_fit();
}

template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::MakeLineData() { // 확인완료

    cout<<"<<<<<<<<<<<<<<<<  MakeLineData  >>>>>>>>>>>>>>>>>>>>"<<endl;
    //cout<<strlen(TextData)<<endl;
    char* token = strtok(this->GetTextData(), "\t\n");
    char* last_sentence = NULL;

    while(token != NULL) {
      //cout<<token<<endl;              //DEBUG
      if(this->GetLineLength()%2==0){
        last_sentence = token;                                              // paired data를 만들기위해 앞에 오는 line 임시 저장
      }
      else {
        string str_last_sentence = this->Preprocess(last_sentence);
        string str_token = this->Preprocess(token);
        m_pairedSentences->push_back(make_pair(str_last_sentence, str_token));           // paired data 저장
        this->AddSentence(this->Preprocess(m_pairedSentences->back().first));
        this->AddSentence(this->Preprocess(m_pairedSentences->back().second));
      }
      //temp->line->push_back(token);                                         // 각 언어에 line 저장
      //MakeVocab(token);
      token = strtok(NULL, "\t\n");
      int temp_lineLength = this->GetLineLength();
      if(temp_lineLength%10000==0)
        cout<<"line_length = "<<temp_lineLength<<endl;

      this->SetLineLength(++temp_lineLength);
    }
    m_pairedSentences->shrink_to_fit();
    //text_lines /=2;
  }

  template<typename DTYPE> vector< pair< int*, int* > >* ParalleledCorpusDataset<DTYPE>::GetPairedIndexedSentences(){
    return m_pairedIndexedSentences;
  }

// template<typename DTYPE> std::vector<Tensor<DTYPE> *>* ParalleledCorpusDataset<DTYPE>:: GetData(int idx){
//   std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

//   Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfInput);
//   Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfLabel);

//   for (int i = 0; i < m_dimOfInput; i++) {
//       //이거는 전체 단어의 개수 안 맞춰주면 이렇게 됨!!!
//       if(m_aaInput[idx][i]==-1)
//           std::cout<<'\n'<<"****************************************************************************************음수존재..."<<'\n';
//       (*input)[i] = m_aaInput[idx][i];
//   }

//   //(*label)[ (int)m_aaLabel[idx][0] ] = 1.f;
//   (*label)[0] = 1.f;

//   result->push_back(input);
//   result->push_back(label);

//   return result;
// }

// int main(){
//   ParalleledCorpusDataset<float>* translation_data = new ParalleledCorpusDataset<float>("eng-fra.txt", "eng", "fra");

//   translation_data->BuildVocab();
//   cout << "LineLength:  " << translation_data->GetLineLength() << endl;
//   cout << "TextLength:  " << translation_data->GetTextLength() << endl;
//   cout << "NumofWords:  " << translation_data->GetNumberofWords() << endl;
//   cout << "NumofVocabs: " << translation_data->GetNumberofVocabs() << endl;

//   map<int, string> *index2vocab = translation_data->GetpIndex2Vocab();
//   map<int, string> :: iterator iter;
//   int count = 0;
//   for ( iter = index2vocab->begin(); iter != index2vocab->end(); iter++ ){
//     cout << iter->first << " : " << iter->second << "\t";
//     if(count%5==0){
//       cout << endl;
//     }
//     count ++;
//   }
// }

template<typename DTYPE>
class RNNParalleledCorpusDataset : public TextDataset<DTYPE>{ //파일 경로 받아서 실제 보캡, Paired문장 등 보관
private:
  pair<string, string> m_languageName;
  vector< pair<string, string> >* m_pairedSentences;          // paired data
  vector< pair< int*, int* > >* m_pairedIndexedSentences;
  int m_EncoderMaxSentenceLength;
  int m_DecoderMaxSentenceLength;

public:
  RNNParalleledCorpusDataset(string path, string srcName, string dstName);

  void                                   Alloc(string path);

  void                                   MakeLineData();

  virtual void                           BuildVocab();

  virtual                                ~RNNParalleledCorpusDataset();

  virtual int                            GetLength();

  virtual std::vector<Tensor<DTYPE>*>*   GetData(int idx);

  vector< pair< int*, int* > >*         GetPairedIndexedSentences();
  int                                   GetEncoderMaxTime();
  int                                   GetDecoderMaxTime();
};


template<typename DTYPE> RNNParalleledCorpusDataset<DTYPE>::RNNParalleledCorpusDataset(string path, string srcName, string dstName) : TextDataset<DTYPE>::TextDataset() {
  m_languageName = make_pair(srcName, dstName);
  m_pairedIndexedSentences = NULL;
  m_pairedSentences = NULL;
  m_EncoderMaxSentenceLength = 0;
  m_DecoderMaxSentenceLength = 0;
  Alloc(path);
}
template<typename DTYPE> RNNParalleledCorpusDataset<DTYPE>::~RNNParalleledCorpusDataset() {
    cout << "RNNParalleledCorpusDataset 소멸자 호출" << endl;
    delete m_pairedSentences;
    delete m_pairedIndexedSentences;
}


template<typename DTYPE> void RNNParalleledCorpusDataset<DTYPE>::Alloc(string path) {
  m_pairedSentences = new vector< pair<string, string> >();
  m_pairedIndexedSentences = new vector< pair< int*, int* > >();
  this->ReadFile(path);
}

template<typename DTYPE> void RNNParalleledCorpusDataset<DTYPE>::BuildVocab() {
  MakeLineData();
  cout<<"<<<<<<<<<<<<<<<<  BuildVocab 호출 >>>>>>>>>>>>>>>>>>>>"<<endl;
  for(int i=0; i<m_pairedSentences->size(); i++){
    pair< string, string > eachPair = m_pairedSentences->at(i);
    vector<string> leftSentences = this->SplitBy(eachPair.first, ' ');
    vector<string> rightSentences = this->SplitBy(eachPair.second, ' ');
    if(leftSentences.size() > m_EncoderMaxSentenceLength) m_EncoderMaxSentenceLength = leftSentences.size();
    if(rightSentences.size() > m_DecoderMaxSentenceLength) m_DecoderMaxSentenceLength = rightSentences.size();
  }

  for(int i=0; i<m_pairedSentences->size(); i++){
    vector<string> temp_words = this->SplitBy(m_pairedSentences->at(i).first, ' ');
    vector<int> temp_first_indexed_words;
    for(string word: temp_words){
      //cout << word << endl;
      temp_first_indexed_words.push_back(this->GetpVocab2Index()->at(word));
    }
    while(temp_first_indexed_words.size() > this->GetEncoderMaxTime()){                 //오류 ????  <
      temp_first_indexed_words.push_back(this->GetpVocab2Index()->at("<PAD>"));
    }
    temp_words = this->SplitBy(m_pairedSentences->at(i).second, ' ');
    vector<int> temp_second_indexed_words;
    for(string word: temp_words){
      temp_second_indexed_words.push_back(this->GetpVocab2Index()->at(word));
    }
    while(temp_second_indexed_words.size() > this->GetDecoderMaxTime()){                // 오류 ???????     일단 여기 오류는 맞고...! 이 오류가 발견이 안된 이유는...
      //std::cout<<"size : "<<temp_second_indexed_words.size()<<'\n';                     // 아래 6줄 내려가면 std::copy에서 복사해주고
      temp_second_indexed_words.push_back(this->GetpVocab2Index()->at("<PAD>"));        //그전에 new int로 배열을 amxTime으로 만들어줘서 된거임!!!
    }
    int* left = new int[this->GetEncoderMaxTime()]{0, };
    int* right = new int[this->GetDecoderMaxTime()]{0, };
    std::copy(temp_first_indexed_words.begin(), temp_first_indexed_words.end(), left);
    std::copy(temp_second_indexed_words.begin(), temp_second_indexed_words.end(), right);

    m_pairedIndexedSentences->push_back(make_pair(left, right));
  }
  m_pairedIndexedSentences->shrink_to_fit();
  m_EncoderMaxSentenceLength++; //EOS, SOS를 위해
  m_DecoderMaxSentenceLength++;
}

template<typename DTYPE> void RNNParalleledCorpusDataset<DTYPE>::MakeLineData() { // 확인완료

    cout<<"<<<<<<<<<<<<<<<<  MakeLineData  >>>>>>>>>>>>>>>>>>>>"<<endl;
    //cout<<strlen(TextData)<<endl;
    char* token = strtok(this->GetTextData(), "\t\n");
    char* last_sentence = NULL;

    while(token != NULL) {
      //cout<<token<<endl;              //DEBUG
      if(this->GetLineLength()%2==0){
        last_sentence = token;                                              // paired data를 만들기위해 앞에 오는 line 임시 저장
      }
      else {
        string str_last_sentence = this->Preprocess(last_sentence);
        string str_token = this->Preprocess(token);
        m_pairedSentences->push_back(make_pair(str_last_sentence, str_token));           // paired data 저장
        this->AddSentence(this->Preprocess(m_pairedSentences->back().first));
        this->AddSentence(this->Preprocess(m_pairedSentences->back().second));
      }
      //temp->line->push_back(token);                                         // 각 언어에 line 저장
      //MakeVocab(token);
      token = strtok(NULL, "\t\n");
      int temp_lineLength = this->GetLineLength();
      if(temp_lineLength%10000==0)
        cout<<"line_length = "<<temp_lineLength<<endl;

      this->SetLineLength(++temp_lineLength);
    }
    m_pairedSentences->shrink_to_fit();
    //text_lines /=2;
  }

  template<typename DTYPE> vector< pair< int*, int* > >* RNNParalleledCorpusDataset<DTYPE>::GetPairedIndexedSentences(){
    return m_pairedIndexedSentences;
  }

  template<typename DTYPE> int RNNParalleledCorpusDataset<DTYPE>::GetEncoderMaxTime(){
    return m_EncoderMaxSentenceLength;
  }

  template<typename DTYPE> int RNNParalleledCorpusDataset<DTYPE>::GetDecoderMaxTime(){
    return m_DecoderMaxSentenceLength;
  }

template<typename DTYPE> int RNNParalleledCorpusDataset<DTYPE>::GetLength(){
  return (int)(this->GetLineLength()/2);                                                    //여기 찬효랑 이야기 해야됨! 이렇게 하기로 !!! 짝이 되어 있어서                                                                                //즉 한줄에 문장이 2개 있어서!!!
}


/*
  template<typename DTYPE> std::vector<Tensor<DTYPE> *>* RNNParalleledCorpusDataset<DTYPE>:: GetData(int idx){
    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);


    //   vector< pair< int*, int* > >* m_pairedIndexedSentences;

    //encoder maxtime
    //Decoder maxtime 이거 2개를... 음.....
    //SOS, EOS, PAD 이거를... 어디서 처리해줄것인가
    //PAD 0 SOS 1 EOS 2
    Tensor<DTYPE> *EncoderInput = Tensor<DTYPE>::Zeros(m_EncoderMaxSentenceLength, 1, 1, 1, 1);
    Tensor<DTYPE> *DecoderInput = Tensor<DTYPE>::Zeros(m_DecoderMaxSentenceLength, 1, 1, 1, 1);
    Tensor<DTYPE> *Label = Tensor<DTYPE>::Zeros(m_DecoderMaxSentenceLength, 1, 1, 1, this->GetNumberofVocabs());

    Shape *LabelShape = Label->GetShape();



    // pair<int*, int*> each_paired_indexed_sentences = m_pairedIndexedSentences->at(idx);
    // for(int i=0; i<m_EncoderMaxSentenceLength; i++){
    //     cout << (int)((each_paired_indexed_sentences.first)[i]) << ' ';
    //     //std::cout<<((m_pairedIndexedSentences->at(idx)).first)[i]<<'\n';
    // }

    //EncoderInput 생성
    (*EncoderInput)[0] = 1;      //SOS로 시작
    for (int i = 1; i < m_EncoderMaxSentenceLength; i++) {
        //std::cout<<((m_pairedIndexedSentences->at(idx)).first)[i-1]<<'\n';
        (*EncoderInput)[i] = ((m_pairedIndexedSentences->at(idx)).first)[i-1];
    }

    // std::cout<<"Encoder Input 생성 완료"<<'\n';\
    // std::cout<<EncoderInput<<'\n';

    //Decoder Input, label 생성
    (*DecoderInput)[0] = 1;    //SOS로 시작
    for(int i=0; i<m_DecoderMaxSentenceLength-1; i++){
        //input
        (*DecoderInput)[i+1] = (m_pairedIndexedSentences->at(idx).second)[i];

        //label
        (*Label)[Index5D(LabelShape, i, 0, 0, 0, (m_pairedIndexedSentences->at(idx).second)[i])] = 1;

    }

    // std::cout<<"Decoder Input 생성 완료"<<'\n';
    // std::cout<<DecoderInput<<'\n';

    (*Label)[Index5D(LabelShape, m_DecoderMaxSentenceLength-1, 0, 0, 0, this->GetNumberofVocabs()-1)] = 1;    //EOS 처리

    // for(int i=0; i<DecoderMaxTimeSize-1; i++){
    //     int index = (m_pairedIndexedSentences->at(idx).second)[i]);
    //     (*Label)[Index5D(LabelShape, i, 0, 0, 0, index)] = 1;
    // }

    // std::cout<<"Label Input 생성 완료"<<'\n';

    result->push_back(EncoderInput);
    result->push_back(DecoderInput);
    result->push_back(Label);

    return result;
  }
*/


  // 맨 처음 0에서만 작동하도록!     -> 이게 무슨말이지....
  //padding 처리를 위해 length정보 넘겨주도록 추가 구현!
  template<typename DTYPE> std::vector<Tensor<DTYPE> *>* RNNParalleledCorpusDataset<DTYPE>:: GetData(int idx){
    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

    // std::cout<<"RNNParalleledCorpusDataset GetData "<<idx<<'\n';
    // std::cout<<"encoder max : "<<m_EncoderMaxSentenceLength<<'\n';
    // std::cout<<"decoder max : "<<m_DecoderMaxSentenceLength<<'\n';
    // std::cout<<"Number of Vocab : "<<this->GetNumberofVocabs()<<'\n';
    // std::cout<<"sentence length : "<<this->GetLength()<<'\n';

    //   vector< pair< int*, int* > >* m_pairedIndexedSentences;

    //encoder maxtime
    //Decoder maxtime 이거 2개를... 음.....
    //SOS, EOS, PAD 이거를... 어디서 처리해줄것인가
    //PAD 0 SOS 1 EOS 2
    Tensor<DTYPE> *EncoderInput = Tensor<DTYPE>::Zeros(m_EncoderMaxSentenceLength, 1, 1, 1, 1);
    Tensor<DTYPE> *DecoderInput = Tensor<DTYPE>::Zeros(m_DecoderMaxSentenceLength, 1, 1, 1, 1);
    Tensor<DTYPE> *Label = Tensor<DTYPE>::Zeros(m_DecoderMaxSentenceLength, 1, 1, 1, this->GetNumberofVocabs());

    Tensor<DTYPE> *EncoderLength = Tensor<DTYPE>::Zeros(1, 1, 1, 1, 1);
    Tensor<DTYPE> *DecoderLength = Tensor<DTYPE>::Zeros(1, 1, 1, 1, 1);

    Shape *LabelShape = Label->GetShape();

    //EncoderInput 생성
    (*EncoderInput)[0] = 1;      //SOS로 시작
    for (int i = 1; i < m_EncoderMaxSentenceLength; i++) {
        //std::cout<<((m_pairedIndexedSentences->at(idx)).first)[i-1]<<'\n';
        (*EncoderInput)[i] = ((m_pairedIndexedSentences->at(idx)).first)[i-1];
        //std::cout<<(*EncoderInput)[i]<<" ";

        //encoder length 처리해주기!
        if( (*EncoderInput)[i] != 0)
          (*EncoderLength)[0] = i;
    }

    //std::cout<<'\n';

    //Decoder Input, label 생성
    (*DecoderInput)[0] = 1;    //SOS로 시작
    for(int i=0; i<m_DecoderMaxSentenceLength-1; i++){
        //input
        (*DecoderInput)[i+1] = (m_pairedIndexedSentences->at(idx).second)[i];


        //label
        //(*Label)[Index5D(LabelShape, i, 0, 0, 0, (m_pairedIndexedSentences->at(idx).second)[i])] = 1;


        //decoder length 처리
        if( (*DecoderInput)[i+1] != 0)
            (*DecoderLength)[0] = i+1;

    }

    // std::cout<<"Decoder Input 생성 완료"<<'\n';
    // std::cout<<DecoderInput<<'\n';

    int flag = 0;
    for(int i=0; i<m_DecoderMaxSentenceLength-1; i++){
        (*Label)[Index5D(LabelShape, i, 0, 0, 0, (m_pairedIndexedSentences->at(idx).second)[i])] = 1;

        if(i ==  (*DecoderLength)[0]){
            flag = 1;
            (*Label)[Index5D(LabelShape, i, 0, 0, 0, 2)] = 1;
            (*Label)[Index5D(LabelShape, i, 0, 0, 0, 0)] = 0;

        }

    }

    if(flag == 1)
      (*Label)[Index5D(LabelShape, m_DecoderMaxSentenceLength-1, 0, 0, 0, 0)] = 1;    //EOS 처리
    else
      (*Label)[Index5D(LabelShape, m_DecoderMaxSentenceLength-1, 0, 0, 0, 2)] = 1;

        //(*Label)[Index5D(LabelShape, m_DecoderMaxSentenceLength-1, 0, 0, 0, 2)] = 1;


    //SOS & EOS 때문에 Length 한개씩 추가!!!
    (*EncoderLength)[0] += 1;
    (*DecoderLength)[0] += 1;

    // std::cout<<"Label Input 생성 완료"<<'\n';

    result->push_back(EncoderInput);
    result->push_back(DecoderInput);
    result->push_back(Label);
    result->push_back(EncoderLength);
    result->push_back(DecoderLength);


    return result;
  }



  //seqLength * batchsize 가 word의 개수보다 많은 경우는?...

  template<typename DTYPE>
  class RNNWordLevelDataset : public TextDataset<DTYPE>{ //파일 경로 받아서 실제 보캡, Paired문장 등 보관
  private:

    vector< string >* m_WordSentences;          // paired data
    vector< int >* m_WordIndexedSentences;

    int seqLength;

    //int n_words;                                // 부모 class에 GetNumberofWords 존재!

  public:
    RNNWordLevelDataset(string path, int pSeqLength );

    void                                   Alloc(string path);

    void                                   MakeWordData();

    virtual void                           BuildVocab();

    virtual                                ~RNNWordLevelDataset();

    virtual int                            GetLength();

    virtual std::vector<Tensor<DTYPE>*>*   GetData(int idx);

  };


template<typename DTYPE> RNNWordLevelDataset<DTYPE>::RNNWordLevelDataset(string path, int pSeqLength) : TextDataset<DTYPE>::TextDataset() {
  m_WordSentences = NULL;
  m_WordIndexedSentences = NULL;
  seqLength = pSeqLength;
  // n_words = 0;
  Alloc(path);
}
template<typename DTYPE> RNNWordLevelDataset<DTYPE>::~RNNWordLevelDataset() {
    cout << "RNNParalleledCorpusDataset 소멸자 호출" << endl;
    delete m_WordSentences;
    delete m_WordIndexedSentences;
}


template<typename DTYPE> void RNNWordLevelDataset<DTYPE>::Alloc(string path) {
  m_WordSentences = new vector<string >();
  m_WordIndexedSentences = new vector< int >();
  this->ReadFile(path);
}

template<typename DTYPE> void RNNWordLevelDataset<DTYPE>::BuildVocab() {
  MakeWordData();
  cout<<"<<<<<<<<<<<<<<<<  BuildVocab 호출 >>>>>>>>>>>>>>>>>>>>"<<endl;

  for(int i=0; i<m_WordSentences->size(); i++){

      // while(temp_second_indexed_words.size() > this->GetDecoderMaxTime()){
      //   temp_second_indexed_words.push_back(this->GetpVocab2Index()->at("<PAD>"));
      // }
      //
      // int* right = new int[this->GetDecoderMaxTime()]{0, };
      // std::copy(temp_second_indexed_words.begin(), temp_second_indexed_words.end(), right);

      //단어 가져오기
      string word = m_WordSentences->at(i);

      //index로 만들어서 넣기!
      m_WordIndexedSentences->push_back(this->GetpVocab2Index()->at(word));
  }
  m_WordIndexedSentences->shrink_to_fit();

}

template<typename DTYPE> void RNNWordLevelDataset<DTYPE>::MakeWordData() { // 확인완료

    cout<<"<<<<<<<<<<<<<<<<  MakeWordData  >>>>>>>>>>>>>>>>>>>>"<<endl;
    //cout<<strlen(TextData)<<endl;
    char* token = strtok(this->GetTextData(), " ");       //splitBy 함수 사용?....      // \t : tap!

    //std::cout<<"???"<<'\n';

    while(token != NULL) {

        string str_token = this->Preprocess(token);
        //std::cout<<str_token<<" ";
        this->AddWord(str_token);

        m_WordSentences->push_back(str_token);

        token = strtok(NULL, " ");
    }
}

template<typename DTYPE> int RNNWordLevelDataset<DTYPE>::GetLength(){
   return m_WordIndexedSentences->size() / seqLength;                                 //drop reminder 괜춘한거 같음!!!
}


// 맨 처음 0에서만 작동하도록!     -> 이게 무슨말이지....
//padding 처리를 위해 length정보 넘겨주도록 추가 구현!
template<typename DTYPE> std::vector<Tensor<DTYPE> *>* RNNWordLevelDataset<DTYPE>:: GetData(int idx){

    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

    //int numOfwords = this->GetNumberofWords();                  //실제 vector에 있는거랑 EOS, PAD, EOS때문에 개수에서 3개 차이남!!! 그래서 vector의 실제 사이즈로!!!
    int numOfwords = m_WordIndexedSentences->size();
    int numOfVocabs = this->GetNumberofVocabs();

    // std::cout<<"---------------------------------RNNWordLevelDataset GetData "<<idx<<'\n';
    // std::cout<<"Number of Vocab : "<<numOfVocabs<<'\n';
    // std::cout<<"numOfwords : "<<numOfwords<<'\n';
    // std::cout<<"실제 vector의 size : "<<m_WordIndexedSentences->size()<<'\n';

    Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(seqLength, 1, 1, 1, 1);                        //입력은 index
    Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(seqLength, 1, 1, 1, numOfVocabs);             //label은 onthot!


    //input 생성!
    int start = idx*seqLength;
    for(int i=0; i < seqLength; i++){
        (*input)[i] = m_WordIndexedSentences->at(start + i);
    }

    //label 생성!
    int colIndex = 0;
    for(int i=0; i < seqLength; i++){

        if((start + i + 1) == numOfwords)
          colIndex = 2;                             //마지막은 EOS
        else
          colIndex = m_WordIndexedSentences->at(start + i+1);

        (*label)[Index5D(label->GetShape(), i, 0, 0, 0, colIndex)] = 1;
    }

    //std::cout<<"Label Input 생성 완료"<<'\n';

    result->push_back(input);
    result->push_back(label);



    return result;
}
