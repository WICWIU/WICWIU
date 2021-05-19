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

template<typename DTYPE> class TextDataset : public Dataset<DTYPE>{
private:
  string path;
  char* TextData;
  int text_length;
  int line_length;
  int max_sentence_length;
  //-----Field 클래스에서 차용------//

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

  int m_numOfInput;             //input data의 개수!!!
  int m_window;                 //window size -> 홀수가 기본이겠지!
  int m_negative;

  int m_dimOfInput;
  int m_dimOfLabel;

public:
  TextDataset();

  void                         ReadFile(string path);

  void                         Pad();

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

  virtual int                           GetLength();

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
  return n_vocabs;
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
  delete[] TextData;
  delete m_pIndex2Vocab;
  delete m_pVocab2Frequency;
  delete m_pVocab2Index;
}


template<typename DTYPE> void TextDataset<DTYPE>::ReadFile(string path) {
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
}



template<typename DTYPE>
class ParalleledCorpusDataset : public TextDataset<DTYPE>{
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
  for(int i=0; i<m_pairedSentences->size(); i++){
    vector<string> temp_words = this->SplitBy(m_pairedSentences->at(i).first, ' ');
    vector<int> temp_first_indexed_words;
    for(string word: temp_words){
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

template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::MakeLineData() {

    char* token = strtok(this->GetTextData(), "\t\n");
    char* last_sentence = NULL;

    while(token != NULL) {
      if(this->GetLineLength()%2==0){
        last_sentence = token;
      }
      else {
        string str_last_sentence = this->Preprocess(last_sentence);
        string str_token = this->Preprocess(token);
        m_pairedSentences->push_back(make_pair(str_last_sentence, str_token));
        this->AddSentence(this->Preprocess(m_pairedSentences->back().first));
        this->AddSentence(this->Preprocess(m_pairedSentences->back().second));
      }
      token = strtok(NULL, "\t\n");
      int temp_lineLength = this->GetLineLength();
      if(temp_lineLength%10000==0)
        cout<<"line_length = "<<temp_lineLength<<endl;

      this->SetLineLength(++temp_lineLength);
    }
    m_pairedSentences->shrink_to_fit();
  }

  template<typename DTYPE> vector< pair< int*, int* > >* ParalleledCorpusDataset<DTYPE>::GetPairedIndexedSentences(){
    return m_pairedIndexedSentences;
  }


  template<typename DTYPE>
  class RNNParalleledCorpusDataset : public TextDataset<DTYPE>{
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
      while(temp_first_indexed_words.size() > this->GetEncoderMaxTime()){
        temp_first_indexed_words.push_back(this->GetpVocab2Index()->at("<PAD>"));
      }
      temp_words = this->SplitBy(m_pairedSentences->at(i).second, ' ');
      vector<int> temp_second_indexed_words;
      for(string word: temp_words){
        temp_second_indexed_words.push_back(this->GetpVocab2Index()->at(word));
      }
      while(temp_second_indexed_words.size() > this->GetDecoderMaxTime()){
        //std::cout<<"size : "<<temp_second_indexed_words.size()<<'\n';
        temp_second_indexed_words.push_back(this->GetpVocab2Index()->at("<PAD>"));
      }
      int* left = new int[this->GetEncoderMaxTime()]{0, };
      int* right = new int[this->GetDecoderMaxTime()]{0, };
      std::copy(temp_first_indexed_words.begin(), temp_first_indexed_words.end(), left);
      std::copy(temp_second_indexed_words.begin(), temp_second_indexed_words.end(), right);

      m_pairedIndexedSentences->push_back(make_pair(left, right));
    }
    m_pairedIndexedSentences->shrink_to_fit();
    m_EncoderMaxSentenceLength++;
    m_DecoderMaxSentenceLength++;
  }

  template<typename DTYPE> void RNNParalleledCorpusDataset<DTYPE>::MakeLineData() {

      cout<<"<<<<<<<<<<<<<<<<  MakeLineData  >>>>>>>>>>>>>>>>>>>>"<<endl;
      //cout<<strlen(TextData)<<endl;
      char* token = strtok(this->GetTextData(), "\t\n");
      char* last_sentence = NULL;

      while(token != NULL) {
        //cout<<token<<endl;              //DEBUG
        if(this->GetLineLength()%2==0){
          last_sentence = token;
        }
        else {
          string str_last_sentence = this->Preprocess(last_sentence);
          string str_token = this->Preprocess(token);
          m_pairedSentences->push_back(make_pair(str_last_sentence, str_token));
          this->AddSentence(this->Preprocess(m_pairedSentences->back().first));
          this->AddSentence(this->Preprocess(m_pairedSentences->back().second));
        }

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
    return (int)(this->GetLineLength()/2);                                                                                                                                //즉 한줄에 문장이 2개 있어서!!!
  }

    template<typename DTYPE> std::vector<Tensor<DTYPE> *>* RNNParalleledCorpusDataset<DTYPE>:: GetData(int idx){
      std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

      Tensor<DTYPE> *EncoderInput = Tensor<DTYPE>::Zeros(m_EncoderMaxSentenceLength, 1, 1, 1, 1);
      Tensor<DTYPE> *DecoderInput = Tensor<DTYPE>::Zeros(m_DecoderMaxSentenceLength, 1, 1, 1, 1);
      Tensor<DTYPE> *Label = Tensor<DTYPE>::Zeros(m_DecoderMaxSentenceLength, 1, 1, 1, this->GetNumberofVocabs());

      Tensor<DTYPE> *EncoderLength = Tensor<DTYPE>::Zeros(1, 1, 1, 1, 1);
      Tensor<DTYPE> *DecoderLength = Tensor<DTYPE>::Zeros(1, 1, 1, 1, 1);

      Shape *LabelShape = Label->GetShape();

      (*EncoderInput)[0] = 1;
      for (int i = 1; i < m_EncoderMaxSentenceLength; i++) {
          (*EncoderInput)[i] = ((m_pairedIndexedSentences->at(idx)).first)[i-1];

          if( (*EncoderInput)[i] != 0)
            (*EncoderLength)[0] = i;
      }

      (*DecoderInput)[0] = 1;
      for(int i=0; i<m_DecoderMaxSentenceLength-1; i++){
          (*DecoderInput)[i+1] = (m_pairedIndexedSentences->at(idx).second)[i];

          if( (*DecoderInput)[i+1] != 0)
              (*DecoderLength)[0] = i+1;

      }

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
        (*Label)[Index5D(LabelShape, m_DecoderMaxSentenceLength-1, 0, 0, 0, 0)] = 1;
      else
        (*Label)[Index5D(LabelShape, m_DecoderMaxSentenceLength-1, 0, 0, 0, 2)] = 1;

      (*EncoderLength)[0] += 1;
      (*DecoderLength)[0] += 1;


      result->push_back(EncoderInput);
      result->push_back(DecoderInput);
      result->push_back(Label);
      result->push_back(EncoderLength);
      result->push_back(DecoderLength);


      return result;
    }


  template<typename DTYPE>
  class RNNWordLevelDataset : public TextDataset<DTYPE>{
  private:

    vector< string >* m_WordSentences;
    vector< int >* m_WordIndexedSentences;

    int seqLength;

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

  for(int i=0; i<m_WordSentences->size(); i++){

      string word = m_WordSentences->at(i);

      m_WordIndexedSentences->push_back(this->GetpVocab2Index()->at(word));
  }
  m_WordIndexedSentences->shrink_to_fit();

}

template<typename DTYPE> void RNNWordLevelDataset<DTYPE>::MakeWordData() {


    char* token = strtok(this->GetTextData(), " ");

    while(token != NULL) {

        string str_token = this->Preprocess(token);
        this->AddWord(str_token);

        m_WordSentences->push_back(str_token);

        token = strtok(NULL, " ");
    }
}

template<typename DTYPE> int RNNWordLevelDataset<DTYPE>::GetLength(){
   return m_WordIndexedSentences->size() / seqLength;
}

template<typename DTYPE> std::vector<Tensor<DTYPE> *>* RNNWordLevelDataset<DTYPE>:: GetData(int idx){

    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

    int numOfwords = m_WordIndexedSentences->size();
    int numOfVocabs = this->GetNumberofVocabs();

    Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(seqLength, 1, 1, 1, 1);
    Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(seqLength, 1, 1, 1, numOfVocabs);

    int start = idx*seqLength;
    for(int i=0; i < seqLength; i++){
        (*input)[i] = m_WordIndexedSentences->at(start + i);
    }

    int colIndex = 0;
    for(int i=0; i < seqLength; i++){

        if((start + i + 1) == numOfwords)
          colIndex = 2;
        else
          colIndex = m_WordIndexedSentences->at(start + i+1);

        (*label)[Index5D(label->GetShape(), i, 0, 0, 0, colIndex)] = 1;
    }

    result->push_back(input);
    result->push_back(label);

    return result;
}
