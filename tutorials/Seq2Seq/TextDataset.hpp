#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../WICWIU_src/DataLoader.hpp"
#include "../../WICWIU_src/Dataset.hpp"
#include "../../WICWIU_src/Operator.hpp"
#include "../../WICWIU_src/Tensor.hpp"

using namespace std;

vector<string> SplitBy(string input, char delimiter) {
    vector<string> answer;
    stringstream   ss(input);
    string         temp;

    while (getline(ss, temp, delimiter)) {
        answer.push_back(temp);
    }
    return answer;
}

string Remove(string str, string delimiters) {
    vector<string> splited_delimiters;
    for (int i = 0; i < delimiters.length(); i++) {
        splited_delimiters.push_back(delimiters.substr(i, 1));
    }
    for (string delimiter : splited_delimiters) {
        int k = str.find(delimiter);
        while (k >= 0) {
            string k_afterStr = str.substr(k + 1, str.length() - k);
            str               = str.erase(k) + k_afterStr;
            k                 = str.find(delimiter);
        }
    }
    return str;
}

string Preprocess(string sentence, bool lower = true) {
    if (lower) {
        transform(sentence.begin(), sentence.end(), sentence.begin(), [](unsigned char c) { return std::tolower(c); });
    }
    // sentence = RemoveQuotation(sentence);
    sentence = Remove(sentence, "#;,.?!\"\'><:-\n\r");
    return sentence;
}

string RemoveNewLine(string s) {
    string::size_type i = 0;
    while (i < s.length()) {
        i = s.find('\n', i);
        if (i == string::npos) break;
        s.erase(i);
    }
    return s;
}

string RemoveQuotation(string s) {
    int k = s.find("\'");
    while (k >= 0) {
        string k_afterString = s.substr(k + 1, s.length() - k);
        s                    = s.erase(k) + " " + k_afterString;
        k                    = s.find("\'");
    }
    return s;
}


class Vocabulary {
private:
    map<int, string> *m_pIndex2Word;                
    map<string, int> *m_pWord2Frequency;
    map<string, int> *m_pWord2Index;
    string            languageName;
    int               numOfUniqueWords;
    int               maxSentenceLength;

public:

    Vocabulary(string name) {
        m_pIndex2Word     = new map<int, string>();
        m_pWord2Frequency = new map<string, int>();
        m_pWord2Index     = new map<string, int>();
        languageName      = name;
        numOfUniqueWords  = 0;
        maxSentenceLength = 0;

        AddWord("<PAD>");
        AddWord("<SOS>");
        AddWord("<EOS>");
        AddWord("<UNK>");
    }

    void AddSentence(string sentence) {
        vector<string> words = SplitBy(sentence, ' ');
        maxSentenceLength = max(maxSentenceLength, (int)words.size());              
        for (string word : words) {
            AddWord(word);
        }
    }

    void AddWord(string word) {
        if (m_pWord2Index->find(word) == m_pWord2Index->end()) {
            m_pWord2Index->insert(make_pair(word, numOfUniqueWords));
            m_pWord2Frequency->insert(make_pair(word, 1));
            m_pIndex2Word->insert(make_pair(numOfUniqueWords, word));
            numOfUniqueWords += 1;
        }
        else {
            m_pWord2Frequency->at(word) += 1;
        }
    }

    ~Vocabulary() {
        delete m_pIndex2Word;
        delete m_pWord2Frequency;
        delete m_pWord2Index;
    }

    map<int, string> *GetIndex2Word() {
        return m_pIndex2Word;
    }

    map<string, int> *GetWord2Frequency() {
        return m_pWord2Frequency;
    }

    map<string, int> *GetWord2Index() {
        return m_pWord2Index;
    }

    string GetLanguageName() {
        return languageName;
    }

    int GetNumberofUniqueWords() {
        return numOfUniqueWords;
    }

    int GetMaxSentenceLength() {
        return maxSentenceLength;
    }

    int FindWord(string s) {
        map<string, int>::iterator string2index = m_pWord2Index->find(s);
        if (string2index == m_pWord2Index->end()) {
            return m_pWord2Index->at("<UNK>");
        }
        return string2index->second;
    }

    void SetVocab(Vocabulary *trainVocab) {
        std::cout<<"Vocabulary class의 SetVocab 호출"<<'\n';
        m_pIndex2Word     = trainVocab->GetIndex2Word();
        m_pWord2Frequency = trainVocab->GetWord2Frequency();
        m_pWord2Index     = trainVocab->GetWord2Index();
        numOfUniqueWords  = trainVocab->GetNumberofUniqueWords();
        maxSentenceLength = trainVocab->GetMaxSentenceLength();
        std::cout<<"SetVocab maxlength : "<<maxSentenceLength<<'\n';
    }

};


template <typename DTYPE> class TextDataset : public Dataset<DTYPE> {
private:
    string path;
    int    numOfLines;
    int    maxSequenceLength;

public:

    TextDataset() {
        path       = "";
        numOfLines = 0;
    }

    virtual ~TextDataset() {}

    virtual void BuildVocab(Mode mode) {}
    virtual void SetVocabs(TextDataset *trainDataset) {}

    virtual int GetLength() {}
    virtual vector<Tensor<DTYPE> *>* GetData(int idx) {}

    string GetPath() {
        return path;
    }

    void SetPath(string path) {
        this->path = path;
    }

    int GetNumberOfLines() {
        return numOfLines;
    }

    void SetNumberOfLines(int n) {
        numOfLines = n;
    }

    int GetMaxSequenceLength() {
        return maxSequenceLength;
    }

    void SetMaxSequenceLength(int n) {
        maxSequenceLength = n;
    }
};


template <typename DTYPE> class ParalleledCorpusDataset : public TextDataset<DTYPE> {
private:
    pair<Vocabulary *, Vocabulary *> *m_pairedVocabulary;                           
    vector<pair<string, string>>     *m_pairedSentences;

public:
    ParalleledCorpusDataset(string path, string srcName, string tgtName) {
        this->SetPath(path);

        m_pairedVocabulary         = new pair<Vocabulary *, Vocabulary *>();
        m_pairedVocabulary->first  = new Vocabulary(srcName);
        m_pairedVocabulary->second = new Vocabulary(tgtName);
        m_pairedSentences          = new vector<pair<string, string>>();
    }

    virtual ~ParalleledCorpusDataset() {
        delete m_pairedSentences;
        delete m_pairedVocabulary->first;
        delete m_pairedVocabulary->second;
        delete m_pairedVocabulary;
    }

    Vocabulary *GetSrcVocabulary() {
        return m_pairedVocabulary->first;
    }

    Vocabulary *GetTgtVocabulary() {
        return m_pairedVocabulary->second;
    }

    virtual void BuildVocab(Mode mode) {
        string filepath = this->GetPath();

        ifstream fin(filepath);
        if (!fin.is_open()) {
            cout << "Cannot Open " + filepath << '\n';
            exit(1);
        }

        string line;
        while (getline(fin, line)){
            vector<string> pair_sentences = SplitBy(line, '\t');
            string src = Preprocess(pair_sentences[0], false);
            string tgt = Preprocess(pair_sentences[1]);
            src = src + " <EOS>";
            tgt = "<SOS> " + tgt + " <EOS>";
            m_pairedSentences->push_back(make_pair(src, tgt));

            if (mode == TRAIN) {
                m_pairedVocabulary->first->AddSentence(src);
                m_pairedVocabulary->second->AddSentence(tgt);
            }

            int tempMaxSequenceLength = max(m_pairedVocabulary->first->GetMaxSentenceLength(), m_pairedVocabulary->second->GetMaxSentenceLength());
            this->SetMaxSequenceLength(max(this->GetMaxSequenceLength(), tempMaxSequenceLength));

            int linelength = this->GetNumberOfLines();
            if ((linelength + 1) % 10000 == 0)
                cout << "line_length = " << linelength + 1 << endl;
            this->SetNumberOfLines(linelength + 1);
        }
    }

    virtual void SetVocabs(TextDataset<DTYPE>* trainDataset) {
        ParalleledCorpusDataset<DTYPE> *tempDataset = (ParalleledCorpusDataset<DTYPE> *)trainDataset;
        m_pairedVocabulary->first->SetVocab(tempDataset->m_pairedVocabulary->first);
        m_pairedVocabulary->second->SetVocab(tempDataset->m_pairedVocabulary->second);
    }

    virtual std::vector<Tensor<DTYPE> *> *GetData(int idx) {
        std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

        Vocabulary *srcVocab = m_pairedVocabulary->first;
        Vocabulary *tgtVocab = m_pairedVocabulary->second;

        int            maxLength    = this->GetMaxSequenceLength();
        Tensor<DTYPE> *encoderInput = Tensor<DTYPE>::Zeros(1, 1, 1, 1, maxLength + 1);
        Tensor<DTYPE> *decoderInput = Tensor<DTYPE>::Zeros(1, 1, 1, 1, maxLength + 1);
        Tensor<DTYPE> *label        = Tensor<DTYPE>::Zeros(1, 1, maxLength + 1, 1, tgtVocab->GetNumberofUniqueWords());

        Shape *labelShape = label->GetShape();

        string src = m_pairedSentences->at(idx).first;
        string tgt = m_pairedSentences->at(idx).second;

        vector<string> splitted_src = SplitBy(src, ' ');
        vector<string> splitted_tgt = SplitBy(tgt, ' ');

        for (int i = 0; i < splitted_src.size(); i++) {
            (*encoderInput)[i] = srcVocab->FindWord(splitted_src[i]);
        }

        for (int i = 0; i < splitted_tgt.size(); i++) {
            (*decoderInput)[i + 1] = tgtVocab->FindWord(splitted_tgt[i]);
        }

        for (int i = 0; i < splitted_tgt.size(); i++) {
            (*label)[Index5D(labelShape, 0, 0, i, 0, tgtVocab->FindWord(splitted_tgt[i]))] = 1;
        }

        result->push_back(encoderInput);
        result->push_back(decoderInput);
        result->push_back(label);

        return result;
    }

    virtual int GetLength() {
        return this->GetNumberOfLines();
    }

};


template <typename DTYPE> class RNNParalleledCorpusDataset : public TextDataset<DTYPE> {
private:
    pair<Vocabulary *, Vocabulary *> *m_pairedVocabulary;
    vector<pair<string, string>>     *m_pairedSentences;

    bool pretrained;
    map<string, float *> *m_pVocab2EmbeddingSrc;
    map<string, float *> *m_pVocab2EmbeddingTgt;

public:
    RNNParalleledCorpusDataset(string path, string srcName, string tgtName, bool pretrained = false) {
        this->SetPath(path);
        this->pretrained = pretrained;

        m_pairedVocabulary         = new pair<Vocabulary *, Vocabulary *>();
        m_pairedVocabulary->first  = new Vocabulary(srcName);
        m_pairedVocabulary->second = new Vocabulary(tgtName);
        m_pairedSentences          = new vector<pair<string, string>>();


        if (pretrained) {
            m_pVocab2EmbeddingSrc = new map<string, float *>();
            m_pVocab2EmbeddingTgt = new map<string, float *>();
            this->MakePretrainedEmbedding();
        }

    }

    Vocabulary *GetSrcVocabulary() {
        return m_pairedVocabulary->first;
    }

    Vocabulary *GetTgtVocabulary() {
        return m_pairedVocabulary->second;
    }

    virtual void BuildVocab(Mode mode) {
        string filepath = this->GetPath();

        ifstream fin(filepath);
        if (!fin.is_open()) {
            cout << "Cannot Open " + filepath << '\n';
            exit(1);
        }

        string line;
        while (getline(fin, line)){
            vector<string> pair_sentences = SplitBy(line, '\t');
            string src = Preprocess(pair_sentences[0], false);
            string tgt = Preprocess(pair_sentences[1]);
            src = src + " <EOS>";
            tgt = "<SOS> " + tgt + " <EOS>";
            m_pairedSentences->push_back(make_pair(src, tgt));

            if (mode == TRAIN) {
                m_pairedVocabulary->first->AddSentence(src);
                m_pairedVocabulary->second->AddSentence(tgt);
            }

            int tempMaxSequenceLength = max(m_pairedVocabulary->first->GetMaxSentenceLength(), m_pairedVocabulary->second->GetMaxSentenceLength());
            this->SetMaxSequenceLength(max(this->GetMaxSequenceLength(), tempMaxSequenceLength));

            //src max length

            //tgt max langth

            int linelength = this->GetNumberOfLines();
            if ((linelength + 1) % 10000 == 0)
                cout << "line_length = " << linelength << endl;
            this->SetNumberOfLines(linelength + 1);
        }
    }

    virtual void SetVocabs(TextDataset<DTYPE>* trainDataset) {
        std::cout<<" ----------- SetVocabs 호출"<<'\n';
        RNNParalleledCorpusDataset<DTYPE> *tempDataset = (RNNParalleledCorpusDataset<DTYPE> *)trainDataset;
        m_pairedVocabulary->first->SetVocab(tempDataset->m_pairedVocabulary->first);
        m_pairedVocabulary->second->SetVocab(tempDataset->m_pairedVocabulary->second);
    }


    virtual ~RNNParalleledCorpusDataset() {
        delete m_pairedSentences;
        if (pretrained) {
            delete m_pVocab2EmbeddingSrc;
            delete m_pVocab2EmbeddingTgt;
        }
    }

    virtual int GetLength() {
        return this->GetNumberOfLines();
    }

    virtual std::vector<Tensor<DTYPE> *> *GetData(int idx) {
        std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);          

        Vocabulary *srcVocab = m_pairedVocabulary->first;
        Vocabulary *tgtVocab = m_pairedVocabulary->second;

        Tensor<DTYPE> *encoderInput = Tensor<DTYPE>::Zeros(srcVocab->GetMaxSentenceLength(), 1, 1, 1, 1);
        Tensor<DTYPE> *decoderInput = Tensor<DTYPE>::Zeros(tgtVocab->GetMaxSentenceLength(), 1, 1, 1, 1);
        Tensor<DTYPE> *label        = Tensor<DTYPE>::Zeros(tgtVocab->GetMaxSentenceLength(), 1, 1, 1, tgtVocab->GetNumberofUniqueWords());

        Tensor<DTYPE> *encoderLength = Tensor<DTYPE>::Zeros(1, 1, 1, 1, 1);
        Tensor<DTYPE> *decoderLength = Tensor<DTYPE>::Zeros(1, 1, 1, 1, 1);

        Shape *labelShape = label->GetShape();

        string src = m_pairedSentences->at(idx).first;
        string tgt = m_pairedSentences->at(idx).second;

        vector<string> splitted_src = SplitBy(src, ' ');
        vector<string> splitted_tgt = SplitBy(tgt, ' ');

        for (int i = 0; i < splitted_src.size(); i++) {
            (*encoderInput)[i] = srcVocab->FindWord(splitted_src[i]);
        }

        for (int i = 0; i < splitted_tgt.size(); i++) {
            (*decoderInput)[i] = tgtVocab->FindWord(splitted_tgt[i]);  
        }

        int maxLength = tgtVocab->GetMaxSentenceLength();
        for (int i = 0; i < maxLength; i++) {
            if (i < splitted_tgt.size() - 1)
                (*label)[Index5D(labelShape, 0, 0, i, 0, tgtVocab->FindWord(splitted_tgt[i+1]))] = 1;
            else
                (*label)[Index5D(labelShape, 0, 0, i, 0, tgtVocab->FindWord("<PAD>"))] = 1;
        }

        (*encoderLength)[0] = (int)splitted_src.size();
        (*decoderLength)[0] = (int)splitted_tgt.size();

        result->push_back(encoderInput);
        result->push_back(decoderInput);
        result->push_back(label);
        result->push_back(encoderLength);
        result->push_back(decoderLength);

        return result;
    }

    void MakePretrainedEmbedding() {

        ifstream fin("wiki.multi.fr.vec", ios::in);
        ifstream fin_en("wiki.multi.en.vec", ios::in);
        string buffer;

        getline(fin, buffer);

        while (fin.peek() != EOF) {
            getline(fin, buffer);

            vector<string> wordAndEmbedding = SplitBy(buffer, ' ');
            vector<float> embeddingVector;
            for (int i = 1; i < 301; i++) {
                embeddingVector.push_back(std::stof(wordAndEmbedding[i]));
            }

            float *embedding = new float[300]{
                0,
            };
            std::copy(embeddingVector.begin(), embeddingVector.end(), embedding);

            m_pVocab2EmbeddingTgt->insert(make_pair(wordAndEmbedding[0], embedding));
        }

        getline(fin_en, buffer);

        while (fin_en.peek() != EOF) {
            getline(fin_en, buffer);

            vector<string> wordAndEmbeddingTgt = SplitBy(buffer, ' ');
            vector<float> embeddingVectorTgt;
            for (int i = 1; i < 301; i++) {
                embeddingVectorTgt.push_back(std::stof(wordAndEmbeddingTgt[i]));
            }

            float *embeddingTgt = new float[300]{
                0,
            };
            std::copy(embeddingVectorTgt.begin(), embeddingVectorTgt.end(), embeddingTgt);

            m_pVocab2EmbeddingSrc->insert(make_pair(wordAndEmbeddingTgt[0], embeddingTgt));
        }

    }

    map<string, float *> *GetVocab2EmbeddingSrc() {
        return m_pVocab2EmbeddingSrc;
    }

    map<string, float *> *GetVocab2EmbeddingTgt() {
        return m_pVocab2EmbeddingTgt;
    }
};
