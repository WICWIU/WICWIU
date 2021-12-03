#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

// 사용하기 위해서는 Module의 analyzeGraph에서 Query 관련 주석을 풀어야합니다.
// 또한 RNNEncoder에서 Bidirectional layer를 사용해야 합니다.
class my_AttentionSeqToSeq : public NeuralNetwork<float> {
private:
public:
	my_AttentionSeqToSeq(Tensorholder<float>* EncoderInput, Tensorholder<float>* DecoderInput, Tensorholder<float>* label, Tensorholder<float>* EncoderLengths, Tensorholder<float>* DecoderLengths, int vocabLength, int embeddingDim, int hiddenDim) {
		SetInput(5, EncoderInput, DecoderInput, label, EncoderLengths, DecoderLengths);

		Operator<float>* out = NULL;

		Operator<float>* mask = new AttentionPaddingMaskRNN<float>(EncoderInput, 0, "srcMasking");    

		// ======================= layer 1=======================
		out = new RNNEncoder<float>(EncoderInput, vocabLength, embeddingDim, hiddenDim, TRUE, "Encoder");

		out = new Bahdanau<float>(DecoderInput, out, mask, vocabLength, embeddingDim, hiddenDim, vocabLength, EncoderLengths, TRUE, "Bahdanau_Decoder");

		AnalyzeGraph(out);

		// ======================= Select LossFunction Function ===================
		// SetLossFunction(new HingeLoss<float>(out, label, "HL"));
		// SetLossFunction(new MSE<float>(out, label, "MSE"));
		//SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
		SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE", DecoderLengths));
		// SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

		// ======================= Select Optimizer ===================
		// SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, 0.9, 1.0, MINIMIZE));                   
		SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.001, 0.9, 1e-08, FALSE, MINIMIZE));
		// SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
		// SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
		// SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));     

	}

	virtual ~my_AttentionSeqToSeq() {}

	int seq2seqBPTT(int EncTimeSize, int DecTimeSize);
	int seq2seqBPTTOnGPU(int EncTimeSize, int DecTimeSize);
	std::string SentenceTranslateOnCPU(std::map<int, std::string>* index2vocab, std::string filename);
	std::string SentenceTranslateOnGPU(std::map<int, std::string>* index2vocab, std::string filename);
};

int my_AttentionSeqToSeq::seq2seqBPTT(int EncTimeSize, int DecTimeSize) {

	LossFunction<float>* m_aLossFunction = this->GetLossFunction();
	Optimizer<float>* m_aOptimizer = this->GetOptimizer();

	this->ResetResult();
	this->ResetGradient();
	this->ResetLossFunctionResult();
	this->ResetLossFunctionGradient();

	Container<Operator<float>*>* ExcutableOperator = this->GetExcutableOperatorContainer();

	//maks forward
	(*ExcutableOperator)[0]->ForwardPropagate(0);

	//encoder forward
	for (int i = 0; i < EncTimeSize; i++)
		(*ExcutableOperator)[1]->ForwardPropagate(i);

	//Decoder & lossfunction forward
	for (int i = 0; i < DecTimeSize; i++) {
		(*ExcutableOperator)[2]->ForwardPropagate(i);
		m_aLossFunction->ForwardPropagate(i);
	}

	//Decoder & loss function backward
	for (int j = DecTimeSize - 1; j >= 0; j--) {
		m_aLossFunction->BackPropagate(j);
		(*ExcutableOperator)[2]->BackPropagate(j);
	}

	//Encoder backward
	for (int j = EncTimeSize - 1; j >= 0; j--) {
		(*ExcutableOperator)[1]->BackPropagate(j);
	}

	m_aOptimizer->UpdateParameter();

	return TRUE;
}

int my_AttentionSeqToSeq::seq2seqBPTTOnGPU(int EncTimeSize, int DecTimeSize) {
#ifdef __CUDNN__
	this->ResetResult();
	this->ResetGradient();
	this->ResetLossFunctionResult();
	this->ResetLossFunctionGradient();

	LossFunction<float>* m_aLossFunction = this->GetLossFunction();
	Optimizer<float>* m_aOptimizer = this->GetOptimizer();

	Container<Operator<float>*>* ExcutableOperator = this->GetExcutableOperatorContainer();

	//mask forward
	(*ExcutableOperator)[0]->ForwardPropagateOnGPU(0);

	//encoder forward
	for (int i = 0; i < EncTimeSize; i++)
		(*ExcutableOperator)[1]->ForwardPropagateOnGPU(i);

	//Decoder & lossfunction forward
	for (int i = 0; i < DecTimeSize; i++) {
		(*ExcutableOperator)[2]->ForwardPropagateOnGPU(i);
		m_aLossFunction->ForwardPropagateOnGPU(i);
	}

	//Decoder & loss function backward
	for (int j = DecTimeSize - 1; j >= 0; j--) {
		m_aLossFunction->BackPropagateOnGPU(j);
		(*ExcutableOperator)[2]->BackPropagateOnGPU(j);
	}

	//Encoder backward
	for (int j = EncTimeSize - 1; j >= 0; j--) {
		(*ExcutableOperator)[1]->BackPropagateOnGPU(j);
	}

	m_aOptimizer->UpdateParameterOnGPU();

#else
	std::cout << "There is no GPU option!" << '\n';
	exit(-1);
#endif

	return TRUE;
}


std::string my_AttentionSeqToSeq::SentenceTranslateOnCPU(std::map<int, std::string>* index2vocab, std::string filename) {

	this->ResetResult();
	this->ResetLossFunctionResult();

	std::ofstream fout(filename, std::ios::app);
	fout << '\n';

	std::string result_str = "";

	//Result
	Tensor<float>* pred = this->GetResult();

	//DecoderInput
	Tensor<float>* DecoderInput = this->GetInput()[1]->GetResult();
	Shape* InputShape = DecoderInput->GetShape();

	//encoder, decoder time size
	int EncoderTimeSize = this->GetInput()[0]->GetResult()->GetTimeSize();
	int DecoderTimeSize = DecoderInput->GetTimeSize();

	//Encoder, Decoder module access
	int numOfExcutableOperator = this->GetNumOfExcutableOperator();
	Container<Operator<float>*>* ExcutableOperator = this->GetExcutableOperatorContainer();

	//maks forward
	(*ExcutableOperator)[0]->ForwardPropagate(0);

	//encoder forward
	for (int ti = 0; ti < EncoderTimeSize; ti++)
		(*ExcutableOperator)[1]->ForwardPropagate(ti);

	(*DecoderInput)[0] = 1;

	for (int ti = 0; ti < DecoderTimeSize; ti++) {

		//decoder forward
		(*ExcutableOperator)[2]->ForwardPropagate(ti);

		int pred_index = this->GetMaxIndex(pred, 0, ti, pred->GetColSize());

		// std::cout << pred_index << " : ";
		// std::cout << index2vocab->at(pred_index) << '\n';
		result_str = result_str + ' ' + index2vocab->at(pred_index);

		// if (fout.is_open()) {
		// 	fout << pred_index << " : ";
		// 	fout << index2vocab->at(pred_index) << '\n';
		// }

		if (pred_index == 2)
			break;

		if (ti != DecoderTimeSize - 1) {
			(*DecoderInput)[Index5D(InputShape, ti + 1, 0, 0, 0, 0)] = pred_index;
		}
	}

	this->ResetResult();

	return result_str;

}

std::string my_AttentionSeqToSeq::SentenceTranslateOnGPU(std::map<int, std::string>* index2vocab, std::string filename) {
#ifdef __CUDNN__

	this->ResetResult();
	this->ResetLossFunctionResult();

	std::ofstream fout(filename, std::ios::app);
	fout << '\n';

	std::string result_str = "";

	//Result
	Tensor<float>* pred = this->GetResult();

	//DecoderInput
	Tensor<float>* DecoderInput = this->GetInput()[1]->GetResult();
	Shape* InputShape = DecoderInput->GetShape();


	//encoder, decoder time size
	int EncoderTimeSize = this->GetInput()[0]->GetResult()->GetTimeSize();
	int DecoderTimeSize = DecoderInput->GetTimeSize();

	//Encoder, Decoder module access
	int numOfExcutableOperator = this->GetNumOfExcutableOperator();
	Container<Operator<float>*>* ExcutableOperator = this->GetExcutableOperatorContainer();

	//maks forward
	(*ExcutableOperator)[0]->ForwardPropagateOnGPU(0);

	//encoder forward
	for (int ti = 0; ti < EncoderTimeSize; ti++)
		(*ExcutableOperator)[1]->ForwardPropagateOnGPU(ti);

	(*DecoderInput)[0] = 1;

	for (int ti = 0; ti < DecoderTimeSize; ti++) {

		//decoder forward
		(*ExcutableOperator)[2]->ForwardPropagateOnGPU(ti);

		int pred_index = this->GetMaxIndex(pred, 0, ti, pred->GetColSize());

		// // print
		// std::cout << pred_index << " : ";
		// std::cout << index2vocab->at(pred_index) << '\n';
		result_str = result_str + ' ' + index2vocab->at(pred_index);

		if (fout.is_open()) {
			fout << pred_index << " : ";
			fout << index2vocab->at(pred_index) << '\n';
		}

		// if EOS End
		if (pred_index == 2)
			break;

		if (ti != DecoderTimeSize - 1) {
			(*DecoderInput)[Index5D(InputShape, ti + 1, 0, 0, 0, 0)] = pred_index;
		}
	}

	fout.close();
	this->ResetResult();

#else  // __CUDNN__
	std::cout << "There is no GPU option!" << '\n';
	exit(-1);
#endif  // __CUDNN__

	return result_str;
}
