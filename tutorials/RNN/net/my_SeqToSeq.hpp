#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"


class my_SeqToSeq : public NeuralNetwork<float>{
private:
public:
    my_SeqToSeq(Tensorholder<float> *input1, Tensorholder<float> *input2, Tensorholder<float> *label, int vocab_length) {
        SetInput(2, input1, input2, label);

        Operator<float> *out = NULL;

        //out = new CBOW<float>(x(입력 배열), 아웃풋크기, "CBOW");
        //out = new OnehotVector<float>(x(입력 배열), 아웃풋크기, "OnehotVector");


        // ======================= layer 1=======================
        out = new Encoder<float>(input1, vocab_length, 32, TRUE, "Encoder");

        out = new Decoder<float>(input2, out, vocab_length, 32, vocab_length, TRUE, "Decoder");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

        // ======================= Select Optimizer ===================
        // 1.0이 clipValue 값! 인자 하나가 더 생김!
        //현재 RMSprop clip값 = 0.5로 되어있음
        //SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, 0.9, 1.0, MINIMIZE));                      // Optimizer의 첫번째 인자로 parameter목록을 전달해주는거고!!!   즉 updateparameter를 할 때 넘겨주는 parameter에 대해서만 함!!!!!
        SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.01, 0.9, 1e-08, FALSE, MINIMIZE));
        //SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
        // SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        //SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));      //MAXIMIZE
    }

    virtual ~my_SeqToSeq() {}
};
