#ifndef __BATCH_NORMALIZE__
#define __BATCH_NORMALIZE__ value

#include "../Operator.h"

template<typename DTYPE>
class BatchNormalize : public Operator<DTYPE>{
public:
	// enum class Mode;

	BatchNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise, std::string pName) : Operator<DTYPE>(pName) {
		std::cout << "BatchNormalize:: BatchNormalize( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, std:: string)" << '\n';

		Allocate(pInput, pScale, pBias, pIsChannelwise, 1e-6f);
	}

	BatchNormalize(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise, float pEpsilon, std::string pName) : Operator<DTYPE>(pName) {
		std::cout << "BatchNormalize:: BatchNormalize( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, float, std:: string)" << '\n';

		Allocate(pInput, pScale, pBias, pIsChannelwise, pEpsilon);
	}

	~BatchNormalize() {
		std::cout << "BatchNormalize:: ~ BatchNormalize()" << '\n';

		Delete();
	}

	int ForwardPropagate() {
		if (m_mode == INFERENCING) {
			Transform(m_pTenInput);
		} else {
			ComputeBatchSummary();

			if (m_mode == ACCUMULATING) {
				Accumulate();
			}
			Normalize();
			Transform(m_aTenNormalizedInput);
		}
		return TRUE;
	}

	int BackPropagate() {
		unsigned int inputIndex = 0;
		unsigned int batchSummaryIndex = 0;

		float derResultValue = 0.f;

		m_pTenDerScale->Reset();
		m_pTenDerBias->Reset();

		for (int example = 0; example < m_inputBatchSize; example++) {
			for (int channel = 0; channel < m_numChannel; channel++) {
				for (int inputRow = 0; inputRow < m_numInputRow; inputRow++) {
					for (int inputColumn = 0; inputColumn < m_numInputColumn; inputColumn++) {
						inputIndex = Index4D(m_pInputShape, example, channel, inputRow, inputColumn);
						batchSummaryIndex = GetBatchSummaryIndex(channel, inputRow, inputColumn);

						derResultValue = (*m_pTenDerResult)[inputIndex];

						(*m_pTenDerScale)[batchSummaryIndex] += derResultValue * (*m_aTenNormalizedInput)[inputIndex];
						(*m_pTenDerBias)[batchSummaryIndex] += derResultValue;
					}
				}
			}
		}

		for (int example = 0; example < m_inputBatchSize; example++) {
			for (int channel = 0; channel < m_numChannel; channel++) {
				for (int inputRow = 0; inputRow < m_numInputRow; inputRow++) {
					for (int inputColumn = 0; inputColumn < m_numInputColumn; inputColumn++) {
						inputIndex = Index4D(m_pInputShape, example, channel, inputRow, inputColumn);
						batchSummaryIndex = GetBatchSummaryIndex(channel, inputRow, inputColumn);

						(*m_pTenDerInput)[inputIndex] = ( (*m_pTenDerResult)[inputIndex] - ( (*m_pTenDerBias)[batchSummaryIndex] + (*m_pTenDerScale)[batchSummaryIndex] * (*m_aTenNormalizedInput)[inputIndex]) / m_effectiveBatchSize ) * (*m_pTenScale)[batchSummaryIndex] / (*m_aTenBatchStandardDeviation)[batchSummaryIndex];
					}
				}
			}
		}
		return TRUE;
	}

	void SetModeTraining() {
		// std::cout << "BatchNormalize::SetModeTraining()" << '\n';

		if (m_mode == ACCUMULATING) {
			;
		} else if (m_mode == INFERENCING) {
			RestoreTransform();
		} else {
			return;
		}
		m_mode = TRAINING;
	}

	void SetModeAccumulating() {
		// std::cout << "BatchNormalize::SetModeAccumulating()" << '\n';

		if (m_mode == TRAINING) {
			m_numBatch = 0;

			m_aTenTotalMean->Reset();
			m_aTenTotalStandardDeviation->Reset();
		} else if (m_mode == INFERENCING) {
			RestoreTransform();
			RestoreAccumulation();
		} else {
			return;
		}
		m_mode = ACCUMULATING;
	}

	void SetModeInferencing() {
		// std::cout << "BatchNormalize::SetModeInferencing()" << '\n';

		if ((m_mode == ACCUMULATING) && (m_numBatch > 0)) {
			ComputeTotalSummary();
			ReplaceTransform();
		} else {
			return;
		}
		m_mode = INFERENCING;
	}

	// enum class Mode {
	// TRAINING,
	// ACCUMULATING,
	// INFERENCING
	// };

private:
	Tensor<DTYPE> *m_pTenInput;
	Tensor<DTYPE> *m_pTenScale;
	Tensor<DTYPE> *m_pTenBias;
	Tensor<DTYPE> *m_pTenResult;

	Tensor<DTYPE> *m_pTenDerInput;
	Tensor<DTYPE> *m_pTenDerScale;
	Tensor<DTYPE> *m_pTenDerBias;
	Tensor<DTYPE> *m_pTenDerResult;

	Shape *m_pInputShape;
	int m_inputBatchSize;
	int m_numChannel;
	int m_numInputRow;
	int m_numInputColumn;

	int m_isChannelwise;
	int m_effectiveBatchSize;
	int m_numBatchSummaryRow;
	int m_numBatchSummaryColumn;
	Shape *m_aBatchSummaryShape;

	Tensor<DTYPE> *m_aTenNormalizedInput;

	Tensor<DTYPE> *m_aTenBatchMean;
	Tensor<DTYPE> *m_aTenBatchStandardDeviation;

	Tensor<DTYPE> *m_aTenTotalMean;
	Tensor<DTYPE> *m_aTenTotalStandardDeviation;

	float m_epsilon;

	Mode m_mode;
	int m_numBatch;

	void Allocate(Operator<DTYPE> *pInput, Operator<DTYPE> *pScale, Operator<DTYPE> *pBias, int pIsChannelwise, float pEpsilon) {
		Operator<DTYPE>::Alloc(3, pInput, pScale, pBias);

		std::cout << "BatchNormalize:: Allocate( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, float)" << '\n';

		m_pTenInput = pInput->GetResult();
		m_pTenScale = pScale->GetResult();
		m_pTenBias = pBias->GetResult();

		m_pTenDerInput = pInput->GetDelta();
		m_pTenDerScale = pScale->GetGradient();
		m_pTenDerBias = pBias->GetGradient();

		m_pInputShape = m_pTenInput->GetShape();
		m_inputBatchSize = m_pTenInput->GetBatchSize();
		m_numChannel = m_pTenInput->GetChannelSize();
		m_numInputRow = m_pTenInput->GetRowSize();
		m_numInputColumn = m_pTenInput->GetColSize();

		m_isChannelwise = pIsChannelwise;

		if (pIsChannelwise) {
			m_effectiveBatchSize = m_inputBatchSize * m_numInputRow * m_numInputColumn;

			m_numBatchSummaryRow = 1;
			m_numBatchSummaryColumn = 1;
		} else {
			m_effectiveBatchSize = m_inputBatchSize;

			m_numBatchSummaryRow = m_numInputRow;
			m_numBatchSummaryColumn = m_numInputColumn;
		}
		m_aBatchSummaryShape = new Shape(1, 1, m_numChannel, m_numBatchSummaryRow, m_numBatchSummaryColumn);

		this->SetResult(new Tensor<DTYPE>(new Shape(m_pInputShape)));
		this->SetDelta(new Tensor<DTYPE>(new Shape(m_pInputShape)));
		m_pTenResult = this->GetResult();
		m_pTenDerResult = this->GetDelta();

		m_aTenNormalizedInput = new Tensor<DTYPE>(new Shape(m_pInputShape));

		m_aTenBatchMean = new Tensor<DTYPE>(new Shape(m_aBatchSummaryShape));
		m_aTenBatchStandardDeviation = new Tensor<DTYPE>(new Shape(m_aBatchSummaryShape));

		m_aTenTotalMean = new Tensor<DTYPE>(new Shape(m_aBatchSummaryShape));
		m_aTenTotalStandardDeviation = new Tensor<DTYPE>(new Shape(m_aBatchSummaryShape));

		m_aTenNormalizedInput->Reset();

		m_aTenBatchMean->Reset();
		m_aTenBatchStandardDeviation->Reset();

		m_aTenTotalMean->Reset();
		m_aTenTotalStandardDeviation->Reset();

		m_epsilon= pEpsilon;

		m_mode = TRAINING;
		m_numBatch = 0;
	}

	void Delete() {
		if (m_aBatchSummaryShape) {
			delete m_aBatchSummaryShape;
			m_aBatchSummaryShape = NULL;
		}
		if (m_aTenNormalizedInput) {
			delete m_aTenNormalizedInput;
			m_aTenNormalizedInput = NULL;
		}
		if (m_aTenBatchMean) {
			delete m_aTenBatchMean;
			m_aTenBatchMean = NULL;
		}
		if (m_aTenBatchStandardDeviation) {
			delete m_aTenBatchStandardDeviation;
			m_aTenBatchStandardDeviation = NULL;
		}
		if (m_aTenTotalMean) {
			delete m_aTenTotalMean;
			m_aTenTotalMean = NULL;
		}
		if (m_aTenTotalStandardDeviation) {
			delete m_aTenTotalStandardDeviation;
			m_aTenTotalStandardDeviation = NULL;
		}
	}

	void ComputeBatchSummary() {
		unsigned int batchSummaryIndex = 0;

		float inputValue = 0.f;
		float meanValue = 0.f;
		float standardDeviationValue = 0.f;

		m_aTenBatchMean->Reset();
		m_aTenBatchStandardDeviation->Reset();

		for (int example = 0; example < m_inputBatchSize; example++) {
			for (int channel = 0; channel < m_numChannel; channel++) {
				for (int inputRow = 0; inputRow < m_numInputRow; inputRow++) {
					for (int inputColumn = 0; inputColumn < m_numInputColumn; inputColumn++) {
						batchSummaryIndex = GetBatchSummaryIndex(channel, inputRow, inputColumn);

						inputValue = (*m_pTenInput)[Index4D(m_pInputShape, example, channel, inputRow, inputColumn)];

						(*m_aTenBatchMean)[batchSummaryIndex] += inputValue;
						(*m_aTenBatchStandardDeviation)[batchSummaryIndex] += inputValue * inputValue;
					}
				}
			}
		}

		for (int channel = 0; channel < m_numChannel; channel++) {
			for (int batchSummaryRow = 0; batchSummaryRow < m_numBatchSummaryRow; batchSummaryRow++) {
				for (int batchSummaryColumn = 0; batchSummaryColumn < m_numBatchSummaryColumn; batchSummaryColumn++) {
					batchSummaryIndex = Index4D(m_aBatchSummaryShape, 0, channel, batchSummaryRow, batchSummaryColumn);

					meanValue = (*m_aTenBatchMean)[batchSummaryIndex] / m_effectiveBatchSize;
					standardDeviationValue = SqrtStable(((*m_aTenBatchStandardDeviation)[batchSummaryIndex] / m_effectiveBatchSize) - (meanValue * meanValue));

					(*m_aTenBatchMean)[batchSummaryIndex] = meanValue;
					(*m_aTenBatchStandardDeviation)[batchSummaryIndex] = standardDeviationValue;
				}
			}
		}
	}

	void Accumulate() {
		unsigned int batchSummaryIndex = 0;

		float standardDeviationValue = 0.f;

		for (int channel = 0; channel < m_numChannel; channel++) {
			for (int batchSummaryRow = 0; batchSummaryRow < m_numBatchSummaryRow; batchSummaryRow++) {
				for (int batchSummaryColumn = 0; batchSummaryColumn < m_numBatchSummaryColumn; batchSummaryColumn++) {
					batchSummaryIndex = Index4D(m_aBatchSummaryShape, 0, channel, batchSummaryRow, batchSummaryColumn);

					standardDeviationValue = (*m_aTenBatchStandardDeviation)[batchSummaryIndex];

					(*m_aTenTotalMean)[batchSummaryIndex] += (*m_aTenBatchMean)[batchSummaryIndex];
					(*m_aTenTotalStandardDeviation)[batchSummaryIndex] += standardDeviationValue * standardDeviationValue;
				}
			}
		}
		m_numBatch++;
	}

	void Normalize() {
		unsigned int inputIndex = 0;
		unsigned int batchSummaryIndex = 0;

		float deviationValue = 0.f;

		for (int example = 0; example < m_inputBatchSize; example++) {
			for (int channel = 0; channel < m_numChannel; channel++) {
				for (int inputRow = 0; inputRow < m_numInputRow; inputRow++) {
					for (int inputColumn = 0; inputColumn < m_numInputColumn; inputColumn++) {
						inputIndex = Index4D(m_pInputShape, example, channel, inputRow, inputColumn);
						batchSummaryIndex = GetBatchSummaryIndex(channel, inputRow, inputColumn);

						deviationValue = (*m_pTenInput)[inputIndex] - (*m_aTenBatchMean)[batchSummaryIndex];

						(*m_aTenNormalizedInput)[inputIndex] = deviationValue / (*m_aTenBatchStandardDeviation)[batchSummaryIndex];
					}
				}
			}
		}
	}

	void Transform(Tensor<DTYPE> *pTenInput) {
		unsigned int inputIndex = 0;
		unsigned int batchSummaryIndex = 0;

		for (int example = 0; example < m_inputBatchSize; example++) {
			for (int channel = 0; channel < m_numChannel; channel++) {
				for (int inputRow = 0; inputRow < m_numInputRow; inputRow++) {
					for (int inputColumn = 0; inputColumn < m_numInputColumn; inputColumn++) {
						inputIndex = Index4D(m_pInputShape, example, channel, inputRow, inputColumn);
						batchSummaryIndex = GetBatchSummaryIndex(channel, inputRow, inputColumn);

						(*m_pTenResult)[inputIndex] = ((*m_pTenScale)[batchSummaryIndex] * (*pTenInput)[inputIndex]) + (*m_pTenBias)[batchSummaryIndex];
					}
				}
			}
		}
	}

	void ComputeTotalSummary() {
		unsigned int batchSummaryIndex = 0;

		float varianceValue = 0.f;

		float varianceBias = GetVarianceBias();

		for (int channel = 0; channel < m_numChannel; channel++) {
			for (int batchSummaryRow = 0; batchSummaryRow < m_numBatchSummaryRow; batchSummaryRow++) {
				for (int batchSummaryColumn = 0; batchSummaryColumn < m_numBatchSummaryColumn; batchSummaryColumn++) {
					batchSummaryIndex = Index4D(m_aBatchSummaryShape, 0, channel, batchSummaryRow, batchSummaryColumn);

					varianceValue = ((*m_aTenTotalStandardDeviation)[batchSummaryIndex] / m_numBatch) / varianceBias;

					(*m_aTenTotalMean)[batchSummaryIndex] /= m_numBatch;
					(*m_aTenTotalStandardDeviation)[batchSummaryIndex] = SqrtStable(varianceValue);
				}
			}
		}
	}

	void ReplaceTransform() {
		unsigned int batchSummaryIndex = 0;

		float scaleValue = 0.f;

		for (int channel = 0; channel < m_numChannel; channel++) {
			for (int batchSummaryRow = 0; batchSummaryRow < m_numBatchSummaryRow; batchSummaryRow++) {
				for (int batchSummaryColumn = 0; batchSummaryColumn < m_numBatchSummaryColumn; batchSummaryColumn++) {
					batchSummaryIndex = Index4D(m_aBatchSummaryShape, 0, channel, batchSummaryRow, batchSummaryColumn);

					scaleValue = (*m_pTenScale)[batchSummaryIndex] / (*m_aTenTotalStandardDeviation)[batchSummaryIndex];

					(*m_pTenScale)[batchSummaryIndex] = scaleValue;
					(*m_pTenBias)[batchSummaryIndex] -= scaleValue * (*m_aTenTotalMean)[batchSummaryIndex];
				}
			}
		}
	}

	void RestoreTransform() {
		unsigned int batchSummaryIndex = 0;

		float scaleValue = 0.f;

		for (int channel = 0; channel < m_numChannel; channel++) {
			for (int batchSummaryRow = 0; batchSummaryRow < m_numBatchSummaryRow; batchSummaryRow++) {
				for (int batchSummaryColumn = 0; batchSummaryColumn < m_numBatchSummaryColumn; batchSummaryColumn++) {
					batchSummaryIndex = Index4D(m_aBatchSummaryShape, 0, channel, batchSummaryRow, batchSummaryColumn);

					scaleValue = (*m_pTenScale)[batchSummaryIndex];

					(*m_pTenScale)[batchSummaryIndex] = scaleValue * (*m_aTenTotalStandardDeviation)[batchSummaryIndex];
					(*m_pTenBias)[batchSummaryIndex] += scaleValue * (*m_aTenTotalMean)[batchSummaryIndex];
				}
			}
		}
	}

	void RestoreAccumulation() {
		unsigned int batchSummaryIndex = 0;

		float standardDeviationValue = 0.f;

		float varianceBias = GetVarianceBias();

		for (int channel = 0; channel < m_numChannel; channel++) {
			for (int batchSummaryRow = 0; batchSummaryRow < m_numBatchSummaryRow; batchSummaryRow++) {
				for (int batchSummaryColumn = 0; batchSummaryColumn < m_numBatchSummaryColumn; batchSummaryColumn++) {
					batchSummaryIndex = Index4D(m_aBatchSummaryShape, 0, channel, batchSummaryRow, batchSummaryColumn);

					standardDeviationValue = (*m_aTenTotalStandardDeviation)[batchSummaryIndex];

					(*m_aTenTotalMean)[batchSummaryIndex] *= m_numBatch;
					(*m_aTenTotalStandardDeviation)[batchSummaryIndex] = standardDeviationValue * standardDeviationValue * m_numBatch * varianceBias;
				}
			}
		}
	}

	unsigned int GetBatchSummaryIndex(int pChannel, int pInputRow, int pInputColumn) {
		if (m_isChannelwise) {
			return Index4D(m_aBatchSummaryShape, 0, pChannel, 0, 0);
		} else {
			return Index4D(m_aBatchSummaryShape, 0, pChannel, pInputRow, pInputColumn);
		}
	}

	float GetVarianceBias() {
		if (m_effectiveBatchSize > 1) {
			return (m_effectiveBatchSize - 1.f) / m_effectiveBatchSize;
		} else {
			return 1.f;
		}
	}

	float SqrtStable(float base) {
		return std::sqrt(base + m_epsilon);
	}
};
#endif // __BATCH_NORMALIZE__
