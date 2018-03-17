#include "Perceptron.h"
#include "NNTools.h"

Perceptron::Perceptron() { }
Perceptron::Perceptron(int inputNumb, outputFunction func, double thre)
{
	m_Function = func;
	m_threshold = thre;
	m_inputLayer = std::vector<double>(inputNumb + 1, 0);
	m_inputLayer[m_inputLayer.size() - 1] = 1.0; // bias
	m_weights = std::vector<double>(inputNumb + 1, 0);

	srand((unsigned int)time(NULL));

	// populate the weights between 0.5 and -0.5
	for (int i = 0; i < inputNumb + 1; ++i)
	{
		int plus_minus = rand() % 2;
		double randWeight = rand() % 6 / 10.0;

		if (plus_minus == 1) {}
		else randWeight *= -1;

		m_weights[i] = randWeight;
	}
} // Perceptron()

void Perceptron::setInput(std::vector<double> input)
{
	if (input.size() == m_inputLayer.size())
	{
		std::cout << "The size of input data should be the same as the number of inputlayer node without bias node\n"
			<< "input size : " << input.size() << ", inputlayer node size : " << m_inputLayer.size() - 1;
		
		throw Invalid();
	}

	for (int i = 0; i < input.size(); ++i)
		m_inputLayer[i] = input[i];
} // SetInput()

void Perceptron::UpdatePerceptron(double Learning_rate, double target)
{
	calculateNet();
	CheckResult(target);
	adjustWeights(Learning_rate, target);
} // UpdatePerceptron()

void Perceptron::UseLearnedPerceptron(std::vector<double>& input, double target)
{
	setInput(input);
	calculateNet();
	CheckResult(target);
} // UseLearningPerceptron()

void Perceptron::calculateNet()
{
	double result = 0;

	for (unsigned int i = 0; i < m_inputLayer.size(); ++i)
			result += m_inputLayer[i] * m_weights[i];

	switch (m_Function)
	{
	case outputFunction::BINARY_STEP:
		if (result > 0) result = 1;
		else result = 0;
		break;
	case outputFunction::BIPOLAR_STEP:
		if (result > 0) result = 1;
		else result = -1;
		break;
	case outputFunction::SIGMOIDAL:
	{
		result = sigmoid(result);
		if (result > 0.5) result = 1;
		else result = 0;
	}
	break;
	case outputFunction::THRESHOLDED_STEP:
		if (result > m_threshold) result = 1;
		else result = 0;
		break;
	default:
		throw Invalid();
		break;
	}

	m_output = result;
} // calculateNet()

void Perceptron::CheckResult(double target)
{
	// Compare the result calculated by perceptron with the actual result.
	if (m_output == target)
		std::cout << "Right^^ -> The output is " << m_output
		<< " // The Target is " << target << '\n';
	else
		std::cout << "Wrong!! -> The output is " << m_output
		<< " // The Target is " << target << '\n';
}  // CheckResult()

void Perceptron::adjustWeights(double Learning_rate, double target)
{
	if (m_output != target)
	{
		double error = target - m_output;

		for (unsigned int i = 0; i < m_weights.size(); ++i)
		{
			m_weights[i] += Learning_rate * error * m_inputLayer[i];
		}
	}
} // adjustWeights()

unsigned int Perceptron::getNumbInput()
{
	return m_inputLayer.size() - 1;
}

std::vector<double>& Perceptron::getInputLayer()
{
	return m_inputLayer;
}

