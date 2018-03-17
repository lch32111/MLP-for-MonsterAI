#ifndef __PERCEPTRON_H__
#define __PERCEPTRON_H__

#include <iostream>
#include <vector>
#include <time.h>


enum class outputFunction
{
	BINARY_STEP,
	BIPOLAR_STEP,
	SIGMOIDAL,
	THRESHOLDED_STEP,
	OUTPUT_FUNCTION_COUNT
};

class Perceptron
{
public:
	Perceptron();

	// If you want to use the function thresholded step, pass the value on thre
	Perceptron(int inputNumb, outputFunction func, double thre);

	void setInput(std::vector<double> input);
	void UpdatePerceptron(double Learning_rate, double target);
	void UseLearnedPerceptron(std::vector<double>& input, double target);
	
	// Get Set
	unsigned int getNumbInput();
	std::vector<double>& getInputLayer();
private:
	std::vector<double> m_inputLayer;
	std::vector<double> m_weights;
	outputFunction m_Function;
	double m_threshold;
	double m_output;

	class Invalid {};

	void calculateNet();
	void CheckResult(double target);
	void adjustWeights(double Learning_rate, double target);
};

#endif
