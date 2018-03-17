#include "cMLP.h"
#include <cstdlib>
#include <string.h>
#include <stdio.h>

cMLP::cMLP() { }
cMLP::~cMLP() { }

// Back Prop Initialization
cMLP::cMLP(double learning_rate, double momentum)
	:learning_rate(learning_rate), momentum_factor(momentum)
{
	propagation_algorithm = 0;
	memset(inputValue, 0, sizeof(double) * (Ninput + 1));
	memset(hiddenValue, 0, sizeof(double) * (Nhidden + 1));
	memset(outputValue, 0, sizeof(double) * Noutput);
	memset(expectedValue, 0, sizeof(double) * Noutput);
	memset(e_output, 0, sizeof(double) * Noutput);
	memset(e_hidden, 0, sizeof(double) * Nhidden);

	memset(wc_inputhidden, 0, sizeof(double) * Nhidden * (Ninput + 1));
	memset(wc_hiddenoutput, 0, sizeof(double) * Noutput * (Nhidden + 1));

	// bias
	inputValue[Ninput] = 1.0;
	hiddenValue[Nhidden] = 1.0;
}

// Resilient Prop Initialization
cMLP::cMLP(int prop_algorithm)
{
	propagation_algorithm = 1;

	// common component
	memset(inputValue, 0, sizeof(double) * (Ninput + 1));
	memset(hiddenValue, 0, sizeof(double) * (Nhidden + 1));
	memset(outputValue, 0, sizeof(double) * Noutput);
	memset(expectedValue, 0, sizeof(double) * Noutput);
	memset(e_output, 0, sizeof(double) * Noutput);
	memset(e_hidden, 0, sizeof(double) * Nhidden);

	memset(Acc_WeightError_inputHidden, 0, sizeof(double) * (Ninput + 1) * Nhidden);
	memset(Acc_WeightError_hiddenOutput, 0, sizeof(double) * (Nhidden + 1) * Noutput);
	memset(Prev_Acc_WeightError_inputHidden, 0, sizeof(double) * (Ninput + 1) * Nhidden);
	memset(Prev_Acc_WeightError_hiddenOutput, 0, sizeof(double) * (Nhidden + 1) * Noutput);
	for (int i = 0; i < Ninput + 1; ++i)
		for (int j = 0; j < Nhidden; ++j)
			prev_wd_inputhidden[i][j] = 0.01;
	for (int i = 0; i < Nhidden + 1; ++i)
		for (int j = 0; j < Noutput; ++j)
			prev_wd_hiddenoutput[i][j] = 0.01;


	// bias
	inputValue[Ninput] = 1.0;
	hiddenValue[Nhidden] = 1.0;
}

double cMLP::sigmoid(double value)
{
	return 1.0 / (1.0 + exp(-value));
}

double cMLP::sigmoid_d(double value)
{
	return value * (1.0 - value);
}

void cMLP::setInputvalue(double* value, double noise)
{
	double real_noise = (1.0 - noise);
	if (noise >= 1.0) real_noise = 0.0000001;
	for (int i = 0; i < Ninput; ++i)
	{
		value[i] *= real_noise;
	}
	memcpy(inputValue, value, Ninput * sizeof(double));
}

void cMLP::setExpectedvalue(double* value)
{
	memcpy(expectedValue, value, Noutput * sizeof(double));
}

void cMLP::Bprop_feedforward()
{
	for (int i = 0; i < Nhidden; ++i)
	{
		hiddenValue[i] = 0.0;

		for (int j = 0; j < Ninput + 1; ++j)
		{
			hiddenValue[i] += w_inputhidden[j][i] * inputValue[j];
		}

		hiddenValue[i] = sigmoid(hiddenValue[i]);
	}

	for (int i = 0; i < Noutput; ++i)
	{
		outputValue[i] = 0.0;

		for (int j = 0; j < Nhidden + 1; ++j)
		{
			outputValue[i] += w_hiddenoutput[j][i] * hiddenValue[j];
		}

		outputValue[i] = sigmoid(outputValue[i]);
	}
}

void cMLP::Bprop_calculateErrors()
{
	// output error
	for (int i = 0; i < Noutput; ++i)
	{
		if (expectedValue[i] != outputValue[i])
			e_output[i] = (expectedValue[i] - outputValue[i]) * sigmoid_d(outputValue[i]);
	}

	// hidden error
	for (int i = 0; i < Nhidden; ++i)
	{
		e_hidden[i] = 0.0;
		for (int j = 0; j < Noutput; ++j)
		{
			e_hidden[i] += e_output[j] * w_hiddenoutput[i][j];
		}
		e_hidden[i] *= sigmoid_d(hiddenValue[i]);
	}
}

void cMLP::Bprop_updateWeights()
{
	double dw = 0;
	double temp = 0;

	// output weight
	for (int i = 0; i < Nhidden; ++i)
	{
		for (int j = 0; j <Noutput; ++j)
		{
			temp = w_hiddenoutput[i][j];
			dw = learning_rate * e_output[j] * hiddenValue[i];

			w_hiddenoutput[i][j] += dw + momentum_factor * wc_hiddenoutput[i][j];
			wc_hiddenoutput[i][j] = w_hiddenoutput[i][j] - temp;
		}
	}
	// output weight bias
	for (int i = 0; i < Noutput; ++i)
		w_hiddenoutput[Nhidden][i] += learning_rate * e_output[i] * hiddenValue[Nhidden];

	// hidden weight
	for (int i = 0; i < Ninput; ++i)
	{
		for (int j = 0; j < Nhidden; ++j)
		{
			temp = w_inputhidden[i][j];
			dw = learning_rate * e_hidden[j] * inputValue[i];

			w_inputhidden[i][j] += dw + momentum_factor * wc_inputhidden[i][j];
			wc_inputhidden[i][j] = w_inputhidden[i][j] - temp;
		}
	}
	// hidden weight bias
	for (int i = 0; i < Nhidden; ++i)
		w_inputhidden[Ninput][i] += learning_rate * e_hidden[i] * inputValue[Ninput];
}

double* cMLP::getOutput(double* input)
{
	setInputvalue(input, 0.0);
	if (propagation_algorithm == 0)
		Bprop_feedforward();
	else if (propagation_algorithm == 1)
		Rprop_feedforward();
	return outputValue;
}

double cMLP::getErrors()
{
	double error{ 0.0 };

	for (int i = 0; i < Noutput; ++i)
		error += pow(expectedValue[i] - outputValue[i], 2);

	error /= Noutput;
	return error;
}

int cMLP::getMaxIndex(double* output)
{
	int MaxIndex = 0;
	double MaxValue = -1.0;

	for (int i = 0; i < Noutput; ++i)
	{
		if (output[i] >= MaxValue)
		{
			MaxValue = output[i];
			MaxIndex = i;
		}
	}

	return MaxIndex;
}

void cMLP::showStates()
{
	printf("** INPUT **\n");
	for (int i = 0; i < Ninput + 1; ++i)
		printf("%lf ", inputValue[i]);
	printf("\n\n");

	printf("** HIDDEN **\n");
	for (int i = 0; i < Nhidden + 1; ++i)
		printf("%lf ", hiddenValue[i]);
	printf("\n\n");

	printf("** Weight Input -> HIdden **\n");
	for (int i = 0; i < Ninput + 1; ++i)
	{
		for (int j = 0; j < Nhidden; ++j)
		{
			printf("%lf ", w_inputhidden[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("** Weight Hidden -> Output **\n");
	for (int i = 0; i < Nhidden + 1; ++i)
	{
		for (int j = 0; j < Noutput; ++j)
		{
			printf("%lf ", w_hiddenoutput[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("** Error hidden**\n");
	for (int i = 0; i < Nhidden; ++i)
		printf("%lf ", e_hidden[i]);
	printf("\n\n");

	printf("** Error Output**\n");
	for (int i = 0; i < Noutput; ++i)
		printf("%lf ", e_output[i]);
	printf("\n\n");


	printf("** Output , Desired Result **\n");
	for (int i = 0; i < Noutput; ++i)
		printf("%lf %lf\n", outputValue[i], expectedValue[i]);
	printf("\n");
}

void cMLP::Rprop_feedforward()
{
	// Hidden
	for (int i = 0; i < Nhidden; ++i)
	{
		hiddenValue[i] = 0.0;

		for (int j = 0; j < Ninput + 1; ++j)
		{
			hiddenValue[i] += w_inputhidden[j][i] * inputValue[j];
		}

		hiddenValue[i] = HyperTan(hiddenValue[i]);
	}

	// Output Value
	for (int i = 0; i < Noutput; ++i)
	{
		outputValue[i] = 0.0;

		for (int j = 0; j < Nhidden + 1; ++j)
		{
			outputValue[i] += w_hiddenoutput[j][i] * hiddenValue[j];
		}
	}
	double max = outputValue[0];
	for (int i = 0; i < Noutput; ++i)
		if (outputValue[i] > max) max = outputValue[i];

	double scale = 0.0;
	for (int i = 0; i < Noutput; ++i)
		scale += exp(outputValue[i] - max);

	for (int i = 0; i < Noutput; ++i)
		outputValue[i] = exp(outputValue[i] - max) / scale;
} // Rprop_feedforward

void cMLP::Rprop_calculateErrors()
{
	for (int i = 0; i < Noutput; ++i)
	{
		e_output[i] = (outputValue[i] - expectedValue[i]) * sigmoid_d(outputValue[i]);
	}

	for (int i = 0; i < Nhidden; ++i)
	{
		double derivative = (1 - hiddenValue[i]) * (1 + hiddenValue[i]);
		e_hidden[i] = 0.0;
		for (int j = 0; j < Noutput; ++j)
		{
			e_hidden[i] += e_output[j] * w_hiddenoutput[i][j];
		}
		e_hidden[i] *= derivative;
	}

	for (int i = 0; i < Nhidden + 1; ++i)
		for (int j = 0; j < Noutput; ++j)
		{
			Acc_WeightError_hiddenOutput[i][j] += e_output[j] * hiddenValue[i];
		}

	for (int i = 0; i < Ninput + 1; ++i)
		for (int j = 0; j < Nhidden; ++j)
		{
			Acc_WeightError_inputHidden[i][j] += e_hidden[j] * inputValue[i];
		}
}

void cMLP::Rprop_updateWeights()
{
	double delta = 0.0;
	// Weight I->H update
	for (int i = 0; i < Ninput + 1; ++i)
		for (int j = 0; j < Nhidden; ++j)
		{
			double key = Prev_Acc_WeightError_inputHidden[i][j] * Acc_WeightError_inputHidden[i][j];

			if (key > 0)
			{
				delta = prev_wd_inputhidden[i][j] * etaPlus;
				if (delta > deltaMax) delta = deltaMax;
				double tmp = -cMLP::sign(Acc_WeightError_inputHidden[i][j]) * delta;
				w_inputhidden[i][j] += tmp;
			}
			else if (key < 0)
			{
				delta = prev_wd_inputhidden[i][j] * etaMinus;
				if (delta < deltaMin) delta = deltaMin;
				w_inputhidden[i][j] -= prev_wd_inputhidden[i][j];
				Acc_WeightError_inputHidden[i][j] = 0;
			}
			else
			{
				delta = prev_wd_inputhidden[i][j];
				double tmp = -cMLP::sign(Acc_WeightError_inputHidden[i][j]) * delta;
				w_inputhidden[i][j] += tmp;
			}

			prev_wd_inputhidden[i][j] = delta;
			Prev_Acc_WeightError_inputHidden[i][j] = Acc_WeightError_inputHidden[i][j];
		}


	// Weight H->O update
	for (int i = 0; i < Nhidden + 1; ++i)
		for (int j = 0; j < Noutput; ++j)
		{
			double key = Prev_Acc_WeightError_hiddenOutput[i][j] * Acc_WeightError_hiddenOutput[i][j];

			if (key > 0)
			{
				delta = prev_wd_hiddenoutput[i][j] * etaPlus;
				if (delta > deltaMax) delta = deltaMax;
				double tmp = -cMLP::sign(Acc_WeightError_hiddenOutput[i][j]) * delta;
				w_hiddenoutput[i][j] += tmp;
			}
			else if (key < 0)
			{
				delta = prev_wd_hiddenoutput[i][j] * etaMinus;
				if (delta < deltaMin) delta = deltaMin;
				w_hiddenoutput[i][j] -= prev_wd_hiddenoutput[i][j];
				Acc_WeightError_hiddenOutput[i][j] = 0;
			}
			else
			{
				delta = prev_wd_hiddenoutput[i][j];
				double tmp = -cMLP::sign(Acc_WeightError_hiddenOutput[i][j]) * delta;
				w_hiddenoutput[i][j] += tmp;
			}

			prev_wd_hiddenoutput[i][j] = delta;
			Prev_Acc_WeightError_hiddenOutput[i][j] = Acc_WeightError_hiddenOutput[i][j];
		}
}

double cMLP::HyperTan(double value)
{
	if (value < -20.0) return -1.0;
	else if (value > 20.0) return 1.0;
	else return tanh(value);
}

double cMLP::sign(double value)
{
	if (value > 0)
		return 1.0;
	else if (value < 0)
		return -1.0;
	else
		return 0.0;
}

void cMLP::zeroOut()
{
	memset(Acc_WeightError_inputHidden, 0, sizeof(double) * (Ninput + 1) * Nhidden);
	memset(Acc_WeightError_hiddenOutput, 0, sizeof(double) * (Nhidden + 1) * Noutput);
}

void cMLP::setWeight(double w_ih[][Nhidden], double w_ho[][Noutput])
{
	memcpy(w_inputhidden, w_ih, sizeof(double) * Ninput * Nhidden);
	memcpy(w_hiddenoutput, w_ho, sizeof(double) * Nhidden * Noutput);
}