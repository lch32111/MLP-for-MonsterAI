#ifndef __C_MLP_H__
#define __C_MLP_H__

#define Ninput 6
#define Nhidden 5
#define Noutput 3

class cMLP
{
private:
	// Common components
	double inputValue[Ninput + 1];   // +1 == bias
	double hiddenValue[Nhidden + 1]; // +1 == bias
	double outputValue[Noutput];
	double expectedValue[Noutput];

	double e_output[Noutput];
	double e_hidden[Nhidden];

	// 0 == Back propagation
	// 1 == Resilient propagation
	int propagation_algorithm;

	double sigmoid_d(double value);
private:
	// Back Propagation
	double learning_rate;
	double momentum_factor;

	double w_inputhidden[Ninput + 1][Nhidden];
	double w_hiddenoutput[Nhidden + 1][Noutput];
	double wc_inputhidden[Ninput + 1][Nhidden];
	double wc_hiddenoutput[Nhidden + 1][Noutput];

	double sigmoid(double value);
private:
	// Resilient Propagation
	double etaPlus = 1.2;
	double etaMinus = 0.5;
	double deltaMax = 50.0;
	double deltaMin = 1E-6;

	double Acc_WeightError_inputHidden[Ninput + 1][Nhidden];
	double Acc_WeightError_hiddenOutput[Nhidden + 1][Noutput];

	double Prev_Acc_WeightError_inputHidden[Ninput + 1][Nhidden];
	double Prev_Acc_WeightError_hiddenOutput[Nhidden + 1][Noutput];

	double prev_wd_inputhidden[Ninput + 1][Nhidden];
	double prev_wd_hiddenoutput[Nhidden + 1][Noutput];

	double sign(double value);
	double HyperTan(double value);
public:
	cMLP();
	~cMLP();
	cMLP(double learning_rate, double momentum);
	cMLP(int prop_algorithm);

	void setWeight(double w_ih[][Nhidden], double w_ho[][Noutput]);
	void zeroOut();
	void setInputvalue(double* value, double noise);
	void setExpectedvalue(double* value);

	void Bprop_feedforward();
	void Bprop_calculateErrors();
	void Bprop_updateWeights();

	void Rprop_feedforward();
	void Rprop_calculateErrors();
	void Rprop_updateWeights();

	double getErrors();
	double* getOutput(double* input);
	int getMaxIndex(double* output);
	void showStates();
};


#endif
