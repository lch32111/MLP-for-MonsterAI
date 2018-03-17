#include "MonsterAI.h"
#include <time.h>
#include <stdio.h>
#include <SDL.h>


MonsterAI::MonsterAI() { }
MonsterAI::~MonsterAI() { }

MonsterAI::MonsterAI(double earlyError, int epochLimit)
	:m_epochLimit(epochLimit)
{
	m_propAlgorithm = 1;

	for(int i = 0; i < STAGE_NUMBER; ++i)
		m_NeuralNetwork[i] = cMLP(1);

	srand((unsigned int)time(NULL));
	// populate the weights between 0.5 and -0.5
	for (int i = 0; i < Ninput + 1; ++i)
	{
		for (int j = 0; j <Nhidden; ++j)
		{
			int plus_minus = rand() % 2;
			double randWeight = rand() % 6 / 10.0;

			if (plus_minus == 1) {}
			else randWeight *= -1;

			unified_w_inputhidden[i][j] = randWeight;
		}
	}

	// populate the weights between 0.5 and -0.5
	for (int i = 0; i < Nhidden + 1; ++i)
	{
		for (int j = 0; j < Noutput; ++j)
		{
			int plus_minus = rand() % 2;
			double randWeight = rand() % 6 / 10.0;

			if (plus_minus == 1) {}
			else randWeight *= -1;

			unified_w_hiddenoutput[i][j] = randWeight;
		}
	}
	for (int i = 0; i < STAGE_NUMBER; ++i)
		m_NeuralNetwork[i].setWeight(unified_w_inputhidden, unified_w_hiddenoutput);

	double t_errorGoal = earlyError;
	for (int i = 0; i < STAGE_NUMBER; ++i)
	{
		m_errorRate[i] = t_errorGoal;
		t_errorGoal -= 0.003;
	}

	double t_noise = 0.8;
	for (int i = 0; i < STAGE_NUMBER; ++i)
	{
		noise[i] = t_noise;
		t_noise -= 0.3;
	}
	noise[STAGE_NUMBER] = 0.0;

	trainAI();
}
MonsterAI::MonsterAI(double learningRate, double momentum, double earlyError, int epochLimit)
	:m_momentum(momentum), m_epochLimit(epochLimit)
{
	m_propAlgorithm = 0;

	double t_learningRate = learningRate;

	for (int i = 0; i < STAGE_NUMBER; ++i)
	{
		m_learningRate[i] = t_learningRate;
		t_learningRate -= 0.03;
	}

	for(int i = 0; i < STAGE_NUMBER; ++i)
		m_NeuralNetwork[i] = cMLP(m_learningRate[i], m_momentum);

	srand((unsigned int)time(NULL));
	// populate the weights between 0.5 and -0.5
	for (int i = 0; i < Ninput + 1; ++i)
	{
		for (int j = 0; j <Nhidden; ++j)
		{
			int plus_minus = rand() % 2;
			double randWeight = rand() % 6 / 10.0;

			if (plus_minus == 1) {}
			else randWeight *= -1;

			unified_w_inputhidden[i][j] = randWeight;
		}
	}

	// populate the weights between 0.5 and -0.5
	for (int i = 0; i < Nhidden + 1; ++i)
	{
		for (int j = 0; j < Noutput; ++j)
		{
			int plus_minus = rand() % 2;
			double randWeight = rand() % 6 / 10.0;

			if (plus_minus == 1) {}
			else randWeight *= -1;

			unified_w_hiddenoutput[i][j] = randWeight;
		}
	}
	for (int i = 0; i < STAGE_NUMBER; ++i)
		m_NeuralNetwork[i].setWeight(unified_w_inputhidden, unified_w_hiddenoutput);

	double t_errorGoal = earlyError;
	for (int i = 0; i < STAGE_NUMBER; ++i)
	{
		m_errorRate[i] = t_errorGoal;
		t_errorGoal -= 0.003;
	}
	
	double t_noise = 0.8;
	for (int i = 0; i < STAGE_NUMBER; ++i)
	{
		noise[i] = t_noise;
		t_noise -= 0.3;
	}
	noise[STAGE_NUMBER - 1] = 0.0;

	trainAI();
}

void MonsterAI::trainAI()
{
	HANDLE t_Thread[STAGE_NUMBER];
	DWORD threadID;

	thread_info temp[STAGE_NUMBER];
	for (int i = 0; i < STAGE_NUMBER; ++i)
	{
		temp[i].origin = this;
		temp[i].thread_id = i;
	}
	
	while(1)
	{
		for (int i = 0; i < STAGE_NUMBER; ++i)
		{
			t_Thread[i] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)multiThread_entry, &temp[i], 0, &threadID);
		}

		WaitForMultipleObjects(STAGE_NUMBER, t_Thread, true, INFINITE);

		for (int i = 0; i < STAGE_NUMBER; ++i)
		{
			CloseHandle(t_Thread[i]);
		}


		double comp_accuracy[STAGE_NUMBER];
		for (int i = 0; i < STAGE_NUMBER; ++i)
			comp_accuracy[i] = getAccuracy(i);
		
		if (isValidAccuracy(comp_accuracy))
		{
			goto trainingend;
		}
		else 
		{
			printf("\nThe Accuracies are invalid to use for Monster AI\n");
			printf("Retraining will start in 3 sec\n");

			unsigned int itimePassed = SDL_GetTicks();
			int second = 0;

			while(1) 
			{  
				if (SDL_GetTicks() - itimePassed > 1000)
				{
					itimePassed = SDL_GetTicks();
					++second;
					printf("%d second passed\n", second);
				}

				if (second == 3)
					break;
			}

			printf("Retraining Start\n\n");

			if (m_propAlgorithm == 0)
			{
				for (int i = 0; i < STAGE_NUMBER; ++i)
					m_NeuralNetwork[i] = cMLP(m_learningRate[i], m_momentum);
				
			}
			else if (m_propAlgorithm == 1)
			{
				for (int i = 0; i < STAGE_NUMBER; ++i)
					m_NeuralNetwork[i] = cMLP(1);
			}

			srand((unsigned int)time(NULL));
			// populate the weights between 0.5 and -0.5
			for (int i = 0; i < Ninput + 1; ++i)
			{
				for (int j = 0; j <Nhidden; ++j)
				{
					int plus_minus = rand() % 2;
					double randWeight = rand() % 6 / 10.0;

					if (plus_minus == 1) {}
					else randWeight *= -1;

					unified_w_inputhidden[i][j] = randWeight;
				}
			}

			// populate the weights between 0.5 and -0.5
			for (int i = 0; i < Nhidden + 1; ++i)
			{
				for (int j = 0; j < Noutput; ++j)
				{
					int plus_minus = rand() % 2;
					double randWeight = rand() % 6 / 10.0;

					if (plus_minus == 1) {}
					else randWeight *= -1;

					unified_w_hiddenoutput[i][j] = randWeight;
				}
			}
			for (int i = 0; i < STAGE_NUMBER; ++i)
				m_NeuralNetwork[i].setWeight(unified_w_inputhidden, unified_w_hiddenoutput);
		}
	}
	
trainingend:
	printf("The Accuracies are valid to use for Monster AI\n");
}

DWORD WINAPI MonsterAI::multiThread_entry(LPVOID lpParam)
{
	thread_info* t_info = (thread_info*)lpParam;

	if (t_info->origin->m_propAlgorithm == 0)
	{
		t_info->origin->multiThread_BpropTrainAI(t_info->thread_id);
	}
	else if(t_info->origin->m_propAlgorithm == 1)
	{
		t_info->origin->multiThread_RpropTrainAI(t_info->thread_id);
	}

	return true;
}

void MonsterAI::multiThread_BpropTrainAI(int NN_ID)
{
	/* Training Zone */
	printf("Bprop Thread %d Training Start!!!\n\n", NN_ID);
	double mse = 999;
	int epoch = 0;
	while (mse > m_errorRate[NN_ID] && epoch < m_epochLimit)
	{
		mse = 0;

		for (int i = 0; i < INPUT_NUMBER; ++i)
		{
			for (int j = 0; j < Ninput; ++j)
				Inputs[j] = gameTrainingSet[i][j];
			for (int k = 0; k < Noutput; ++k)
				Results[k] = gameTrainingSet[i][k + 6];

			m_NeuralNetwork[NN_ID].setInputvalue(Inputs, noise[NN_ID]);
			m_NeuralNetwork[NN_ID].setExpectedvalue(Results);
			m_NeuralNetwork[NN_ID].Bprop_feedforward();
			m_NeuralNetwork[NN_ID].Bprop_calculateErrors();
			m_NeuralNetwork[NN_ID].Bprop_updateWeights();
			mse += m_NeuralNetwork[NN_ID].getErrors();
			// MonsterAI.showStates();
		}

		mse /= INPUT_NUMBER;
		++epoch;

		printf("Bprop Thread %d, epoch %d : mse %lf\n", NN_ID, epoch, mse);
	}
	printf("Bprop Thread %d Training Done!!!\n\n", NN_ID);
	/* Training Zone */
}

void MonsterAI::multiThread_RpropTrainAI(int NN_ID)
{
	/* Training Zone */
	printf("Rprop Thread %d Training Start!!!\n\n", NN_ID);
	double mse = 999;
	int epoch = 0;
	while (mse > m_errorRate[NN_ID] && epoch < m_epochLimit)
	{
		mse = 0;

		for (int i = 0; i < INPUT_NUMBER; ++i)
		{
			for (int j = 0; j < Ninput; ++j)
				Inputs[j] = gameTrainingSet[i][j];
			for (int k = 0; k < Noutput; ++k)
				Results[k] = gameTrainingSet[i][k + 6];

			m_NeuralNetwork[NN_ID].setInputvalue(Inputs, noise[NN_ID]);
			m_NeuralNetwork[NN_ID].setExpectedvalue(Results);
			m_NeuralNetwork[NN_ID].Rprop_feedforward();
			m_NeuralNetwork[NN_ID].Rprop_calculateErrors();
			mse += m_NeuralNetwork[NN_ID].getErrors();
			// MonsterAI.showStates();
		}

		m_NeuralNetwork[NN_ID].Rprop_updateWeights();
		mse /= INPUT_NUMBER;
		++epoch;

		printf("epoch %d : mse %lf\n", epoch, mse);
	}
	printf("Rprop Thread %d Training Done!!!\n\n", NN_ID);
	/* Training Zone */
}

double MonsterAI::getAccuracy(int NN_ID)
{
	int right = 0, wrong = 0;
	for (int j = 0; j < INPUT_NUMBER; ++j)
	{
		for (int i = 0; i < Ninput; ++i)
			Inputs[i] = gameTrainingSet[j][i];
		double* output = m_NeuralNetwork[NN_ID].getOutput(Inputs);

		int MaxIndex = m_NeuralNetwork[NN_ID].getMaxIndex(output);
		for (int k = 0; k < Noutput; ++k)
			Results[k] = gameTrainingSet[j][k + 6];
		int DesiredMaxIndex = m_NeuralNetwork[NN_ID].getMaxIndex(Results);

		if (MaxIndex == DesiredMaxIndex)
			++right;
		else
			++wrong;
	}
	double Accuracy = (double)right / (right + wrong);
	printf("Learning rate : %lf, momentum : %lf, Error Goal : %lf, noise : %lf\n", m_learningRate[NN_ID], m_momentum, m_errorRate[NN_ID], noise[NN_ID]);
	printf("Test Result -> Right : %d, Wrong : %d, Total Data : %d\n", right, wrong, INPUT_NUMBER);
	printf("Thread %d Accuracy : %lf Percentage\n", NN_ID, Accuracy * 100.0);

	return Accuracy;
}

int MonsterAI::decideBehavior(double* input, int currentStage)
{
	int decision = -1;
	double* output = m_NeuralNetwork[currentStage].getOutput(input);
	decision = m_NeuralNetwork[currentStage].getMaxIndex(output);
	return decision;
}

void MonsterAI::resetNN(double error_goal)
{
	/*
	m_NeuralNetwork = cMLP(m_learningRate, m_momentum);
	trainAI(error_goal);
	*/
}

bool MonsterAI::isValidAccuracy(double* accuracy)
{
	// The last stage should be more than 90%
	if (accuracy[STAGE_NUMBER - 1] < 0.9) return false;

	// The accuracy should be going up as the stage goes on
	for (int i = 0; i < STAGE_NUMBER - 1; ++i)
	{
		if (accuracy[i] > accuracy[i + 1])
		{
			return false;
		}
	}

	return true;
}