# MLP-for-MonsterAI
This is for my portfolio which was implemented for the submission of 
"Artificial Intelligence and Affective Computing" Module on Northumbria University in the UK.

You can play the exe file. There is a memo which let you know how to play the game.
This exe file can be played on the windows 10.

In the 'Code' folder, you will see the core AI implementation : Monster AI, Multi Layer Perceptron(MLP), Single Layer Perceptron(SLP).

The SLP is just a kind of pratice for me to implement MLP.
Only the MLP is used to implement the monster AI code.

When I implemented the MLP and monster AI, I considered faster algorithm for Neural Network(NN) because it will be used on games. 
As the result of a research, I found that Resilient Propagation(Rprop) will make the process faster 
instead of Back Propagation(Bprop) on typical NN

So, I referred to C# Rprop code and algorithm on the internet and thesis, and changed and modified it into the C/C++ code.
However, After some experiments on Rprop, I concluded that I can't use Rprop because of instability of the training result.
Consequently, I just used Bprop on my NN. 

the monster AI code is used like this :

1. Before game starts, monster AI code train three NNs by Multi-threading (each one for each stage, There are totally 3 stages).
2. If the accuracy of training is not increasing as each stage clear, the monster AI should train three NNs again.
3. If the accuracy of training is okay with that condition, Each monster will get a decision from each stage NN.

* you will be able to see the console showing the decision of each monster.
* you will be able to see the training data on monster AI code.
  I gathered that kinds of data by playing my game directly. So, this kind of data may have a bias which I intended to implement.
  
  
 After this project, I feel that unsupervised training is needed to make the monster AI which has enough high intelligence to defeat any user.


