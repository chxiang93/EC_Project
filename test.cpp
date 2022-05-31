//#include <iostream>
//#include <fstream>
//#define PY_SSIZE_T_CLEAN
//#include "D:\Development\Python\Python310\include\Python.h"
//
//int main(int argc, char* argv[])
//{
//	wchar_t* program = Py_DecodeLocale(argv[0], NULL);
//
//	if (program == NULL)
//	{
//		fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
//		exit(1);
//	}
//
//	Py_SetProgramName(program);
//	Py_Initialize();
//	PyRun_SimpleString(
//		"import numpy as np\n"
//		"import pandas as pd\n"
//		"import matplotlib.pyplot as plt\n"
//		"from sklearn.model_selection import train_test_split\n"
//		"from sklearn.neighbors import KNeighborsClassifier\n"
//		"from sklearn.metrics import confusion_matrix\n"
//		"from sklearn.metrics import roc_curve\n"
//		"from sklearn.metrics import roc_auc_score\n"
//		"plt.style.use('ggplot')\n"
//	);
//
//	PyRun_SimpleString(
//		"df = pd.read_csv('breast-cancer-wisconsin.csv')\n"
//		"df.columns = ['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']\n"
//		"df = df.drop(columns = 'Sample Code Number')\n"
//		"df = df.replace('?', np.NaN)\n"
//		"df = df.dropna()\n"
//		"X = df.drop('Class', axis = 1).values\n"
//		"y = df['Class'].values\n"
//	);
//
//	PyRun_SimpleString(
//		"for c in range(len(y)) :\n"
//		"	if y[c] == 2 :\n"
//		"		y[c] = 0\n"
//		"	elif y[c] == 4 :\n"
//		"		y[c] = 1\n"
//		"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42, stratify = y)\n"
//		"neighbors = np.arange(1, 9)\n"
//		"train_accuracy = np.empty(len(neighbors))\n"
//		"test_accuracy = np.empty(len(neighbors))\n"
//		"for i, k in enumerate(neighbors) :\n"
//		"	knn = KNeighborsClassifier(n_neighbors = k)\n"
//		"	knn.fit(X_train, y_train)\n"
//		"	train_accuracy[i] = knn.score(X_train, y_train)\n"
//		"	test_accuracy[i] = knn.score(X_test, y_test)\n"
//	);
//
//	PyRun_SimpleString(
//		"knn = KNeighborsClassifier(n_neighbors = 3)\n"
//
//		"knn.fit(X_train, y_train)\n"
//
//		"knn.score(X_test, y_test)\n"
//
//		"y_pred = knn.predict(X_test)\n"
//
//		"cm = confusion_matrix(y_test, y_pred)\n"
//		"print('Confusion Matrix :')\n"
//		"print(cm)\n"
//		"print()\n"
//
//		"TP = cm[0][0]\n"
//		"FN = cm[0][1]\n"
//		"FP = cm[1][0]\n"
//		"TN = cm[1][1]\n"
//
//		"print('Sensitivity :', TP / (TP + FN))\n"
//		"print('Specificity :', TN / (TN + FP))\n"
//		"print('Precision \t:', TP / (TP + FP))\n"
//		"print('Recall \t\t:', TP / (TP + FN))\n"
//		"print('Accuracy \t:', (TP + TN) / (TP + TN + FP + FN))\n"
//		"print()\n"
//	);
//
//	if (Py_FinalizeEx() < 0)
//	{
//		exit(120);
//	}
//
//	PyMem_RawFree(program);
//
//	return 0;
//}

//class GA
//{
//public:
//    GA(int argc, char* argv[]);
//    ~GA();
//    void printChromosome();
//    void initialisePopulation();
//    void evaluateChromosome();
//    void parentSelection();
//    void crossover();
//    void mutation();
//    void survivalSelection();
//    void calculateAverageFitness();
//    void recordBestFitness();
//
//private:
//    double chromosome[POP_SIZE][GENE];
//    double fitnessValue[POP_SIZE];
//    int parents[2][GENE];
//};