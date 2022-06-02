#include <iostream>
#include <string>
#include <fstream>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
using namespace std;

const int GENE = 33;
const int POP_SIZE = 500;
//const string ATTRIBUTE[GENE] = { "radius(mean)", "texture(mean)", "perimeter(mean)", "area(mean)", "smoothness(mean)", "compactness(mean)", "cancavity(mean)", "concave points(mean)","symmetry(mean)","fractal dimension(mean)",    
//                                 "radius(standard error)", "texture(standard error)", "perimeter(standard error)", "area(standard error)", "smoothness(standard error)", "compactness(standard error)", "cancavity(standard error)", 
//                                 "concave points(standard error)","symmetry(standard error)","fractal dimension(standard error)", "radius(worst)", "texture(worst)", "perimeter(worst)", "area(worst)", "smoothness(worst)", 
//                                 "compactness(worst)", "cancavity(worst)", "concave points(worst)","symmetry(worst)","fractal dimension(worst)" };

const string ATTRIBUTE[GENE] = { "time", "radius(mean)", "texture(mean)", "perimeter(mean)", "area(mean)", "smoothness(mean)", "compactness(mean)", "cancavity(mean)", "concave points(mean)","symmetry(mean)","fractal dimension(mean)",
                                 "radius(standard error)", "texture(standard error)", "perimeter(standard error)", "area(standard error)", "smoothness(standard error)", "compactness(standard error)", "cancavity(standard error)",
                                 "concave points(standard error)","symmetry(standard error)","fractal dimension(standard error)", "radius(worst)", "texture(worst)", "perimeter(worst)", "area(worst)", "smoothness(worst)",
                                 "compactness(worst)", "cancavity(worst)", "concave points(worst)","symmetry(worst)","fractal dimension(worst)", "tumor size", "lymph node status" };
const double CROSSOVER_PROBABILITY = 0.9;
const double MUTATION_PROBABILITY = 0.2;
const int MAXIMUM_GENERATION = 30;

int chromosome[POP_SIZE][GENE];
double fitnessValue[POP_SIZE];
int parents[2][GENE];

void initialisePopulation()
{
    srand(time(0));

    for (int c = 0; c < POP_SIZE; c++)
    {
        int sum = 0;

        //do {
            sum = 0;

            for (int g = 0; g < GENE; g++)
            {
                chromosome[c][g] = rand() % 2;
                sum += chromosome[c][g];
            }
        //} while (sum > 2);
    }
}

void evaluateChromosome(int argc, char* argv[])
{
    string choosenAttribute[POP_SIZE];

    for (int c = 0; c < POP_SIZE; c++)
    {
        for (int g = 0; g < GENE; g++)
        {
            if (chromosome[c][g] == 1)
            {
                string str = "'" + ATTRIBUTE[g] + "'" + ",";
                choosenAttribute[c] += str;
            }
        }
    }
    
    wchar_t* program = Py_DecodeLocale(argv[0], NULL);

    if (program == NULL)
    {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    Py_SetProgramName(program);
    Py_Initialize();

    PyRun_SimpleString(
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.metrics import confusion_matrix\n"
        "from sklearn.metrics import roc_curve\n"
        "from sklearn.metrics import roc_auc_score\n"
        "accuracyFile = open('accuracy_result.txt','w')\n"
        "accuracyFile.close()\n"
    );
    
    for (int i = 0; i < POP_SIZE; i++)
    {
        PyRun_SimpleString(
            //"df = pd.read_csv('wdbc.csv')\n"
            "df = pd.read_csv('wpbc.csv')\n"
            //"df.columns = ['ID number', 'Class', 'radius(mean)', 'texture(mean)', 'perimeter(mean)', 'area(mean)', 'smoothness(mean)', 'compactness(mean)', 'cancavity(mean)', 'concave points(mean)', 'symmetry(mean)', 'fractal dimension(mean)', \\"
           // "\n'radius(standard error)', 'texture(standard error)', 'perimeter(standard error)', 'area(standard error)', 'smoothness(standard error)', 'compactness(standard error)', 'cancavity(standard error)', 'concave points(standard error)', 'symmetry(standard error)', 'fractal dimension(standard error)', \\"
           // "\n'radius(worst)', 'texture(worst)', 'perimeter(worst)', 'area(worst)', 'smoothness(worst)', 'compactness(worst)', 'cancavity(worst)', 'concave points(worst)', 'symmetry(worst)', 'fractal dimension(worst)']\n"
            
            "df.columns = ['ID number', 'Class', 'time', 'radius(mean)', 'texture(mean)', 'perimeter(mean)', 'area(mean)', 'smoothness(mean)', 'compactness(mean)', 'cancavity(mean)', 'concave points(mean)', 'symmetry(mean)', 'fractal dimension(mean)', \\"
            "\n'radius(standard error)', 'texture(standard error)', 'perimeter(standard error)', 'area(standard error)', 'smoothness(standard error)', 'compactness(standard error)', 'cancavity(standard error)', 'concave points(standard error)', 'symmetry(standard error)', 'fractal dimension(standard error)', \\"
            "\n'radius(worst)', 'texture(worst)', 'perimeter(worst)', 'area(worst)', 'smoothness(worst)', 'compactness(worst)', 'cancavity(worst)', 'concave points(worst)', 'symmetry(worst)', 'fractal dimension(worst)', 'tumor size', 'lymph node status']\n"
            "df = df.drop(columns = 'ID number')\n"
            "df = df.replace('?', np.NaN)\n"
            "df = df.dropna()\n"
        );

        string str = "features = [" + choosenAttribute[i] + "]\n";

        PyRun_SimpleString(str.c_str());

        PyRun_SimpleString(
            "X = df[features].values\n"
            "y = df['Class'].values\n"
            "for c in range(len(y)):\n"
            "\tif y[c] == 'R' :\n"
            "\t\ty[c] = 0\n"
            "\telif y[c] == 'N' :\n"
            "\t\ty[c] = 1\n"
            "y = y.astype('int')\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42, stratify = y)\n"
            "neighbors = np.arange(1, 9)\n"
            "train_accuracy = np.empty(len(neighbors))\n"
            "test_accuracy = np.empty(len(neighbors))\n"
            "for i, k in enumerate(neighbors) :\n"
            "\tknn = KNeighborsClassifier(n_neighbors = k)\n"
            "\tknn.fit(X_train, y_train)\n"
            "\ttrain_accuracy[i] = knn.score(X_train, y_train)\n"
            "\ttest_accuracy[i] = knn.score(X_test, y_test)\n"
            //"knn = KNeighborsClassifier(n_neighbors = 10)\n"
            //"knn.fit(X_train, y_train)\n"
            //"accuracy = knn.score(X_test, y_test)\n"
            "model = RandomForestClassifier(n_estimators = 100, random_state = 0, n_jobs = -1)\n"
            "model.fit(X_train, y_train)\n"
            "y_pred = model.predict(X_test)\n"
            "accuracy = model.score(X_test, y_test)\n"
            "with open('accuracy_result.txt', 'a') as f :\n"
            "\tf.write(f'{accuracy}\\n')\n"
        );
    }

    if (Py_FinalizeEx() < 0)
    {
        exit(120);
    }

    PyMem_RawFree(program);

    ifstream fitnessFile("accuracy_result.txt");

    for (int i = 0; fitnessFile; i++)
    {
        fitnessFile >> fitnessValue[i];
    }

    fitnessFile.close();

    for (int c = 0; c < POP_SIZE; c++)
    {
        cout << "\n\tChr " << c + 1 << "\t: " << "fitness value: " << fitnessValue[c];
    }
}

void printChromosome()
{
    for (int c = 0; c < POP_SIZE; c++)
    {
        cout << "Chr " << c + 1 << "\t: ";

        for (int g = 0; g < GENE; g++)
        {
            cout << chromosome[c][g] << ' ';
        }
        cout << '\n';
    }
}

void parentSelection()
{
    // tournament selection 2 player

    int indexplayer1, indexplayer2;
    int winnerIndex[2];

    do
    {
        for (int p = 0; p < 2; p++)	// 2 players
        {
            indexplayer1 = rand() % POP_SIZE;

            do
            {
                indexplayer2 = rand() % POP_SIZE;

            } while (indexplayer1 == indexplayer2);

            cout << "\n\tround " << p + 1 << ":";
            cout << "\n\t player 1 :" << indexplayer1;
            cout << "\n\t player 2 :" << indexplayer2;

            if (fitnessValue[indexplayer1] > fitnessValue[indexplayer2])
            {
                winnerIndex[p] = indexplayer1;
            }
            else
            {
                winnerIndex[p] = indexplayer2;
            }

            cout << "\nwinner :" << winnerIndex[p];
            for (int g = 0; g < GENE; g++)
            {
                parents[p][g] = chromosome[winnerIndex[p]][g];
            }
        }// end tournament

    } while (winnerIndex[0] == winnerIndex[1]);

    cout << "\n\nResult:";
    for (int p = 0; p < 2; p++)
    {
        cout << "\n\tparent " << p << ": ";

        for (int g = 0; g < GENE; g++)
        {
            cout << parents[p][g] << " ";
        }
    }
}

void crossover()
{
    
}

void mutation()
{

}

void survivalSelection()
{

}

void calculateAverageFitness()
{

}

void recordBestFitness()
{

}

int main(int argc, char* argv[])
{
    cout << "\nINITIALIZE POPULATION\n";
    initialisePopulation();
    printChromosome();

    cout << "\nFITNESS EVALUATION\n";
    evaluateChromosome(argc, argv);

    cout << "\n\nPARENT SELECTION";
    parentSelection();
}
