#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
using namespace std;

const int GENE = 30;
const int POP_SIZE = 30;
const string ATTRIBUTE[GENE] = { "radius(mean)", "texture(mean)", "perimeter(mean)", "area(mean)", "smoothness(mean)", "compactness(mean)", "cancavity(mean)", "concave points(mean)","symmetry(mean)","fractal dimension(mean)",    
                                 "radius(standard error)", "texture(standard error)", "perimeter(standard error)", "area(standard error)", "smoothness(standard error)", "compactness(standard error)", "cancavity(standard error)", 
                                 "concave points(standard error)","symmetry(standard error)","fractal dimension(standard error)", "radius(worst)", "texture(worst)", "perimeter(worst)", "area(worst)", "smoothness(worst)", 
                                 "compactness(worst)", "cancavity(worst)", "concave points(worst)","symmetry(worst)","fractal dimension(worst)" };

//const string ATTRIBUTE[GENE] = { "time", "radius(mean)", "texture(mean)", "perimeter(mean)", "area(mean)", "smoothness(mean)", "compactness(mean)", "cancavity(mean)", "concave points(mean)","symmetry(mean)","fractal dimension(mean)",
//                                 "radius(standard error)", "texture(standard error)", "perimeter(standard error)", "area(standard error)", "smoothness(standard error)", "compactness(standard error)", "cancavity(standard error)",
//                                 "concave points(standard error)","symmetry(standard error)","fractal dimension(standard error)", "radius(worst)", "texture(worst)", "perimeter(worst)", "area(worst)", "smoothness(worst)",
//                                 "compactness(worst)", "cancavity(worst)", "concave points(worst)","symmetry(worst)","fractal dimension(worst)", "tumor size", "lymph node status" };
const double CROSSOVER_PROBABILITY = 0.9;
const double MUTATION_PROBABILITY = 0.2;
const int MAXIMUM_GENERATION = 10;

int chromosome[POP_SIZE][GENE];
double fitnessValue[POP_SIZE];
int parents[2][GENE];
int children[2][GENE];
int survivor[POP_SIZE][GENE];
int counterSurvival{ 0 };
double averageFitness = 0;
double bestFitness = -1;
int bestChromosome[GENE];
ofstream ACP_File("ACP_File.txt");
ofstream BSF_File("BSF_File.txt");
ofstream bestChromosome_File("bestChromosome_File.txt");

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

void evaluateChromosome(const char* path)
{
    vector<string> choosenAttribute(POP_SIZE);

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
    
    /*wchar_t* program = Py_DecodeLocale(path, NULL);

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
    );*/

    PyRun_SimpleString(
        "accuracyFile = open('accuracy_result.txt','w')\n"
        "accuracyFile.close()\n"
    );
    
    for (int i = 0; i < POP_SIZE; i++)
    {
        PyRun_SimpleString(
            "df = pd.read_csv('wdbc.csv')\n"
            //"df = pd.read_csv('wpbc.csv')\n"
            "df.columns = ['ID number', 'Class', 'radius(mean)', 'texture(mean)', 'perimeter(mean)', 'area(mean)', 'smoothness(mean)', 'compactness(mean)', 'cancavity(mean)', 'concave points(mean)', 'symmetry(mean)', 'fractal dimension(mean)', \\"
            "\n'radius(standard error)', 'texture(standard error)', 'perimeter(standard error)', 'area(standard error)', 'smoothness(standard error)', 'compactness(standard error)', 'cancavity(standard error)', 'concave points(standard error)', 'symmetry(standard error)', 'fractal dimension(standard error)', \\"
            "\n'radius(worst)', 'texture(worst)', 'perimeter(worst)', 'area(worst)', 'smoothness(worst)', 'compactness(worst)', 'cancavity(worst)', 'concave points(worst)', 'symmetry(worst)', 'fractal dimension(worst)']\n"
            
            //"df.columns = ['ID number', 'Class', 'time', 'radius(mean)', 'texture(mean)', 'perimeter(mean)', 'area(mean)', 'smoothness(mean)', 'compactness(mean)', 'cancavity(mean)', 'concave points(mean)', 'symmetry(mean)', 'fractal dimension(mean)', \\"
            //"\n'radius(standard error)', 'texture(standard error)', 'perimeter(standard error)', 'area(standard error)', 'smoothness(standard error)', 'compactness(standard error)', 'cancavity(standard error)', 'concave points(standard error)', 'symmetry(standard error)', 'fractal dimension(standard error)', \\"
            //"\n'radius(worst)', 'texture(worst)', 'perimeter(worst)', 'area(worst)', 'smoothness(worst)', 'compactness(worst)', 'cancavity(worst)', 'concave points(worst)', 'symmetry(worst)', 'fractal dimension(worst)', 'tumor size', 'lymph node status']\n"
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
            "\tif y[c] == 'B' :\n"
            "\t\ty[c] = 0\n"
            "\telif y[c] == 'M' :\n"
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

    /*if (Py_FinalizeEx() < 0)
    {
        exit(120);
    }

    PyMem_RawFree(program);*/

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
    float randomValue;
    int crossoverPoint;

    for (int r = 0; r < 2; r++)
    {
        for (int g = 0; g < GENE; g++)
        {
            children[r][g] = parents[r][g];
        }
    }

    randomValue = (rand() % 10) / 10.0;
    cout << "\n\tRandomValue = " << randomValue;

    if (randomValue < CROSSOVER_PROBABILITY)
    {
        srand(time(0));
        crossoverPoint = rand() % GENE;

        cout << "\n\tCrossover Point = " << crossoverPoint;

        for (int g = crossoverPoint; g < GENE; g++)
        {
            children[0][g] = parents[1][g];
            children[1][g] = parents[0][g];
        }

        cout << "\nResult of crossover:";
        for (int p = 0; p < 2; p++)
        {
            cout << "\nChildren " << p << ": ";

            for (int g = 0; g < GENE; g++)
            {
                cout << children[p][g] << " ";
            }
        }
    }
}

void mutation()
{
    float randomValue;
    int mutationBit;

    for (int c = 0; c < 2; c++)
    {
        randomValue = (rand() % 10) / 10.0;

        cout << "\n  Children " << c << ":";
        cout << "\n\tRandomValue = " << randomValue;

        if (randomValue < MUTATION_PROBABILITY)
        {
            mutationBit = rand() % GENE;

            cout << "\n\tMutation Bit = " << mutationBit;

            if (children[c][mutationBit] == 0)
            {
                children[c][mutationBit] = 1;
            }
            else
            {
                children[c][mutationBit] = 0;
            }
        }
    }

    cout << "\nResult of mutation:";
    for (int p = 0; p < 2; p++)
    {
        cout << "\nChildren " << p << ": ";

        for (int g = 0; g < GENE; g++)
        {
            cout << children[p][g] << " ";
        }
    }
}

void survivalSelection()
{
    for (int c = 0; c < 2; c++)
    {
        for (int g = 0; g < GENE; g++)
        {
            survivor[counterSurvival][g] = children[c][g];
        }

        counterSurvival++;
    }

    cout << "\nSurvivor: \n";

    for (int c = 0; c < POP_SIZE; c++)
    {
        cout << "survivor " << c << " : ";

        for (int g = 0; g < GENE; g++)
        {
            cout << survivor[c][g] << " ";
        }

        cout << endl;
    }
}

void copyChromosome()
{
    for (int c = 0; c < POP_SIZE; c++)
    {
        for (int g = 0; g < GENE; g++)
        {
            chromosome[c][g] = survivor[c][g];
        }
    }
}

void calculateAverageFitness()
{
    double totalFitness = 0;

    for (int i = 0; i < POP_SIZE; i++)
    {
        totalFitness += fitnessValue[i];
    }

    averageFitness = totalFitness / POP_SIZE;

    cout << "\n\tAverage Fitness = " << averageFitness;
    ACP_File << averageFitness << endl;
}

void recordBestFitness()
{
    for (int i = 0; i < POP_SIZE; i++)
    {
        if (fitnessValue[i] > bestFitness)
        {
            bestFitness = fitnessValue[i];

            for (int g = 0; g < GENE; g++)
            {
                bestChromosome[g] = chromosome[i][g];
            }
        }
    }

    cout << "\n\tBest Fitness = " << bestFitness;
    BSF_File << bestFitness << endl;

    cout << "\n\tBest Chromosome = ";

    for (int g = 0; g < GENE; g++)
    {
        cout << bestChromosome[g] << " ";
        bestChromosome_File << bestChromosome[g] << " ";
    }

    bestChromosome_File << endl;
}

int main(int argc, char* argv[])
{
    const char* path = argv[0];

   /* cout << "\nINITIALIZE POPULATION\n";
    initialisePopulation();
    printChromosome();

    cout << "\nFITNESS EVALUATION\n";
    evaluateChromosome(path);

    cout << "\n\nPARENT SELECTION";
    parentSelection();*/

    wchar_t* program = Py_DecodeLocale(path, NULL);

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

    cout << "\nINITIALIZE POPULATION\n";
    initialisePopulation();

    for (int j = 0; j < MAXIMUM_GENERATION; j++)
    {
        cout << "\n***************************GENERATION " << j + 1 << "****************************";
        cout << "\n\nNEW POPULATION\n";
        printChromosome();
        //getchar();

        counterSurvival = 0;

        cout << "\nFITNESS EVALUATION";
        evaluateChromosome(path);
        calculateAverageFitness();
        recordBestFitness();

        for (int i = 0; i < POP_SIZE / 2; i++)
        {
            cout << "\n\nPARENT SELECTION";
            parentSelection();

            cout << "\n\nCROSSOVER";
            crossover();

            cout << "\n\nMUTATION";
            mutation();

            cout << "\n\nSURVIVAL SELECTION";
            survivalSelection();
        }

        copyChromosome();
    }

    cout << "\n\nNEW POPULATION\n";
    printChromosome();

    ACP_File.close();

    if (Py_FinalizeEx() < 0)
    {
        exit(120);
    }

    PyMem_RawFree(program);
}
