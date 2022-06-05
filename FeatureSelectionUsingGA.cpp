#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
using namespace std;

const int GENE = 34;
const int POP_SIZE = 30;
const string ATTRIBUTE[GENE] = { "erythema", "scaling", "definite_borders","itching","koebner_phenomenon","polygonal_papules","follicular_papules","oral_mucosal_involvement","knee_and_elbow_involvement","scalp_involvement",
                                 "family_history","melanin_incontinence","eosinophils_in_the_infiltrate","PNL_infiltrate","fibrosis_of_the_papillary_dermis","exocytosis","acanthosis","hyperkeratosis","parakeratosis","clubbing_of_the_rete_ridges",
                                 "elongation_of_the_rete_ridges","thinning_of_the_suprapapillary_epidermis","spongiform_pustule","munro_microabcess","focal_hypergranulosis","disappearance_of_the_granular_layer","vacuolisation_and_damage_of_basal_layer","spongiosis","saw-tooth_appearance_of_rete","follicular_horn_plug",
                                 "perifollicular_parakeratosis","inflammatory_monoluclear_inflitrate","band-like_infiltrate","age"};
const double CROSSOVER_PROBABILITY = 0.8;
const double MUTATION_PROBABILITY = 0.2;
const int MAXIMUM_GENERATION = 50;

//int chromosome[POP_SIZE][GENE];
//int sumChromosome[POP_SIZE];
//double fitnessValue[POP_SIZE];
//int parents[2][GENE];
//int children[2][GENE];
//int survivor[POP_SIZE][GENE];
//int counterSurvival{ 0 };
//double averageFitness = 0;
//double bestFitness = -1;
//int bestChromosome[GENE];
ofstream ACP_File("ACP_File.txt");
ofstream BSF_File("BSF_File.txt");
ofstream bestChromosome_File("bestChromosome_File.txt");

class GA
{
public:
    GA(const char* path);
    ~GA();
    void printChromosome();
    void initialisePopulation();
    void evaluateChromosome();
    void parentSelection();
    void crossover();
    void mutation();
    void survivalSelection();
    void copyChromosome();
    void calculateAverageFitness();
    void recordBestFitness();
    int counterSurvival{ 0 };
    
private:
    int chromosome[POP_SIZE][GENE];
    int sumChromosome[POP_SIZE];
    double fitnessValue[POP_SIZE];
    int parents[2][GENE];
    int children[2][GENE];
    int survivor[POP_SIZE][GENE];
    double averageFitness = 0;
    double bestFitness = -1;
    int bestChromosome[GENE];

    wchar_t* program;
    void initializePython(const char* path);
    void closePython();
};

int main(int argc, char* argv[])
{
    const char* path = argv[0];

    GA featureSelectionGA(path);
   /* cout << "\nINITIALIZE POPULATION\n";
    initialisePopulation();
    printChromosome();

    cout << "\nFITNESS EVALUATION\n";
    evaluateChromosome(path);

    cout << "\n\nPARENT SELECTION";
    parentSelection();*/

    cout << "\nINITIALIZE POPULATION\n";
    featureSelectionGA.initialisePopulation();

    for (int j = 0; j < MAXIMUM_GENERATION; j++)
    {
        cout << "\n***************************GENERATION " << j + 1 << "****************************";
        cout << "\n\nNEW POPULATION\n";
        featureSelectionGA.printChromosome();
        //getchar();

        featureSelectionGA.counterSurvival = 0;

        cout << "\nFITNESS EVALUATION";
        featureSelectionGA.evaluateChromosome();
        featureSelectionGA.calculateAverageFitness();
        featureSelectionGA.recordBestFitness();

        for (int i = 0; i < POP_SIZE / 2; i++)
        {
            cout << "\n\nPARENT SELECTION";
            featureSelectionGA.parentSelection();

            cout << "\n\nCROSSOVER";
            featureSelectionGA.crossover();

            cout << "\n\nMUTATION";
            featureSelectionGA.mutation();

            cout << "\n\nSURVIVAL SELECTION";
            featureSelectionGA.survivalSelection();
        }

        featureSelectionGA.copyChromosome();
    }

    cout << "\n\nNEW POPULATION\n";
    featureSelectionGA.printChromosome();
}

GA::GA(const char* path)
{
    initializePython(path);
}

GA::~GA()
{
    closePython();
    ACP_File.close();
    BSF_File.close();
    bestChromosome_File.close();
}

void GA::printChromosome()
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

void GA::initialisePopulation()
{
    srand(time(0));

    for (int c = 0; c < POP_SIZE; c++)
    {
        int sum = 0;

       // do {
            sum = 0;

            for (int g = 0; g < GENE; g++)
            {
                chromosome[c][g] = rand() % 2;
                sum += chromosome[c][g];
            }

            sumChromosome[c] = sum;
       // } while (sum > 10);
    }
}

void GA::evaluateChromosome()
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

    PyRun_SimpleString(
        "accuracyFile = open('accuracy_result.txt','w')\n"
        "accuracyFile.close()\n"
    );

    for (int i = 0; i < POP_SIZE; i++)
    {
        PyRun_SimpleString(
            "df = pd.read_csv('dermatology.csv')\n"
            "df.columns = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 'polygonal_papules', 'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement', 'scalp_involvement', \\"
            "\n'family_history', 'melanin_incontinence', 'eosinophils_in_the_infiltrate', 'PNL_infiltrate', 'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges', \\"
            "\n'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule', 'munro_microabcess', 'focal_hypergranulosis', 'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer', 'spongiosis', 'saw-tooth_appearance_of_rete', 'follicular_horn_plug',   \\"
            "\n'perifollicular_parakeratosis', 'inflammatory_monoluclear_inflitrate', 'band-like_infiltrate', 'age','Class']\n"

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
            "knn = KNeighborsClassifier(n_neighbors = 10)\n"
            "knn.fit(X_train, y_train)\n"
            "accuracy = knn.score(X_test, y_test)\n"
            "with open('accuracy_result.txt', 'a') as f :\n"
            "\tf.write(f'{accuracy}\\n')\n"
        );
    }

    ifstream fitnessFile("accuracy_result.txt");

    for (int i = 0; fitnessFile; i++)
    {
        double accuracy;
        //fitnessFile >> fitnessValue[i];
        fitnessFile >> accuracy;

        //fitnessValue[i] = (accuracy + (1 / (sumChromosome[i] + 0.01))) / 2.0;
        fitnessValue[i] = accuracy * ((static_cast<double>(GENE) - sumChromosome[i]) / (GENE));
    }

    fitnessFile.close();

    for (int c = 0; c < POP_SIZE; c++)
    {
        cout << "\n\tChr " << c + 1 << "\t: " << "fitness value: " << fitnessValue[c];
    }
}

void GA::parentSelection()
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

void GA::crossover()
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

void GA::mutation()
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

void GA::survivalSelection()
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

void GA::copyChromosome()
{
    for (int c = 0; c < POP_SIZE; c++)
    {
        for (int g = 0; g < GENE; g++)
        {
            chromosome[c][g] = survivor[c][g];
        }
    }
}

void GA::calculateAverageFitness()
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

void GA::recordBestFitness()
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

void GA::initializePython(const char* path)
{
    program = Py_DecodeLocale(path, NULL);

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
    );
}

void GA::closePython()
{
    if (Py_FinalizeEx() < 0)
    {
        exit(120);
    }

    PyMem_RawFree(program);
}
