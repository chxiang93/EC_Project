#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
using namespace std;

// Declaration of hyperparameter of the genetic algorithm
const int GENE = 34;
const string ATTRIBUTE[GENE] = { "erythema", "scaling", "definite_borders","itching","koebner_phenomenon","polygonal_papules","follicular_papules","oral_mucosal_involvement","knee_and_elbow_involvement","scalp_involvement",
                                 "family_history","melanin_incontinence","eosinophils_in_the_infiltrate","PNL_infiltrate","fibrosis_of_the_papillary_dermis","exocytosis","acanthosis","hyperkeratosis","parakeratosis","clubbing_of_the_rete_ridges",
                                 "elongation_of_the_rete_ridges","thinning_of_the_suprapapillary_epidermis","spongiform_pustule","munro_microabcess","focal_hypergranulosis","disappearance_of_the_granular_layer","vacuolisation_and_damage_of_basal_layer","spongiosis","saw-tooth_appearance_of_rete","follicular_horn_plug",
                                 "perifollicular_parakeratosis","inflammatory_monoluclear_inflitrate","band-like_infiltrate","age"};
const int POP_SIZE = 50;
const double CROSSOVER_PROBABILITY = 0.9;
const double MUTATION_PROBABILITY = 0.2;
const int MAXIMUM_GENERATION = 30;
ofstream ACP_File("ACP_File.txt");
ofstream BSF_File("BSF_File.txt");
ofstream bestChromosome_File("bestChromosome_File.txt");

// Declaration of class GA which consist of the whole process of the genetic algorithm
class GA
{
public:
    GA(const char* path);   // Constructor that accept a path parameter to initialize python environment
    ~GA();                  // Destructor
    void printChromosome(); // Print all the chromosome in the population
    void initialisePopulation();    // Initialise the population of chromosome randomly
    void evaluateChromosome();      // Evaluate the fitness of every chromosome 
    void parentSelection();         // Tournament selection with 3 players to select parents
    void crossover();               // 2-point crossover to produce children
    void mutation();                // Bit-flip mutation on random 2 gene in a chromosome
    void survivalSelection();       // All children replace parents strategy
    void copyChromosome();          // Copy all the survivor to the population of chromosome for next generation
    void calculateAverageFitness(); // Calculate the average fitness (ACP)
    void recordBestFitness();       // Record BSF and best chromosome

    int counterSurvival{ 0 };       // Counter that keep track of number of surivor
    
private:
    int chromosome[POP_SIZE][GENE]; // Chromosome
    int sumAttribute[POP_SIZE];     // Number of attribute selected for every chromosome
    double fitnessValue[POP_SIZE];  // Fitness value of every chromosome
    int parents[2][GENE];           // Parents
    int children[2][GENE];          // Children
    int survivor[POP_SIZE][GENE];   // Survivor
    double averageFitness = 0;      // ACP
    double bestFitness = -1;        // BSF
    int bestChromosome[GENE];       // Best chromosome

    wchar_t* program;               // Python program
    void initializePython(const char* path);    // Function to initialize python environment
    void closePython();             // Function to clean up python
};

// Main function
int main(int argc, char* argv[])
{
    // Path that contain the solution file
    const char* path = argv[0];

    // Instantiate a GA object for featureSelection
    GA featureSelectionGA(path);
   
    // Initialize population
    cout << "\nINITIALIZE POPULATION\n";
    featureSelectionGA.initialisePopulation();

    // Iteration until MAXIMUM_GENERATION
    for (int j = 0; j < MAXIMUM_GENERATION; j++)
    {
        cout << "\n********************************GENERATION " << j + 1 << "*************************************";
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
        int sum;

		sum = 0;

		for (int g = 0; g < GENE; g++)
		{
			chromosome[c][g] = rand() % 2;
			sum += chromosome[c][g];
		}

		sumAttribute[c] = sum;
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

    ifstream accuracyFile("accuracy_result.txt");

    for (int i = 0; i < POP_SIZE; i++)
    {
        double accuracy;
        
        accuracyFile >> accuracy;

        fitnessValue[i] = (accuracy * 0.7) + (((static_cast<double>(GENE) - sumAttribute[i]) / (GENE)) * 0.3);
        
        //fitnessValue[i] = accuracy * ((static_cast<double>(GENE) - sumAttribute[i]) / (GENE));
    }

    accuracyFile.close();

    for (int c = 0; c < POP_SIZE; c++)
    {
        cout << "\n\tChr " << c + 1 << "\t: " << "fitness value: " << fitnessValue[c];
    }
}

void GA::parentSelection()
{
    // tournament selection 3 player

    int indexplayer1, indexplayer2, indexplayer3;
    int winnerIndex[2] = {};

    do
    {
        for (int p = 0; p < 2; p++)	// 2 parents
        {
            indexplayer1 = rand() % POP_SIZE;

            do
            {
                indexplayer2 = rand() % POP_SIZE;
                indexplayer3 = rand() % POP_SIZE;

            } while ((indexplayer1 == indexplayer2) || (indexplayer1 == indexplayer3) || (indexplayer2 == indexplayer3));

            cout << "\n\tRound " << p + 1 << ":";
            cout << "\n\t Player 1 :" << indexplayer1;
            cout << "\n\t Player 2 :" << indexplayer2;
            cout << "\n\t Player 3 :" << indexplayer3;

            if (fitnessValue[indexplayer1] >= fitnessValue[indexplayer2]) {		// Check if fitness value for player1 is greater than player2

                if (fitnessValue[indexplayer1] >= fitnessValue[indexplayer3])	// If yes, then compare player 1 with player3. Player2 does not need to compare
                {											// because it is lesser than player1
                    winnerIndex[p] = indexplayer1;
                }
                else	  // If player1 lesser than player3, the player3 will be the winner
                {
                    winnerIndex[p] = indexplayer3;
                }
            }
            else if (fitnessValue[indexplayer1] < fitnessValue[indexplayer2]) { // Check if fitness value for player1 is lesser than player2

                if (fitnessValue[indexplayer2] >= fitnessValue[indexplayer3])  // If yes, then compare player 2 with player3. Player1 does not need to compare
                {										   // because it is lesser than player2
                    winnerIndex[p] = indexplayer2;
                }
                else   // If player2 lesser than player3, the player3 will be the winner
                {
                    winnerIndex[p] = indexplayer3;
                }
            }

            cout << "\n\tPlayers: " << indexplayer1 << " vs " << indexplayer2 << " vs " << indexplayer3;
            cout << "\tFitness: " << fitnessValue[indexplayer1] << " vs " << fitnessValue[indexplayer2] << " vs " << fitnessValue[indexplayer3];
            cout << "\n\tWinner - Parents: " << winnerIndex[p] << "\tFitness: " << fitnessValue[winnerIndex[p]];
            cout << endl;

            for (int g = 0; g < GENE; g++)
            {
                parents[p][g] = chromosome[winnerIndex[p]][g];
            }
        }// end tournament

    } while (winnerIndex[0] == winnerIndex[1]);

    cout << "\n\nResult:";
    for (int p = 0; p < 2; p++)
    {
        cout << "\n\tParent " << p + 1 << ": ";

        for (int g = 0; g < GENE; g++)
        {
            cout << parents[p][g] << " ";
        }
    }
}

void GA::crossover()
{
    // 2 point crossover
    float randomValue;
    int crossover_point;
    int crossover_point2;

    for (int r = 0; r < 2; r++)
    {
        for (int g = 0; g < GENE; g++)
        {
            children[r][g] = parents[r][g];
        }
    }

    randomValue = (rand() % 34) / 34.0;
    cout << "\n\tRandomValue = " << randomValue;

    if (randomValue < CROSSOVER_PROBABILITY)
    {
        //srand(time(0));
        crossover_point = rand() % GENE;

        do
        {
            crossover_point2 = rand() % GENE;

        } while (crossover_point == crossover_point2);

        if (crossover_point > crossover_point2) {
            int tmp = crossover_point;
            crossover_point = crossover_point2;
            crossover_point2 = tmp;
        }

        cout << "\n\tCrossover Point 1 = " << crossover_point;
        cout << "\n\tCrossover Point 2 = " << crossover_point2;

        for (int c = crossover_point; c <= crossover_point2; c++)
        {
            children[0][c] = parents[1][c];
            children[1][c] = parents[0][c];
        }

        cout << "\nResult of crossover:";
        for (int c = 0; c < 2; c++)
        {
            cout << "\nChildren " << c + 1 << ": ";

            for (int g = 0; g < GENE; g++)
            {
                cout << children[c][g] << " ";
            }
        }
    }
}

void GA::mutation()
{
    // bit flip mutation (2 genes per chromosome)
    float randomValue;
    int mutationBit;
    int mutationBit2;

    for (int c = 0; c < 2; c++)
    {
        randomValue = (rand() % 34 / 34.0);

        cout << "\nChildren " << c + 1 << ":";
        cout << "\n\tRandomValue = " << randomValue;

        if (randomValue < MUTATION_PROBABILITY)
        {
            mutationBit = rand() % GENE;

            do
            {
                mutationBit2 = rand() % GENE;

            } while (mutationBit == mutationBit2);

            cout << "\n\tMutation Bit 1 = " << mutationBit;
            cout << "\n\tMutation Bit 2 = " << mutationBit2;

            if (children[c][mutationBit] == 1)
            {
                children[c][mutationBit] = 0;
            }
            else
            {
                children[c][mutationBit] = 1;
            }

            if (children[c][mutationBit2] == 1)
            {
                children[c][mutationBit2] = 0;
            }
            else
            {
                children[c][mutationBit2] = 1;
            }//if check bit
        }//if random value
    }//for 2 children

    cout << "\nResult of mutation:";
    for (int c = 0; c < 2; c++)
    {
        cout << "\n\tChildren " << c + 1 << ": ";

        for (int g = 0; g < GENE; g++)
        {
            cout << children[c][g] << " ";
        }
    }
}

void GA::survivalSelection()
{
    // All children replace parents
    for (int c = 0; c < 2; c++)
    {
        for (int g = 0; g < GENE; g++)//copy survivor from children
        {
            survivor[counterSurvival][g] = children[c][g];
        }

        counterSurvival++;
    }

    cout << "\nSurvivor: \n";

    //2. Update array counter
    for (int c = 0; c < POP_SIZE; c++)
    {
        cout << "\tSurvivor " << c + 1 << " : ";

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
