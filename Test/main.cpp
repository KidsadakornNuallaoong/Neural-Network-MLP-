#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// #include "../Library/Perceptron/Perceptron.hpp"
#include "../Library/MLP/MLP.hpp"

int main() {
    vector<int> layersSize = {2, 6, 1};
    MultiLayerPerceptron<double> mlp(layersSize);

    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};

    mlp.setActivation({"sigmoid", "sigmoid"});
    mlp.setAccuracy(0.01);

    double learningRate = 0.01;

    mlp.predict(inputs, R_D);

    mlp.train(inputs, targets, learningRate);

    mlp.display();
    mlp.predict(inputs, R_D);

    mlp.export_to_json("mlp_export.json");

    MultiLayerPerceptron<double> mlp2;

    cout << "MLP2" << endl;

    mlp2.import_from_json("mlp_export.json");

    // mlp2.display();
    mlp2.predict(inputs, R_D);

    return 0;
}
