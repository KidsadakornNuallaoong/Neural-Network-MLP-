#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>

using namespace std;

// #include "../Library/Perceptron/Perceptron.hpp"
#include "../Library/MLP/MLP.hpp"

vector<nlohmann::json> Data_train = {
    {
    {"Data", {
        {"CO2", 0.0},
        {"VOC", 0.0},
        {"RA", 0.0},
        {"TEMP", 0.0},
        {"HUMID", 0.0},
        {"PRESSURE", 0.0}
    }}
},
};

nlohmann::json Prediction_train = {
    {"Prediction", {
        {"Cold", 0.0},
        {"Warm", 0.0},
        {"Hot", 0.0},
        {"Dry", 0.0},
        {"Wet", 0.0},
        {"Normal", 0.0},
        {"Unknown", 0.0}
    }}
};

int main() {
    // vector<int> layersSize = {2, 6, 1};
    // MultiLayerPerceptron<double> mlp(layersSize);

    // vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    // vector<vector<double>> targets = {{0}, {1}, {1}, {0}};

    // mlp.setActivation({"sigmoid", "sigmoid"});
    // mlp.setAccuracy(0.01);

    // double learningRate = 0.01;

    // mlp.predict(inputs, R_D);

    // mlp.train(inputs, targets, learningRate);

    // mlp.display();
    // mlp.predict(inputs, R_D);

    // mlp.export_to_json("mlp_export.json");

    // MultiLayerPerceptron<double> mlp2;

    // cout << "MLP2" << endl;

    // mlp2.import_from_json("mlp_export.json");

    // // mlp2.display();
    // mlp2.predict(inputs, R_D);

    // * set layers size for data train 
    vector<int> layersSize = {6, 200, 7};
    MultiLayerPerceptron<double> mlp(layersSize);

    vector<vector<double>> inputs = {
        {400.0, 0.5, 0.1, 22.0, 45.0, 1013.0},
        {420.0, 0.6, 0.2, 23.0, 50.0, 1012.0},
        {450.0, 0.7, 0.3, 24.0, 55.0, 1011.0},
        {480.0, 0.8, 0.4, 25.0, 60.0, 1010.0},
        {500.0, 0.9, 0.5, 26.0, 65.0, 1009.0}
    };

    // * percentage of each class
    vector<vector<double>> targets = {
        {0.001, 0.001, 0.001, 0.001, 1.0, 0.001, 0.001},
        {0.001, 0.001, 0.001, 0.001, 0.001, 1.0, 0.001},
        {0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 1.0},
        {1.0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001},
        {0.001, 1.0, 0.001, 0.001, 0.001, 0.001, 0.001} 
    };

    mlp.setActivation({"sigmoid", "sigmoid"});
    mlp.setAccuracy(0.01);

    double learningRate = 0.01;

    mlp.train(inputs, targets, learningRate);

    mlp.display();

    cout << "Prediction" << endl;

    mlp.predict(inputs, DISPLAY);

    mlp.export_to_json("mlp_export.json");

    return 0;
}
