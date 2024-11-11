#include <iostream>
#include <vector>

using namespace std;

// #include "../Library/Perceptron/Perceptron.hpp"
#include "./Library/MLP/MLP.hpp"

vector<int> text_to_binary(string text){
    vector<int> binary;
    for (int i = 0; i < int(text.size()); i++){
        for (int j = 0; j < 8; j++){
            binary.push_back((text[i] >> j) & 1);
        }
    }
    return binary;
}

string binary_to_text(vector<int> binary){
    string text = "";
    for (int i = 0; i < int(binary.size()); i += 8){
        int c = 0;
        for (int j = 0; j < 8; j++){
            c += binary[i + j] << j;
        }
        text += c;
    }
    return text;
}

int count_max_size(vector<vector<int>> binary_texts){
    int max_size = 0;
    for (int i = 0; i < int(binary_texts.size()); i++){
        if (int(binary_texts[i].size()) > max_size){
            max_size = int(binary_texts[i].size());
        }
    }
    return max_size;
}

vector<int> padding(vector<int> binary_text, int max_size){
    vector<int> padded_text = binary_text;
    for (int i = 0; i < max_size - int(binary_text.size()); i++){
        padded_text.push_back(0);
    }
    return padded_text;
}

vector<int> double_to_int(vector<double> double_vector){
    vector<int> int_vector;
    for (int i = 0; i < int(double_vector.size()); i++){
        int_vector.push_back(round(double_vector[i]));
    }
    return int_vector;
}

vector<double> int_to_double(vector<int> int_vector){
    vector<double> double_vector;
    for (int i = 0; i < int(int_vector.size()); i++){
        double_vector.push_back(double(int_vector[i]));
    }
    return double_vector;
}

void add_noise(vector<vector<double>>& img, double noise_level) {
    srand(time(0)); // Seed for random number generation

    for (auto& image : img) {
        for (auto& pixel : image) {
            // Add random noise within the range [-noise_level, noise_level]
            double noise = ((rand() / double(RAND_MAX)) * 2 - 1) * noise_level;
            pixel += noise;
        }
    }
}

int main() {

    vector<string> texts = {"zero", "zero", "one", "two"};

    vector<vector<int>> binary_texts;

    for (int i = 0; i < int(texts.size()); i++){
        binary_texts.push_back(text_to_binary(texts[i]));
    }

    // * padding
    int max_size = count_max_size(binary_texts);
    for (int i = 0; i < int(binary_texts.size()); i++){
        binary_texts[i] = padding(binary_texts[i], max_size);
    }

    // * convert binary to text
    // for (int i = 0; i < int(binary_texts.size()); i++){
    //     cout <<  << endl;
    // }

    // * display binary texts
    // for (int i = 0; i < int(binary_texts.size()); i++){
    //     for (int j = 0; j < int(binary_texts[i].size()); j++){
    //         cout << binary_texts[i][j];
    //     }
    //     cout << endl;
    // }
    
    // * display binary texts
    // for (int i = 0; i < int(binary_texts.size()); i++){
    //     for (int j = 0; j < int(binary_texts[i].size()); j++){
    //         cout << binary_texts[i][j];
    //     }
    //     cout << endl;
    // }

    // * change vector int to vector double
    vector<vector<double>> binary_texts_double;
    for (int i = 0; i < int(binary_texts.size()); i++){
        binary_texts_double.push_back(int_to_double(binary_texts[i]));
    }

    // * display binary texts double
    // for (int i = 0; i < int(binary_texts_double.size()); i++){
    //     for (int j = 0; j < int(binary_texts_double[i].size()); j++){
    //         cout << binary_texts_double[i][j];
    //     }
    //     cout << endl;
    // }
    
    // * 0, 1, 2, 3, 4
    vector<vector<double>> img = {
        // 0
        {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0},

        // 0
        {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 0, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0},

        // 1
        {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0},

        // 2
        {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0},

        // 3
        {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0},

        // 4
        {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 0, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0},

        // 5
        {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0},

        // 6
        {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 0, 0, 0,
        0, 1, 1, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0},

        // 7
        {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0},

        // 8
        {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 0, 0, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 0, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0},

        // 9
        {0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 0, 0, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0},
    };


    // * XOR Problem
    // vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> inputs = img;
    vector<vector<double>> targets = {{0}, {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};

    // vector<vector<double>> targets = {
    //     binary_texts_double[0],
    //     binary_texts_double[1],
    //     binary_texts_double[2],
    //     binary_texts_double[3]
    // };

    cout << "Max size: " << count_max_size(binary_texts) << endl;

    // * Test Multi-Layer Perceptron
    vector<int> layersSize = {8*8, 32, 1};
    MultiLayerPerceptron<double> mlp(layersSize);

    mlp.setActivation({"sigmoid", "linear"});
    mlp.setAccuracy(0.01);
    // mlp.display();

    double learningRate = 0.01;

    // cout << "Initial outputs:" << endl;
    mlp.predict(inputs, NONE);

    // cout << "Training..." << endl;

    mlp.train(inputs, targets, learningRate);
    // mlp.display();

    // cout << "Initial outputs:" << endl;
    // vector<vector<double>> inputs_test = {
    //     // * 1
    //     {0.1, 0, 0, 0.05, 0, 0, 0, 0,
    //     0, 0, 0, 0.02, 1, 0.01, 0, 0.03,
    //     0, 0, 0, 1, 1, 0.02, 0, 0,
    //     0.05, 0, 0, 0, 1, 0, 0.04, 0,
    //     0, 0.03, 0, 0, 1, 0, 0, 0.01,
    //     0, 0, 0, 0, 1, 0.03, 0, 0,
    //     0, 0.02, 0, 1, 1, 1, 0, 0.04,
    //     0, 0, 0.05, 0.01, 0, 0.03, 0, 0},

    //     // * 1
    //     {0, 0.02, 0, 0, 0, 0.03, 0, 0,
    //     0, 0.04, 0, 0.01, 1, 0.02, 0.5, 0.03,
    //     0.01, 0, 0, 1, 1, 0.05, 0, 0.01,
    //     0.03, 0.5, 0.02, 0.04, 1, 0, 0.03, 0,
    //     0, 0.01, 0.03, 0.02, 1, 0, 0, 0.04,
    //     0.02, 0.04, 1, 0, 1, 0, 0, 0.01,
    //     0.01, 0.2, 0.03, 1, 1, 1, 0, 0.02,
    //     0.04, 0.01, 0.03, 0, 0, 0.02, 0.01, 0},

    //     // * 2
    //     {0, 0.01, 0, 0.02, 0, 0.01, 0.03, 0,
    //     0, 0, 0.03, 1, 1, 0.02, 0.01, 0.04,
    //     0.01, 0, 1, 1, 1, 1, 0.05, 0.01,
    //     0, 0.04, 0, 0, 1, 1, 0.02, 0,
    //     0.03, 0.01, 0, 1, 1, 0.03, 0.04, 0.02,
    //     0, 0.02, 1, 1, 0, 0, 0.01, 0.03,
    //     0.01, 0.02, 1, 1, 1, 1, 0.02, 0.03,
    //     0, 0, 0.01, 0.04, 0, 0.01, 0, 0.02},

    //     // * 2
    //     {0.02, 0, 0.01, 0, 0.03, 0, 0, 0.01,
    //     0, 0.03, 0.02, 1, 1, 0.02, 0.01, 0.2,
    //     0.04, 0.02, 1, 1, 1, 1, 0.03, 0.01,
    //     0.01, 0.01, 0.02, 0, 1, 1, 0.04, 0,
    //     0.01, 0.03, 0.01, 1, 1, 0.01, 0.2, 0.03,
    //     0.02, 0.02, 1, 1, 0.04, 0.03, 0.01, 0,
    //     0.03, 0.01, 1, 1, 1, 1, 0.04, 0,
    //     0, 0.02, 0, 0.01, 0.02, 0.01, 0, 0},
    // };

    vector<vector<double>> inputs_test = img;

    add_noise(inputs_test, 0.01723);

    // for (const auto& image : inputs_test) {
    //     for (size_t i = 0; i < image.size(); ++i) {
    //         cout << image[i] << " ";
    //         if ((i + 1) % 8 == 0) cout << endl;
    //     }
    //     cout << endl;
    // }

    vector<vector<double>> result = mlp.predict(inputs_test, ROUND);

    // mlp.display();

    for (int i = 0; i < int(result.size()); i++){
        cout << "Result: ";
        for (int j = 0; j < int(result[i].size()); j++){
            cout << result[i][j] << " ";
        }
        cout << endl;
    }

    mlp.export_to_json("model.json");

    MultiLayerPerceptron<double> mlp2("model.json");
    // mlp2.display();

    cout << endl;

    cout << "Result Import Model" << endl;

    mlp2.display();

    mlp2.predict(inputs_test, ROUND);

    vector<vector<double>> result2 = mlp2.predict(inputs_test, ROUND);

    for (int i = 0; i < int(result2.size()); i++){
        cout << "Result: ";
        for (int j = 0; j < int(result2[i].size()); j++){
            cout << result2[i][j] << " ";
        }
        cout << endl;
    }


    return 0;
}
