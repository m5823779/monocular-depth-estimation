# define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <sstream>
#include <filesystem>
#include <experimental/filesystem>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include <stdlib.h>
#include <Shlobj.h>
#include <chrono>

#include "infer_engine.h"

using namespace cv;
using namespace std;

// inference device
std::string infer_device = "CPU"; // CPU, GPU, VPUX

// model path
std::string model_path;
std::string cache_path = "../cache";

// input shape
int input_w;
int input_h;
int input_c = 3;

int model_id;
string user_input_str;
string input_folder = "../input";

void UserInput();
void PrintConfig();
void CVCamera();
void CVImage();

// color normalization
std::vector<float> color_norm_mean({ 0.485, 0.456, 0.406 });
std::vector<float> color_norm_std({ 0.229, 0.224, 0.225 });

// parameters for drawing depth map
float max_val = (pow(2, 8)) - 1;

bool model_status = false;

Mat frame, blob_image, show;
InferEngine* ai_core = nullptr;

// entry points
int main(int argc, char* argv[]) {
    UserInput();
    PrintConfig();

    std::vector<int> input_shape({ 1, input_c, input_h, input_w });

    // CV----------------------------------------------------------------------------
    // init inference engine 
    ai_core = new InferEngine(model_path, cache_path, infer_device, input_shape);
    
    if (input_folder == "")
        CVCamera();
    else
        CVImage();

    return EXIT_SUCCESS;
}

void UserInput() {
    // set input image folder
    std::cout << "Enter input image folder directory (press enter to use camera): ";
    std::getline(std::cin, input_folder);

    // check folder
    struct stat buffer;
    if (input_folder != "" && stat(input_folder.c_str(), &buffer) != 0) {
        printf("Input folder '%s' is not exit, please check ...", input_folder);
        return;
    }

    // set model
    std::vector<std::filesystem::path> model_list;
    for (const auto& file : std::filesystem::recursive_directory_iterator("../")) {
        if (file.path().extension().string() == ".xml" && file.path().filename().string() != "plugins.xml") model_list.push_back(file);
    }

    if (!model_list.size()) {
        printf("Can not find any IR model (.xml) please put it in root folder\n");
        return;
    }
    else if (model_list.size() == 1) {
        model_path = model_list[0].string();
    }
    else {
        std::cout << "Find model: \n";
        for (int i = 0; i < model_list.size(); ++i) {
            printf("(%d): %s\n", i, model_list[i].string().c_str());
        }
        std::cout << "Choose model: ";
        std::getline(std::cin, user_input_str);
        model_id = (user_input_str == "") ? 0 : stoi(user_input_str);
        model_path = model_list[min(model_id, model_list.size())].string();
    }

    // set model input shape
    std::cout << "Input model width  [default 256]: ";
    std::getline(std::cin, user_input_str);
    input_w = (user_input_str == "") ? 256 : stoi(user_input_str);

    std::cout << "Input model height [default 256]: ";
    std::getline(std::cin, user_input_str);
    input_h = (user_input_str == "") ? 256 : stoi(user_input_str);


    if (input_w % 32 != 0) {  // Input shape for model must be a multiple of 32
        input_w = input_w - (input_w % 32);
    }

    if (input_h % 32 != 0) {  // Input shape for model must be a multiple of 32
        input_h = input_h - (input_h % 32);
    }

    // Select adapter
    InferenceEngine::Core ie;
    std::vector<std::string> devices = ie.GetAvailableDevices();
    std::cout << "Find alailable devices: \n";
    for (int i = 0; i < devices.size(); ++i) printf("(%d): %s\n", i, devices[i].c_str());
    
    printf("Choose device [default CPU]: ");
    std::getline(std::cin, user_input_str);
    infer_device = (user_input_str == "") ? "CPU" : user_input_str;
}

void PrintConfig() {
    std::cout << "\nConfiguration: " << std::endl;
    std::cout << "Model Input Shape: ( 1 x 3 x " << input_h << " x " << input_w << ")" << std::endl;
    std::cout << std::endl;
}

void CVCamera() {
    // set camera 
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    while (true) {
        cap.read(frame);

        // pre-processing
        cv::resize(frame, frame, cv::Size(input_w, input_h));
        cv::cvtColor(frame, blob_image, cv::COLOR_BGR2RGB);
        blob_image.convertTo(blob_image, CV_32F, 1. / 255.);

        // mat to array
        float* pixels = (float*)(blob_image.data);

        // pass input array to blob
        ai_core->GetInputBlob(pixels, color_norm_mean, color_norm_std);

        // do inference
        ai_core->Inference();

        float* output_data = new float[input_w * input_h];
        ai_core->GetOutputBlob(output_data, true);


        std::vector<float> output_img(input_w * input_h);
        for (size_t i = 0; i < input_w * input_h; i++) {
            float val = static_cast<float>(output_data[i]);
            output_img[i] = val;
        }

        cv::Mat depth_map(input_h, input_w, CV_32FC1, output_img.data());
        depth_map = max_val * (depth_map);
        depth_map.convertTo(depth_map, CV_8UC1);

        cv::cvtColor(depth_map, depth_map, cv::COLOR_GRAY2BGR);
        cv::hconcat(frame, depth_map, show);

        cv::imshow("Output", show);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
}

void CVImage() {
    // check input folder
    if (!std::experimental::filesystem::is_directory(input_folder) || !std::experimental::filesystem::exists(input_folder)) {
        printf("\nWarning: folder '%s' is not exit ...", std::filesystem::current_path().string().append(input_folder).c_str());
        return;
    }

    // find all image in folder
    std::vector<std::filesystem::path> fn;
    std::vector<std::string> allowedextensions = { ".jpg", ".png", ".jpeg", ".bmp"};
    for (const auto& file : std::filesystem::recursive_directory_iterator(input_folder)) {
        for (int i = 0; i < allowedextensions.size(); i++) {
            if (file.path().extension().string() == allowedextensions[i]) fn.push_back(file);
        }
    }
    size_t count = fn.size(); 

    for (size_t i = 0; i < count; i++) {
        string input_path = fn[i].string();
        printf("Loading image from '%s'\n", input_path.c_str());

        // read image
        frame = cv::imread(input_path);
        const int org_h = frame.rows;
        const int org_w = frame.cols;

        // pre-processing
        cv::resize(frame, frame, cv::Size(input_w, input_h));
        cv::cvtColor(frame, blob_image, cv::COLOR_BGR2RGB);
        blob_image.convertTo(blob_image, CV_32F, 1. / 255.);

        // Mat to array
        float* pixels = (float*)(blob_image.data);

        // pass input array to blob
        ai_core->GetInputBlob(pixels, color_norm_mean, color_norm_std);

        // do inference
        ai_core->Inference();

        float* output_data = new float[input_w * input_h];
        ai_core->GetOutputBlob(output_data, true);

        std::vector<float> output_img(input_w * input_h);
        for (size_t i = 0; i < input_w * input_h; i++) {
            float val = static_cast<float>(output_data[i]);
            output_img[i] = val;
        }

        cv::Mat depth_map(input_h, input_w, CV_32FC1, output_img.data());
        depth_map = max_val * (depth_map);
        depth_map.convertTo(depth_map, CV_8UC1);

        cv::cvtColor(depth_map, depth_map, cv::COLOR_GRAY2BGR);
        cv::hconcat(frame, depth_map, show);

        cv::imshow("Output", show);
        cv::waitKey(1000);

        std::string base_filename = input_path.substr(input_path.find_last_of("/\\") + 1);
        std::string::size_type const p(base_filename.find_last_of('.'));
        std::string output_name = base_filename.substr(0, p).append("_depth.png");

        std::string output_folder = "../output/";
        if (!std::experimental::filesystem::is_directory(output_folder) || !std::experimental::filesystem::exists(output_folder)) {
            std::experimental::filesystem::create_directory(output_folder); // create src folder
        }

        std::string output_path = output_folder + output_name;

        cv::resize(depth_map, depth_map, cv::Size(org_w, org_h));
        cv::imwrite(output_path, depth_map);
    }
}

