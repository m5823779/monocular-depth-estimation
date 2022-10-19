#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <windows.h>
#include <iostream>

using namespace std;
using namespace cv;

auto ToTensor(cv::Mat img, bool unsqueeze = false, int unsqueeze_dim = 0) {
	at::Tensor tensor_image =
		torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);
	if (unsqueeze) {
		tensor_image.unsqueeze_(unsqueeze_dim);
	}
	return tensor_image;
}

auto ToInput(at::Tensor tensor_image) {
	// Create a vector of inputs.
	return std::vector<torch::jit::IValue>{tensor_image};
}

auto ToCvImage(at::Tensor tensor, int cv_type = CV_8UC3) {
	int width = tensor.sizes()[0];
	int height = tensor.sizes()[1];
	try {
		cv::Mat output_mat;
		if (cv_type == CV_8UC3 || cv_type == CV_8UC1) {
			cv::Mat cv_image(cv::Size{ height, width }, cv_type,
				tensor.data_ptr<uchar>());
			output_mat = cv_image;
		}

		else if (cv_type == CV_32FC3 || cv_type == CV_32FC1) {
			cv::Mat cv_image(cv::Size{ height, width }, cv_type,
				tensor.data_ptr<float>());
			output_mat = cv_image;
		}

		else if (cv_type == CV_64FC3 || cv_type == CV_64FC1) {
			cv::Mat cv_image(cv::Size{ height, width }, cv_type,
				tensor.data_ptr<double>());
			output_mat = cv_image;
		}

		return output_mat.clone();
	}

	catch (const c10::Error& e) {
		std::cout << "an error has occured : " << e.msg() << std::endl;
	}
	return cv::Mat(height, width, CV_8UC3);
}

int main(int argc, const char* argv[]) {
	string model_name = "../weights/model-small-traced.pt";
	int net_width = 256;
	int net_height = 256;

	// setting Device
	torch::jit::script::Module module;
	torch::Device device = torch::kCPU;


	// timer
	double program_duration;
	clock_t program_start, program_end;

	// model settings
	at::Tensor mean, std;
	at::Tensor output, tensor;

	// prepareing model
	try {
		module = torch::jit::load(model_name);
	}
	catch (const c10::Error& e) {
		std::cerr << "Error loading the model\n";
		exit(0);
	}

	try {
		module.eval();
		torch::jit::getProfilingMode() = false;
		torch::jit::setGraphExecutorOptimize(true);

		mean = torch::tensor({ 0.485, 0.456, 0.406 });
		std = torch::tensor({ 0.229, 0.224, 0.225 });

		if (torch::hasCUDA()) {
			std::cout << "cuda is available" << std::endl;
			at::globalContext().setBenchmarkCuDNN(true);
			device = torch::Device(torch::kCUDA);
			module.to(device);
			mean = mean.to(device);
			std = std.to(device);
		}
	}
	catch (const c10::Error& e) {
		std::cerr << " module initialization: " << e.msg() << std::endl;
	}

	Mat frame;
	Mat input;
	Mat sbs_image;

	cv::VideoCapture cap(0, cv::CAP_DSHOW);
	//cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	//cap.set(CAP_PROP_FRAME_HEIGHT, 720);

	while (true) {
		cap.read(frame);

		int input_h = frame.rows;
		int input_w = frame.cols;

		// start generate side by side image
		program_start = clock();

		// image pre processing
		cvtColor(frame, input, cv::COLOR_BGR2RGB);

		// pre-processing
		auto tensor_cpu = ToTensor(input);  // OpenCV-image -> Libtorch-tensor

		try {
			tensor = tensor_cpu.to(device);  // move to device (CPU or GPU)

			tensor = tensor.toType(c10::kFloat);
			tensor = tensor.permute({ 2, 0, 1 });  // HWC -> CHW
			tensor = tensor.unsqueeze(0);
			tensor = at::upsample_bilinear2d(tensor, { net_height, net_width },
				true);  // resize

			tensor = tensor.squeeze(0);
			tensor = tensor.permute({ 1, 2, 0 });  // CHW -> HWC

			tensor = tensor.div(255).sub(mean).div(std);  // normalization
			tensor = tensor.permute({ 2, 0, 1 });           // HWC -> CHW
			tensor.unsqueeze_(0);                         // CHW -> NCHW
		}
		catch (const c10::Error& e) {
			std::cerr << " pre-processing exception: " << e.msg() << std::endl;
		}

		auto input_to_net = ToInput(tensor);  // input to the network

		// inference
		output;
		try {
			output = module.forward(input_to_net).toTensor();  // run inference
		}
		catch (const c10::Error& e) {
			std::cerr << " module.forward() exception: " << e.msg() << std::endl;
		}

		output = output.detach().to(torch::kF32);

		torch::Tensor max_val = (torch::max)(output);
		torch::Tensor min_val = (torch::min)(output);

		// normalization
		torch::Tensor range_val = max_val - min_val;

		// Save depth map as image
		at::Tensor depth = output.sub(min_val).div(range_val).mul(255.0F).clamp(0, 255);
		depth = at::upsample_bilinear2d(depth.unsqueeze(0), { int(input_h), int(input_w) }, true);
		depth = depth.squeeze(0);
		depth = depth.to(torch::kU8);
		depth = depth.permute({ 1, 2, 0 }).to(torch::kCPU);
		auto depth_map = ToCvImage(depth, CV_8UC1);

		program_end = clock();
		program_duration = (double)(program_end - program_start) / CLOCKS_PER_SEC;
		cout << "Framerate: " << 1.0 / program_duration << " \tfps" << "\r";
		imshow("depth mape", depth_map);
		waitKey(1);
	}
}

