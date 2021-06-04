#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <opencv2/dnn/layer.details.hpp> 
#include "../opencvCustomOps/GreaterLayer.hpp"
#include "../opencvCustomOps/Sampling.hpp"
#include "../opencvCustomOps/Grouping.hpp"
#include "../opencvCustomOps/Calculate.hpp"

using namespace cv;
using namespace dnn;
using namespace std;

template <class Type>
Type stringToNum(const string& str) {
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}
cv::Mat getInput(string idx) {
	string inputDir = "./data/pre_data/"+idx+".txt";
	ifstream ifs(inputDir);
	string str;
	float* inp = new  float[65536];
	for (int i = 0; i < 65536; i++) {
		ifs >> str;
		inp[i] = stringToNum<float>(str);
	}
	int size[] = { 1,16384,4 };
	cv::Mat C = cv::Mat(3, size, CV_32F, inp);
	return C;
}

int main(int argc, char** argv)
{
    int modelnum = atoi(argv[1]);
	CV_DNN_REGISTER_LAYER_CLASS(Max, MaxLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Sin, SinLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Cos, CosLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Square, SquareLayer);
	CV_DNN_REGISTER_LAYER_CLASS(QueryBallPoint, QueryBallPointLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MySqueeze, MySqueezeLayer);
	CV_DNN_REGISTER_LAYER_CLASS(GreaterThenCast, GreaterThenCastLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MyExpandDims, MyExpandDimsLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MyExpandDimsF, MyExpandDimsFLayer);
	CV_DNN_REGISTER_LAYER_CLASS(FarthestPointSample, FarthestPointSampleLayer);
	CV_DNN_REGISTER_LAYER_CLASS(GatherPoint, GatherPointLayer);
	CV_DNN_REGISTER_LAYER_CLASS(GroupPoint, GroupPointLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MyTranspose, MyTransposeLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MyTransposeBack, MyTransposeBackLayer);
	CV_DNN_REGISTER_LAYER_CLASS(CiSum, CiSumLayer);
	CV_DNN_REGISTER_LAYER_CLASS(CiiSum, CiiSumLayer);
	CV_DNN_REGISTER_LAYER_CLASS(NonMaxSuppressionV2, NonMaxSuppressionV2Layer);
	CV_DNN_REGISTER_LAYER_CLASS(Cast, CastLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Unpack, UnpackLayer);
	CV_DNN_REGISTER_LAYER_CLASS(StackWithZero, StackWithZeroLayer);
	CV_DNN_REGISTER_LAYER_CLASS(OneHot, OneHotLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Maximum, MaximumLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Minimum, MinimumLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MyConcat, MyConcatLayer);
	CV_DNN_REGISTER_LAYER_CLASS(ArgMax, ArgMaxLayer);
	CV_DNN_REGISTER_LAYER_CLASS(FarthestPointSampleWithDistance, FarthestPointSampleWithDistanceLayer);
	CV_DNN_REGISTER_LAYER_CLASS(BatchMatMul, BatchMatMulLayer);
	CV_DNN_REGISTER_LAYER_CLASS(QueryBallPointDilated, QueryBallPointDilatedLayer);
	CV_DNN_REGISTER_LAYER_CLASS(QueryBallPointDilatedV2, QueryBallPointDilatedV2Layer);
	CV_DNN_REGISTER_LAYER_CLASS(Gather, GatherLayer);
	CV_DNN_REGISTER_LAYER_CLASS(GatherV2, GatherV2Layer);
	String modelPath;
	if(modelnum==0)
	    modelPath = "./model/3DSSD-4096.pb";
	else if(modelnum==1)
		modelPath = "./model/3DSSD-2048.pb";
	else
		modelPath = "./model/3DSSD-4096.pb";
	ifstream in("./data/list.txt");
	vector<string> Idx;
	string idx;
	while (getline(in, idx)) {
		Idx.push_back(idx);
	}
	for (auto i : Idx) {
        std::cout << "Running: "<< i << std::endl;
		Mat blob = getInput(i);
        Net net = readNetFromTensorflow(modelPath);
		net.setInput(blob);
		std::vector<String> outNames(2);
		outNames[0] = "import/Gather";
		outNames[1] = "import/Gather_1";
		std::vector<Mat> outs(2);
		net.forward(outs, outNames);
		std::ofstream result_3d("./data/aft_data/"+i+".txt", std::ios::out);
		for (int i = 0; i < 1; i++) {
			for (int k = 0; k < 100; k++) {
				for (int j = 0; j < 7; j++) {
					result_3d << outs[0].ptr<float>(k)[j] << " ";
				}
				result_3d << outs[1].ptr<float>(0)[k] << " " << std::endl;
			}
		}
	}
	std::cout << "finished" << std::endl;
	return 0;
}