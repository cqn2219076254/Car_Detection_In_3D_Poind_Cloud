#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include<cstring>

#define SIZE1 32
#define SIZE2 64

using namespace cv;
using namespace dnn;

class FarthestPointSampleLayer : public cv::dnn::Layer
{
public:
	FarthestPointSampleLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
		npoint = params.get<int>("npoint");
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new FarthestPointSampleLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(2);
		std::vector<int> outShape1(3);
		std::vector<int> outShape2(3);
		outShape[0] = 1;
		outShape[1] = npoint;
		outShape1[0] = 1;
		outShape1[1] = npoint;
		outShape1[2] = SIZE1;
		outShape2[0] = 1;
		outShape2[1] = npoint;
		outShape2[2] = SIZE2;
		// remove the last dimension
		outputs.assign(1, outShape);
		outputs.push_back(outShape1);
		outputs.push_back(outShape1);
		outputs.push_back(outShape2);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
		cv::Mat& out1 = outputs[1];
		cv::Mat& out2 = outputs[2];
		cv::Mat& out3 = outputs[3];
		std::vector<int> set;
		bool* inSet = new bool[inp.size[1]]{ false };
		float* dist = new float[inp.size[1]]{ -1 };
		set.push_back(0);
		float* inpdata = inp.ptr<float>();
		int* outdata = out.ptr<int>();
		int* outdata1 = out1.ptr<int>();
		int* outdata2 = out2.ptr<int>();
		int* outdata3 = out3.ptr<int>();
		inSet[0] = true;
		float farthestdist = 0;
		int farthestID = 0;
		outdata[0] = 0;
		int temp;
		float Distance; 
		int k1 = 0;
		int k2 = 0;
		int k3 = 0;
		for (int i = 1; i < npoint; ++i) {
			farthestdist = -1.0;
			k1 = 0;
			k2 = 0;
			k3 = 0;
			for (int j = 0; j < inp.size[1]; ++j) {
				Distance = (inpdata[farthestID * 3] - inpdata[j * 3]) * (inpdata[farthestID * 3] - inpdata[j * 3]);
				Distance += (inpdata[farthestID * 3 + 1] - inpdata[j * 3 + 1]) * (inpdata[farthestID * 3 + 1] - inpdata[j * 3 + 1]);
				Distance += (inpdata[farthestID * 3 + 2] - inpdata[j * 3 + 2]) * (inpdata[farthestID * 3 + 2] - inpdata[j * 3 + 2]);
				if (Distance < 0.8) {
					if (Distance == 0) {
						outdata1[k1 + SIZE1 * (i - 1)] = j; k1++;
						outdata2[k2 + SIZE1 * (i - 1)] = j; k2++;
						outdata3[k3 + SIZE2 * (i - 1)] = j; k3++;
					}
					else if (Distance < 0.04 && k1 < SIZE1) { outdata1[k1 + SIZE1 * (i - 1)] = j; k1++; }
					else if (Distance < 0.16 && k2 < SIZE1) { outdata2[k2 + SIZE1 * (i - 1)] = j; k2++; }
					else if (Distance < 0.64 && k3 < SIZE2) { outdata3[k3 + SIZE2 * (i - 1)] = j; k3++; }
				}
				if (!inSet[j]) {
					if (dist[j] > 0) dist[j] = dist[j] < Distance ? dist[j] : Distance;
					else dist[j] = Distance;
					if (farthestdist < dist[j]) {
						temp = j;
						farthestdist = dist[j];
					}
				}
			}
			while (k1 < SIZE1) { outdata1[k1 + SIZE1 * (i - 1)] = outdata1[SIZE1 * (i - 1)]; k1++; }
			while (k2 < SIZE1) { outdata2[k2 + SIZE1 * (i - 1)] = outdata2[SIZE1 * (i - 1)]; k2++; }
			while (k3 < SIZE2) { outdata3[k3 + SIZE2 * (i - 1)] = outdata3[SIZE2 * (i - 1)]; k3++; }
			farthestID = temp;
			outdata[i] = farthestID;
			inSet[farthestID] = true;
		}
		k1 = 0;
		k2 = 0;
		k3 = 0;
		for (int j = 0; j < inp.size[1]; ++j) {
			Distance = (inpdata[farthestID * 3] - inpdata[j * 3]) * (inpdata[farthestID * 3] - inpdata[j * 3]);
			Distance += (inpdata[farthestID * 3 + 1] - inpdata[j * 3 + 1]) * (inpdata[farthestID * 3 + 1] - inpdata[j * 3 + 1]);
			Distance += (inpdata[farthestID * 3 + 2] - inpdata[j * 3 + 2]) * (inpdata[farthestID * 3 + 2] - inpdata[j * 3 + 2]);
			if (Distance < 0.8) {
				if (Distance == 0) {
					outdata1[k1 + SIZE1 * (npoint - 1)] = j; k1++;
					outdata2[k2 + SIZE1 * (npoint - 1)] = j; k2++;
					outdata3[k3 + SIZE2 * (npoint - 1)] = j; k3++;
				}
				else if (Distance < 0.04 && k1 < SIZE1) { outdata1[k1 + SIZE1 * (npoint - 1)] = j; k1++; }
				else if (Distance < 0.16 && k2 < SIZE1) { outdata2[k2 + SIZE1 * (npoint - 1)] = j; k2++; }
				else if (Distance < 0.64 && k3 < SIZE2) { outdata3[k3 + SIZE2 * (npoint - 1)] = j; k3++; }
			}
		}
		while (k1 < SIZE1) { outdata1[k1 + SIZE1 * (npoint - 1)] = outdata1[SIZE1 * (npoint - 1)]; k1++; }
		while (k2 < SIZE1) { outdata2[k2 + SIZE1 * (npoint - 1)] = outdata2[SIZE1 * (npoint - 1)]; k2++; }
		while (k3 < SIZE2) { outdata3[k3 + SIZE2 * (npoint - 1)] = outdata3[SIZE2 * (npoint - 1)]; k3++; }

		delete[] inSet, dist;
	}
private:
	int npoint;
};

class FarthestPointSampleWithDistanceLayer : public cv::dnn::Layer
{
public:
	FarthestPointSampleWithDistanceLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
		npoint = params.get<int>("npoint");
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new FarthestPointSampleWithDistanceLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(2);
		outShape[0] = 1;
		outShape[1] = npoint;
		// remove the last dimension
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
		std::vector<int> set;
		bool* inSet = new bool[inp.size[1]]{ false };
		float* dist = new float[inp.size[1]]{ -1 };
		set.push_back(0);
		float* inpdata = (float*)inp.data;
		int* outdata = (int*)out.data;
		inSet[0] = true;
		float farthestdist = 0;
		int farthestID = 0;
		outdata[0] = 0;
		int temp;
		float Distance;
		for (int i = 1; i < npoint; ++i) {
			farthestdist = -1.0;
			for (int j = 1; j < inp.size[1]; ++j) {
				if (!inSet[j]) {
					Distance = inp.ptr<float>(0, j)[farthestID];
					if (dist[j] > 0) dist[j] = dist[j] < Distance ? dist[j] : Distance;
					else dist[j] = Distance;
					if (farthestdist < dist[j]) {
						temp = j;
						farthestdist = dist[j];
					}
				}
			}
			farthestID = temp;
			outdata[i] = farthestID;
			inSet[farthestID] = true;
		}
		delete[] inSet, dist;
	}
private:
	int npoint;
};
class GatherPointLayer : public cv::dnn::Layer
{
public:
	GatherPointLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new GatherPointLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(3);
		if (inputs.size() == 2) {
			outShape[0] = 1;
			outShape[1] = inputs[1][1];
			outShape[2] = inputs[0][2];
		}
		else if (inputs.size() == 1) {
			outShape[0] = 1;
			outShape[1] = blobs[0].size[1];
			outShape[2] = inputs[0][2];
		}
		// remove the last dimension
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE
	{
		cv::Mat& inp0 = *inputs[0];
		cv::Mat& out = outputs[0];
		float* inpdata0 = (float*)inp0.data;
		float* outdata = (float*)out.data;
		int i = 0;
		if (inputs.size() == 2) {
			cv::Mat& inp1 = *inputs[1];
			int* inpdata1 = (int*)inp1.data;
			for (int j = 0; j < out.size[1]; j++) {
				for (int k = 0; k < inp0.size[2]; k++) {
					outdata[i + k] = inpdata0[inpdata1[j] * inp0.size[2] + k];
				}
				i += inp0.size[2];
			}
		}
		else if (inputs.size() == 1) {
			int* inpdata1 = (int*)blobs[0].data;
			for (int j = 0; j < out.size[1]; j++) {
				for (int k = 0; k < inp0.size[2]; k++) {
					outdata[i + k] = inpdata0[inpdata1[j] * inp0.size[2] + k];
				}
				i += inp0.size[2];
			}
		}
	}
};