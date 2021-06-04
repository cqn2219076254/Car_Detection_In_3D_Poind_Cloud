
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include<cstring>
using namespace cv;
using namespace dnn;

class QueryBallPointDilatedLayer : public cv::dnn::Layer
{
public:
	QueryBallPointDilatedLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
		MaxR = params.get<float>("max_radius");
		MinR = params.get<float>("min_radius");
		npoint = params.get<float>("nsample");
		
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new QueryBallPointDilatedLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(3);
		std::vector<int> outShape2(2);
		outShape[0] = 1;
		outShape[1] = inputs[1][1];
		outShape[2] = npoint;
		outShape2[0] = 1;
		outShape2[1] = inputs[1][1];
		outputs.assign(1, outShape);
		outputs.push_back(outShape2);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp0 = *inputs[0];
		cv::Mat& inp1 = *inputs[1];
		cv::Mat& out0 = outputs[0];
		cv::Mat& out1 = outputs[1];
		float* inpdata0 = (float*)inp0.data;
		float* inpdata1 = (float*)inp1.data;
		int* outdata0 = (int*)out0.data;
		int* outdata1 = (int*)out1.data;
		int i = 0;
		while (i < inp1.size[1]) {
			int k = 0;
			for (int j = 0; j < inp0.size[1]; ++j) {
				float distance = (inpdata0[j*3] - inpdata1[i * 3]) * (inpdata0[j * 3] - inpdata1[i * 3]);
				distance += (inpdata0[j * 3 + 1] - inpdata1[i * 3 + 1]) * (inpdata0[j * 3 + 1] - inpdata1[i * 3 + 1]);
				distance += (inpdata0[j * 3 + 2] - inpdata1[i * 3 + 2]) * (inpdata0[j * 3 + 2] - inpdata1[i * 3 + 2]);
				if (distance==0||(distance<MaxR * MaxR && distance > MinR * MinR)) {
					outdata0[i * npoint + k] = j;
					++k;
					if (k == npoint) {
						break;
					}
				}
			}
			outdata1[i] = k;
			while (k < npoint) {
				outdata0[i * npoint + k] = outdata0[i * npoint];
				++k;
			}
			++i;
		}
		
	}
private:
	float MaxR;
	float MinR;
	int npoint;
};
class QueryBallPointDilatedV2Layer : public cv::dnn::Layer
{
public:
	QueryBallPointDilatedV2Layer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
		MaxR = params.get<float>("max_radius");
		MinR = params.get<float>("min_radius");
		npoint = params.get<float>("nsample");

	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new QueryBallPointDilatedV2Layer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(3);
		std::vector<int> outShape2(2);
		outShape[0] = 1;
		outShape[1] = inputs[0][0];
		outShape[2] = npoint;
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp0 = *inputs[0];
		cv::Mat& out0 = outputs[0];
		float* inpdata0 = (float*)inp0.data;
		int* outdata0 = (int*)out0.data;
		int i = 0;
		while (i < inp0.size[0]) {
			int k = 0;
			for (int j = 0; j < inp0.size[1]; ++j) {
				float distance = inpdata0[i * inp0.size[1]+j];
				if (distance == 0 || (distance<MaxR * MaxR && distance > MinR * MinR)) {
					outdata0[i * npoint + k] = j;
					++k;
					if (k == npoint) {
						break;
					}
				}
			}
			while (k < npoint){
				outdata0[i * npoint + k] = outdata0[i * npoint];
				++k;
			}
			++i;
		}
	}
private:
	float MaxR;
	float MinR;
	int npoint;
};

class QueryBallPointLayer : public cv::dnn::Layer
{
public:
	QueryBallPointLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
		MaxR = params.get<float>("radius");
		npoint = params.get<float>("nsample");

	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new QueryBallPointLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(3);
		std::vector<int> outShape2(2);
		outShape[0] = 1;
		outShape[1] = inputs[1][1];
		outShape[2] = npoint;
		outShape2[0] = 1;
		outShape2[1] = inputs[1][1];
		outputs.assign(1, outShape);
		outputs.push_back(outShape2);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp0 = *inputs[0];
		cv::Mat& inp1 = *inputs[1];
		cv::Mat& out0 = outputs[0];
		cv::Mat& out1 = outputs[1];
		float* inpdata0 = (float*)inp0.data;
		float* inpdata1 = (float*)inp1.data;
		int* outdata0 = (int*)out0.data;
		int* outdata1 = (int*)out1.data;
		int i = 0;
		while (i < inp1.size[1]) {
			int k = 0;
			for (int j = 0; j < inp0.size[1]; ++j) {
				float distance = (inpdata0[j * 3] - inpdata1[i * 3]) * (inpdata0[j * 3] - inpdata1[i * 3]);
				distance += (inpdata0[j * 3 + 1] - inpdata1[i * 3 + 1]) * (inpdata0[j * 3 + 1] - inpdata1[i * 3 + 1]);
				distance += (inpdata0[j * 3 + 2] - inpdata1[i * 3 + 2]) * (inpdata0[j * 3 + 2] - inpdata1[i * 3 + 2]);
				if (distance == 0 || (distance<MaxR * MaxR)) {
					outdata0[i * npoint + k] = j;
					++k;
					if (k == npoint) {
						break;
					}
				}
			}
			outdata1[i] = k;
			while (k < npoint) {
				outdata0[i * npoint + k] = outdata0[i * npoint];
				++k;
			}
			++i;
		}
	}
private:
	float MaxR;
	int npoint;
};
class GroupPointLayer : public cv::dnn::Layer
{
public:
	GroupPointLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new GroupPointLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(inputs[1].size()+1);
		outShape.assign(inputs[1].begin(), inputs[1].end());
		outShape.push_back(inputs[0][2]);
		outputs.assign(1, outShape);
		
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp0 = *inputs[0];
		cv::Mat& inp1 = *inputs[1];
		cv::Mat& out0 = outputs[0];
		float* inpdata0 = (float*)inp0.data;
		int* inpdata1 = (int*)inp1.data;
		float* outdata0 = (float*)out0.data;
		for (int i = 0; i < inp1.total(); i++) {
			for (int j = 0; j < inp0.size[2]; j++) {
				outdata0[i * inp0.size[2] + j] = inpdata0[inpdata1[i] * inp0.size[2] + j];
			}
		}
	}
};