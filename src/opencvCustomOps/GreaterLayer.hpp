#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace dnn;
class GreaterThenCastLayer : public cv::dnn::Layer
{
public:
	GreaterThenCastLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{

	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new GreaterThenCastLayer(params));
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE
	{

		cv::Mat& inp = *inputs[0];   // 
		cv::Mat& out = outputs[0];  //
		int* outdata = (int*)out.data;
		int* inpdata = (int*)inp.data;

		for (int i = 0; i < inp.total(); i++) {
			outdata[i] = inpdata[i] > 0 ? 1 : 0;
		}

	}
};
class CastLayer : public cv::dnn::Layer
{
public:
	CastLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{

	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new CastLayer(params));
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
		float* outdata = (float*)out.data;
		int* inpdata = (int*)inp.data;
		/*	memcpy(outdata, inpdata, sizeof(int) * out.total());*/
		for (int i = 0; i < out.total(); i++) {
			outdata[i] = inpdata[i];
		}
	}
};