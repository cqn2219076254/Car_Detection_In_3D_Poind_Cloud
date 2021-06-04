#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace dnn;
class CosLayer : public cv::dnn::Layer
{
public:
	CosLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}
	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new CosLayer(params));
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];

		float* outdata = (float*)out.data;
		float* inpdata = (float*)inp.data;
	
		for (int i = 0; i < inp.total(); i++) {
			outdata[i] = cos(inpdata[i]);
		}
	}
};
class SinLayer : public cv::dnn::Layer
{
public:
	SinLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}
	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new SinLayer(params));
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];

		
		float* outdata = (float*)out.data;
		float* inpdata = (float*)inp.data;
		for (int i = 0; i < inp.total(); i++) {
			outdata[i] = sin(inpdata[i]);
		}
	}
};
class SquareLayer : public cv::dnn::Layer
{
public:
	SquareLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new SquareLayer(params));
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];

		
		float* outdata = (float*)out.data;
		float* inpdata = (float*)inp.data;
		for (int i = 0; i < inp.total(); i++) {
			outdata[i] = inpdata[i] * inpdata[i];
		}
	}
};

class MaxLayer : public cv::dnn::Layer
{
public:
	MaxLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
		idx = blobs[0].at<int>(0, 0);
		if (idx == 2) {
			idx = 3;       //NHWC -->NCHW
		}
		
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new MaxLayer(params));
	}

	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		
		std::vector<int> outShape(inputs[0].size()-1);
		int j = 0;
		for (int i = 0; i < inputs[0].size(); i++) {
			if (i == idx) continue;
			else {
				outShape[j] = inputs[0][i];
				j++;
			}
		}
		outputs.assign(1, outShape);
		return false;
	}


	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
		In = inp.size[idx];
		int m = 0;
		for (; m < idx; m++) pre *= inp.size[m];
		for (m++; m < inp.dims; m++) post *= inp.size[m];
	
		float* outdata = (float*)out.data;
		float* inpdata = (float*)inp.data;
		int l = 0;
		float max = inpdata[0];
		if (idx == 1) {
			memcpy(outdata, inpdata, sizeof(float) * out.total());
		}
		else {
			//for (int i = 0; i < pre; i++) {
			//	for (int k = 0; k < In; k++) {
			//		for (int j = 0; j < post; j++) {
			//			//outdata[i * post + j] = outdata[i * post + j] < inpdata[In * post * i + j + k * post] ? inpdata[In * post * i + j + k * post] : outdata[i * post + j];
			//		}
			//	}
			//}

			for (int i = 0; i < pre; i++) {
				for (int j = 0; j < post; j++) {
					for (int k = 0; k < In; k++) {
						max = max < inpdata[In * post * i + j + k * post] ? inpdata[In * post * i + j + k * post] : max;
					}
					outdata[l++] = max;
					max = -3;
				}
				max = -3;
			}
		}
	}
private:
	int idx;
	int pre = 1;
	int post = 1;
	int In = 1;
};

class MaximumLayer : public cv::dnn::Layer
{
public:
	MaximumLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new MaximumLayer(params));
	}


	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
	
		float* outdata = (float*)out.data;
		float* inpdata = (float*)inp.data;
		float max = inpdata[0]; 
		if(blobs[0].size[1]!=1) //just for 1*3 and 1 suit my model
		for (int i = 0; i < inp.total(); i++) {
			outdata[i] = inpdata[i] > blobs[0].ptr<float>(0)[i % 3] ? inpdata[i] : blobs[0].ptr<float>(0)[i % 3];
		}
		else
			for (int i = 0; i < inp.total(); i++) {
				outdata[i] = inpdata[i] > blobs[0].ptr<float>(0)[0] ? inpdata[i] : blobs[0].ptr<float>(0)[0];
			}
	}
	
};
class MinimumLayer : public cv::dnn::Layer
{
public:
	MinimumLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new MinimumLayer(params));
	}
	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
	
		float* outdata = (float*)out.data;
		float* inpdata = (float*)inp.data;
		float max = inpdata[0];
		if (blobs[0].size[1] != 1) //just for 1*3 and 1 suit my model
			for (int i = 0; i < inp.total(); i++) {
				outdata[i] = inpdata[i] < blobs[0].ptr<float>(0)[i % 3] ? inpdata[i] : blobs[0].ptr<float>(0)[i % 3];
			}
		else
			for (int i = 0; i < inp.total(); i++) {
				outdata[i] = inpdata[i] < blobs[0].ptr<float>(0)[0] ? inpdata[i] : blobs[0].ptr<float>(0)[0];
			}
	}

};

class ShapeLayer : public cv::dnn::Layer
{
public:
	ShapeLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
		 
	
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new ShapeLayer(params));
	}

	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		std::vector<int> outShape(1);
		outShape[0] = inputs[0].size();
		outputs.assign(1, outShape);
		return false;
	}
	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
		
	
		float* outdata = (float*)out.data;
		float* inpdata = (float*)inp.data;
		for (int i = 0; i < out.total(); i++) {
			outdata[i] = inp.size[i];
		}
	}
};

class ArgMaxLayer : public cv::dnn::Layer
{
public:
	ArgMaxLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new ArgMaxLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(inputs[0].size() - 1);

		// remove the last dimension
		outShape[0] = inputs[0][0];
		outShape[1] = inputs[0][2];
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{   //case when idx = -1 and the input dim is x*y*z
		//from tensorflow's NHWC to opencv's NCHW
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];

		int* outdata = (int*)out.data;
		float* inpdata = (float*)inp.data;
		float max;
		int Maxidx;
		/*for (int i = 0; i < inp.total(); i += inp.size[2]) {
			max = inpdata[i];
			Maxidx = 0;
			for (int j = 1; j < inp.size[2]; j++) {
				if (max < inpdata[i + j]) {
					max = inpdata[i + j];
					Maxidx = (i + j) % inp.size[2];
				}
			}
			outdata[i / inp.size[2]] = Maxidx;
		}*/

		for (int i = 0; i < inp.size[2]; i++) {
			max = inpdata[i];
			Maxidx = 0;
			for (int j = 1; j < inp.size[1]; j++) {
				if (max < inpdata[i + j * inp.size[2]]) {
					max = inpdata[i + j * inp.size[2]];
					Maxidx = j;
				}
			}
			outdata[i] = Maxidx;
		}
	}

};

class OneHotLayer : public cv::dnn::Layer
{
public:
	OneHotLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{


	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new OneHotLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(inputs[0].size()+1);
		outShape.assign(inputs[0].begin(), inputs[0].end());
		outShape.push_back(blobs[0].ptr<int>(0)[0]);
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{   
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
	
		int* inpdata = (int*)inp.data;
		int* Out = new int[out.total()]{0};
		for (int i = 0; i < inp.total(); i++) {
			Out[i * blobs[0].ptr<int>(0)[0] + (int)inpdata[i]] = 1;
		}
		out = cv::Mat(3, out.size, CV_32S, Out);
	}
};
class UnpackLayer : public cv::dnn::Layer
{
public:
	UnpackLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new UnpackLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(inputs[0].size()-1);

		// remove the fisrt dimension
		outShape.assign(inputs[0].begin()+1, inputs[0].end());
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		// for axis = 1; And first Dim is 1
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
	
		float* inpdata = (float*)inp.data;
		float* outdata = (float*)out.data;
		memcpy(outdata, inpdata, sizeof(float) * out.total());
	}
};

class MyExpandDimsLayer : public cv::dnn::Layer
{
public:
	MyExpandDimsLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{

	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new MyExpandDimsLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(inputs[0].size()+1);
	
		int j = 0;
		for (int i = 0; i < inputs[0].size()+1; ++i) {
			if (i == blobs[0].ptr<int>(0)[0]) outShape[i] = 1;
			else {
				outShape[i] = inputs[0][j];
				++j;
			}
		}
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
	

		int* inpdata = (int*)inp.data;
		int* outdata = (int*)out.data;
	
		memcpy(outdata, inpdata, sizeof(int) * out.total());
	}
};

class MyExpandDimsFLayer : public cv::dnn::Layer
{
public:
	MyExpandDimsFLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
		edimsize = params.get<int>("edimsize");
		expanddim = params.get<int>("expanddim");
		datalayout = params.get<String>("DataLayout");
	
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new MyExpandDimsFLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		
		std::vector<int> outShape(inputs[0].size() + 1);
		
		int axis = blobs[0].ptr<int>(0)[0];
		int j = 0;
		if (datalayout == "NHWC") {
			switch (blobs[0].ptr<int>(0)[0])
			{
			case 1: {
				axis = 2;
				break; }
			case 2: {
				axis = 3;
				break; }
			case 3: {
				axis = 1;
				break; }
			default:
				break;
			}
		}
		for (int i = 0; i < inputs[0].size() + 1; ++i) {
			if (i == axis) outShape[i] = edimsize;
			else {
				outShape[i] = inputs[0][j];
				++j;
			}
		}
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];


		float* inpdata = (float*)inp.data;
		float* outdata = (float*)out.data;
		//32 is from querypoint
		
		if (edimsize != 1) {
			for (int i = 0; i < inp.size[1]; i++) {
				for (int j = 0; j < edimsize; ++j) {
					outdata[i * edimsize * 3 + j * 3] = inpdata[i * 3];
					outdata[i * edimsize * 3 + j * 3 + 1] = inpdata[i * 3 + 1];
					outdata[i * edimsize * 3 + j * 3 + 2] = inpdata[i * 3 + 2];
				}
			}
		}
		else memcpy(outdata, inpdata, sizeof(float) * out.total());
	}
private:
	int edimsize;
	int expanddim;
	String datalayout;
};


class MyTransposeLayer : public cv::dnn::Layer
{ 
	//Transepose from NWHC TO NCWH
public:
	MyTransposeLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{

	
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new MyTransposeLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		std::vector<int> outShape(inputs[0].size());
		if (inputs[0].size() == 4) {
			outShape[0] = inputs[0][0];
			outShape[1] = inputs[0][3];
			outShape[2] = inputs[0][1];
			outShape[3] = inputs[0][2];
		}
		else if (inputs[0].size() == 3) {
			outShape[0] = inputs[0][0];
			outShape[1] = inputs[0][2];
			outShape[2] = inputs[0][1];
		}
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];

		float* inpdata = (float*)inp.data;
		float* outdata = (float*)out.data;
		cv::Mat re;
		int k = 0;
		if (inp.dims == 4) {
			re = inp.reshape(0, inp.size[1] * inp.size[2]).t();
		}
		else if (inp.dims == 3) {
			re = inp.reshape(0, inp.size[1]).t();
		}
		float* redata = (float*)re.data;
		memcpy(outdata, redata, sizeof(float) * out.total());
	}
};

class MyTransposeBackLayer : public cv::dnn::Layer
{
	//Transepose from NWHC TO NCWH
public:
	MyTransposeBackLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{

		
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new MyTransposeBackLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		std::vector<int> outShape(inputs[0].size());
		//NCW --> NWC
		outShape[0] = inputs[0][0];
		outShape[1] = inputs[0][2];
		outShape[2] = inputs[0][1];
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
	
		float* inpdata = (float*)inp.data;
		float* outdata = (float*)out.data;
		int k = 0;
		for (int i = 0; i < inp.size[2]; i++) {
			for (int j = 0; j < inp.size[1] ; j++) {
				outdata[k] = inpdata[j * inp.size[2] + i];
				k++;
			}
		}
	}
};

class MySqueezeLayer : public cv::dnn::Layer
{
	//Transepose from NWHC TO NCWH
public:
	MySqueezeLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
		idx = params.get<int>("squeeze_dims");
		DataLayout = params.get<String>("DataLayout");
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new MySqueezeLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		std::vector<int> outShape(inputs[0].size()-1);
		int axis = idx;
		if(DataLayout=="NHWC"){
			switch (idx)
			{
			case 1: {
				axis = 2;
				break; }
			case 2: {
				axis = 3;
				break; }
			case 3: {
				axis = 1;
				break; }
			default:
				break;
			}
		}
		int j = 0;
		for (int i = 0; i < inputs[0].size(); ++i) {
			if (i != axis) {
				outShape[j] = inputs[0][i];
				++j;
			}
		}
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];
		float* inpdata = (float*)inp.data;
		float* outdata = (float*)out.data;
		memcpy(outdata, inpdata, sizeof(float) * out.total());
	}
private:
	int idx;
	String DataLayout;
};


class CiSumLayer : public cv::dnn::Layer
{

public:
	CiSumLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new CiSumLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		std::vector<int> outShape(inputs[0].size() - 1);
		outShape.assign(inputs[0].begin(), inputs[0].end()-1);
		if (inputs[0][1] == 1) outShape[1] = outShape[2];
		if (inputs[0][2] == 1) outShape[2] = outShape[1];
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];

		float* inpdata = (float*)inp.data;
		float* outdata = (float*)out.data;
		// inp 1x1x4096x70
		if (inp.size[1] == 1) {
			for (int i = 0; i < inp.size[2]; i++) {
				float tmp = 0;
				for (int j = 0; j < inp.size[3]; j++) {
					tmp += inpdata[i * inp.size[3] + j];
				}
				for (int j = 0; j < inp.size[2]; j++) {
					outdata[i + j * inp.size[2]] = tmp;
				}
			}
		}
		else if (inp.size[2] == 1) {
			for (int i = 0; i < inp.size[1]; i++) {
				float tmp = 0;
				for (int j = 0; j < inp.size[3]; j++) {
					tmp += inpdata[i * inp.size[3] + j];
				}
				for (int j = 0; j < inp.size[1]; j++) {
					outdata[j + i * inp.size[1]] = tmp;
				}
			}
		}
		
	}
};
class CiiSumLayer : public cv::dnn::Layer
{

public:
	CiiSumLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new CiiSumLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		std::vector<int> outShape(inputs[0].size() - 1);
		outShape.assign(inputs[0].begin(), inputs[0].end() - 1);
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp = *inputs[0];
		cv::Mat& out = outputs[0];

		float* inpdata = (float*)inp.data;
		float* outdata = (float*)out.data;
		// inp 
	    for (int i = 0; i < inp.size[1]; i++) {
			float tmp = 0;
			for (int j = 0; j < inp.size[2]; j++) {
				tmp += inpdata[i * inp.size[2] + j];
			}
			outdata[i] = tmp;
		}
	}
	
};

class BatchMatMulLayer : public cv::dnn::Layer
{

public:
	BatchMatMulLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new BatchMatMulLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		std::vector<int> outShape(inputs[0].size());
		outShape[0] = inputs[0][0];
		outShape[1] = inputs[0][1];
		outShape[2] = inputs[1][2];
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp0 = *inputs[0];
		cv::Mat& inp1 = *inputs[1];
		cv::Mat& out = outputs[0];

		//float* outdata = (float*)out.data;
		/*int* size0 = new int[2]{ inp0.size[1],inp0.size[2] };
		int* size1 = new int[2]{ inp1.size[1],inp1.size[2] };

		cv::Mat tmp0 = cv::Mat(2, size0, CV_32F, (float*)inp0.data);
		cv::Mat tmp1 = cv::Mat(2, size1, CV_32F, (float*)inp1.data);
		cv::Mat re = tmp0 * tmp1;*/

		cv::Mat re = inp0.reshape(0, inp0.size[1]) * inp1.reshape(0, inp1.size[1]);
		memcpy((float*)out.data, (float*)re.data, sizeof(float) * out.total());

 	}
};

class MyConcatLayer : public cv::dnn::Layer
{

public:
	MyConcatLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
		outnum = params.get<int>("outnum");
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new MyConcatLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		std::vector<int> outShape(inputs[0].size());
		outShape[0] = inputs[0][0];
		outShape[1] = inputs[0][1]*2;
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp0 = *inputs[0];
		cv::Mat& out = outputs[0];

		int* outdata = (int*)out.data;
		int* inpdata = (int*)inp0.data;
		if (outnum == 1024) {
			for (int i = 0; i < 512; i++) {
				outdata[i + 512] = i;
				outdata[i] = inpdata[i];
			}
		}
		else if (outnum == 512) {
			for (int i = 0; i < 256; i++) {
				outdata[i] = inpdata[i];
				outdata[i + 256] = i+512;
			}
		}
	}
private:
	int outnum;
};

class StackWithZeroLayer : public cv::dnn::Layer
{
	// 1*256 --> 1*256*3 
public:
	StackWithZeroLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new StackWithZeroLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(inputs[0].size());
		outShape[0] = inputs[0][0];
		outShape[1] = inputs[0][1];
		outShape[2] = 3;
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp0 = *inputs[0];
		cv::Mat& out = outputs[0];
		int* outdata = (int*)out.data;
		int* inpdata = (int*)inp0.data;
		for (int i = 0; i < inp0.size[1]; i++) {
			outdata[3 * i] = 0;
			outdata[3 * i + 1] = inpdata[i];
			outdata[3 * i + 2] = 0;
		}
	}

};

class NonMaxSuppressionV2Layer : public cv::dnn::Layer
{
public:
	NonMaxSuppressionV2Layer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new NonMaxSuppressionV2Layer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		std::vector<int> outShape(2);
		outShape[0] = 1;
		outShape[1] = 100;
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat& inp0 = *inputs[0];
		cv::Mat& inp1 = *inputs[1]; //scores
		cv::Mat& out = outputs[0];

		int* outdata = (int*)out.data;
		float* inpdata1 = (float*)inp1.data;
		float* inpdata0 = (float*)inp0.data;
		std::vector<int> index(256, 0);
		for (int i = 0; i != index.size(); i++) {
			index[i] = i;
		}
		sort(index.begin(), index.end(),
			[&](const int& a, const int& b) {
				return (inpdata1[a] > inpdata1[b]);
			}
		);
		int j = 1;
		bool remove = false;
		outdata[0] = index[0];
		for (int i = 1; i < 256; i++) {
			remove = false;
			float* box2 = inpdata0 + 4 * index[i];
			for (int k = j-1; k >= 0; k--) {
				if (IoU(inpdata0 + 4 * outdata[k], box2) > 0.1) { //non_max_suppression/iou_threshold is 0.1
					remove = true;
					break;
				}
			}
			if (!remove) {
				outdata[j] = index[i];
				j++;
			}
			if (j == 100) break;
		}
		while (j < 100) {
			outdata[j] = -1;
			j++;
		}
		
	}
private:
	virtual float IoU(float* box1,float* box2) {
		float s1,s2;
		float a = std::min(box1[2], box2[2]) - std::max(box1[0], box2[0]);
		float b = std::min(box1[3], box2[3]) - std::max(box1[1], box2[1]);
		if (a <= 0 || b <= 0) s1 = 0;
		else s1 = a * b;
		s2 = (box1[3] - box1[1]) * (box1[2] - box1[0]) + (box2[3] - box2[1]) * (box2[2] - box2[0]) - s1;
		return s1 / s2;
	}
};

class GatherLayer : public cv::dnn::Layer
{
public:
	GatherLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new GatherLayer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
	
		std::vector<int> outShape(inputs[0].size());
		for (int i = 0; i < inputs[0].size();i++) {
			if (i == 0) outShape[i] = inputs[1][1];
			else outShape[i] = inputs[0][i];
		}
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat* inp1 = inputs[1]; // index
		cv::Mat& inp0 = *inputs[0]; // orignal data
		cv::Mat& out = outputs[0];
		float* inpdata0 = (float*)inp0.data;
		int* inpdata1 = (int*)inp1->data;
		float* outdata = (float*)out.data;
		for (int i = 0; i < out.size[0]; i++) {
			if (inpdata1[i] != -1) {
				for (int j = 0; j < out.size[1]; j++) {
					outdata[i * out.size[1] + j] = inpdata0[j + inpdata1[i] * out.size[1]];
				}
			}
			else {
				for (int j = 0; j < out.size[1]; j++) {
					outdata[i * out.size[1] + j] = -1;
				}
			}
		}
	}

};

class GatherV2Layer : public cv::dnn::Layer
{
public:
	GatherV2Layer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
		if (blobs.size() == 0) axis = 0;
		else {
			axis = blobs[0].ptr<int>(0)[0];
		}
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new GatherV2Layer(params));
	}
	virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> >& outputs,
		std::vector<std::vector<int> >& internals) const CV_OVERRIDE
	{
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

		std::vector<int> outShape(inputs[0].size());
		for (int i = 0; i < inputs[0].size(); i++) {
			if (i == axis) outShape[i] = inputs[1][1];
			else outShape[i] = inputs[0][i];
		}
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat* inp1 = inputs[1]; // index
		cv::Mat* inp0 = inputs[0]; // orignal data
		cv::Mat& out = outputs[0];
		float* inpdata0 = (float*)inp0->data;
		int* inpdata1 = (int*)inp1->data;
		float* outdata = (float*)out.data;
		if (axis == 0) {
			for (int i = 0; i < out.size[0]; i++) {
				if (inpdata1[i] != -1) {
					for (int j = 0; j < out.size[1]; j++) {
						outdata[i * out.size[1] + j] = inpdata0[j + inpdata1[i] * out.size[1]];
					}
				}
				else {
					for (int j = 0; j < out.size[1]; j++) {
						outdata[i * out.size[1] + j] = -1;
					}
				}
			}
		}
		else if (axis == 1) {
			for (int i = 0; i < out.total(); i++) {
				if (inpdata1[i] != -1) {
					outdata[i] = inpdata0[inpdata1[i]];
				}
				else outdata[i] = -1;
			}
		}
	}
private:
	int axis;
};


class DivLayer : public cv::dnn::Layer
{
public:
	DivLayer(const cv::dnn::LayerParams& params) : cv::dnn::Layer(params)
	{
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
	{
		return cv::Ptr<cv::dnn::Layer>(new DivLayer(params));
	}

	virtual void forward(std::vector<cv::Mat*>& inputs, std::vector<cv::Mat>& outputs, std::vector<cv::Mat>& internals) CV_OVERRIDE

	{
		cv::Mat* inp = inputs[0];
		cv::Mat& out = outputs[0];
		float* inpdata0 = (float*)inp->data;
		float* outdata = (float*)out.data;
		for (int i = 0; i < out.total(); i++) {
			outdata[i] = inpdata0[i] / blobs[0].ptr<float>(0)[0];
		}
		
	}

};