#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

Ptr<ANN_MLP> model;

double minValx2 = 0, maxvalx2 = 0, minValx1 = 0, maxvalx1 = 0;

// This function reads data and responses from the file <filename>
static bool
read_num_class_data(const string& filename, int var_count,
Mat* _data, Mat* _responses)
{
	const int M = 1024;
	char buf[M + 2];

	Mat el_ptr(1, var_count, CV_32F);
	int i;
	vector<int> responses;

	_data->release();
	_responses->release();

	FILE* f = fopen(filename.c_str(), "rt");
	if (!f)
	{
		cout << "Could not read the database " << filename << endl;
		return false;
	}

	for (;;)
	{
		char* ptr;
		if (!fgets(buf, M, f) || !strchr(buf, ','))
			break;
		responses.push_back((int)buf[0]);
		ptr = buf + 2;
		for (i = 0; i < var_count; i++)
		{
			int n = 0;
			sscanf(ptr, "%f%n", &el_ptr.at<float>(i), &n);
			ptr += n + 1;
		}
		if (i < var_count)
			break;
		_data->push_back(el_ptr);
	}
	fclose(f);
	Mat(responses).copyTo(*_responses);
	cout << "The database " << filename << " is loaded.\n";
	return true;
}

template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
	// load classifier from the specified file
	Ptr<T> model = StatModel::load<T>(filename_to_load);
	if (model.empty())
		cout << "Could not read the classifier " << filename_to_load << endl;
	else
		cout << "The classifier " << filename_to_load << " is loaded.\n";

	return model;
}

static Ptr<TrainData>
prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
{
	Mat sample_idx = Mat::zeros(1, data.rows, CV_8U);
	Mat train_samples = sample_idx.colRange(0, ntrain_samples);
	train_samples.setTo(Scalar::all(1));
	int nvars = data.cols;
	Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

	return TrainData::create(data, ROW_SAMPLE, responses,
		noArray(), sample_idx, noArray(), var_type);
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static void test_and_save_classifier(const Ptr<StatModel>& model,
	const Mat& data, const Mat& responses,
	int ntrain_samples, int rdelta,
	const string& filename_to_save)
{
	int i, nsamples_all = data.rows;
	double train_hr = 0, test_hr = 0;

	// compute prediction error on train and test data
	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = data.row(i);
		double x1 = (int)data.at<float>(i, 0);//157
		double x2 = (int)data.at<float>(i, 1);//67.74
		float r = model->predict(sample);
		printf("Predict: %f", r);
		r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
		if (i < ntrain_samples)
			train_hr += r;
		else
			test_hr += r;
	}

	test_hr /= nsamples_all - ntrain_samples;
	train_hr = ntrain_samples > 0 ? train_hr / ntrain_samples : 1.;

	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);

	if (!filename_to_save.empty())
	{
		model->save(filename_to_save);
	}
}

static bool
build_mlp_classifier(const string& data_filename,
const string& filename_to_save,
const string& filename_to_load)
{
	const int class_count = 5;
	Mat data;
	Mat responses;

	bool ok = read_num_class_data(data_filename, 7, &data, &responses);
	if (!ok)
		return ok;

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);

	minMaxLoc(data, &minValx1, &maxvalx1, 0, 0);
	minMaxLoc(data, &minValx2, &maxvalx2, 0, 0);

	// Create or load MLP classifier
	if (!filename_to_load.empty())
	{
		model = load_classifier<ANN_MLP>(filename_to_load);
		if (model.empty())
			return false;
		ntrain_samples = 0;
	}
	else
	{
		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		//
		// MLP does not support categorical variables by explicitly.
		// So, instead of the output class label, we will use
		// a binary vector of <class_count> components for training and,
		// therefore, MLP will give us a vector of "probabilities" at the
		// prediction stage
		//
		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

		Mat train_data = data.rowRange(0, ntrain_samples);
		Mat train_responses = Mat::zeros(ntrain_samples, class_count, CV_32F);

		// 1. unroll the responses
		cout << "Unrolling the responses...\n";
		for (int i = 0; i < ntrain_samples; i++)
		{
			int cls_label = responses.at<int>(i) -'A';
			train_responses.at<float>(i, cls_label) = 1.f;
		}

		// 2. train classifier
		int layer_sz[] = { data.cols, 50, 40, class_count };
		int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
		Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

#if 0
		int method = ANN_MLP::BACKPROP;
		double method_param = 0.001;
		int max_iter = 300;
#else
		int method = ANN_MLP::RPROP;
		double method_param = 0.1;
		int max_iter = 1000;
#endif

		Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);

		cout << "Training the classifier (may take a few minutes)...\n";
		model = ANN_MLP::create();
		model->setLayerSizes(layer_sizes);
		model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
		model->setTermCriteria(TC(max_iter, 0));
		model->setTrainMethod(method, method_param);
		model->train(tdata);
		cout << endl;
	}

	//test_and_save_classifier(model, data, responses, ntrain_samples, 'A', filename_to_save);
	predecir(model, data, responses);
	return true;
}

void predecir(const Ptr<StatModel>& model,
	const Mat& data, Mat& responses
	) {

	int i, nsamples_all = data.rows;

	double train_hr = 0, test_hr = 0;

	// compute prediction error on train and test data
	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = data.row(i);
		double x1 = (int)data.at<float>(i, 0);//157

		double x2 = (int)data.at<float>(i, 1);//67.74
		float r = model->predict(sample);
		responses.at<int>(i) = r;
	}
}

int main(int argc, char *argv[])
{
	string filename_to_save = "";
	string filename_to_load = "mlp_datosEntrenamientoLetras1.data";
	string data_filename = "datosEntrenamientoLetrasRandom.csv";
	int method = 0;

	build_mlp_classifier(data_filename, filename_to_save, filename_to_load);

	//graficarFeatures(data_filename);
	//decision(data_filename);

	getchar();
	return 0;
}