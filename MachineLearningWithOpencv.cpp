#include "stdafx.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <vector>

using namespace cv;
using namespace cv::ml;

int main(int, char**)
{
	// Training data
	const int trainingValues = 110;
	float trainingData[trainingValues][3] = {	
	{ 100, 100 ,1},{ 200, 100 ,1},{ 300, 100 ,1},{ 400, 100 ,1},{ 500, 100 ,1},
	{ 100, 200 ,1},{ 200, 200 ,1},{ 300, 200 ,1},{ 400, 200 ,1},{ 500, 200 ,1},
	{ 100, 300 ,1},{ 200, 300 ,1},{ 300, 300 ,1},{ 400, 300 ,1},{ 500, 300 ,1},
	{ 100, 400 ,1},{ 200, 400 ,1},{ 300, 400 ,1},{ 400, 400 ,1},{ 500, 400 ,1},
	{ 100, 500 ,1},{ 200, 500 ,1},{ 300, 500 ,1},{ 400, 500 ,1},{ 500, 500 ,1},
	{ 100, 50  ,1},{ 200, 50  ,1},{ 300, 50  ,1},{ 400, 50  ,1},{ 500, 50  ,1},
	{ 100, 150 ,1},{ 200, 150 ,1},{ 300, 150 ,1},{ 400, 150 ,1},{ 500, 150 ,1},
	{ 100, 250 ,1},{ 200, 250 ,1},{ 300, 250 ,1},{ 400, 250 ,1},{ 500, 250 ,1},
	{ 100, 350 ,1},{ 200, 350 ,1},{ 300, 350 ,1},{ 400, 350 ,1},{ 500, 350 ,1},
	{ 100, 450 ,1},{ 200, 450 ,1},{ 300, 450 ,1},{ 400, 450 ,1},{ 500, 450 ,1},
	{ 50 , 100 ,1},{ 150, 100 ,1},{ 250, 100 ,1},{ 350, 100 ,1},{ 450, 100 ,1},
	{ 50 , 200 ,1},{ 150, 200 ,1},{ 250, 200 ,1},{ 350, 200 ,1},{ 450, 200 ,1},
	{ 50 , 300 ,1},{ 150, 300 ,1},{ 250, 300 ,1},{ 350, 300 ,1},{ 450, 300 ,1},
	{ 50 , 400 ,1},{ 150, 400 ,1},{ 250, 400 ,1},{ 350, 400 ,1},{ 450, 400 ,1},
	{ 50 , 500 ,1},{ 150, 500 ,1},{ 250, 500 ,1},{ 350, 500 ,1},{ 450, 500 ,1},
	{ 50 , 50  ,1},{ 150, 50  ,1},{ 250, 50  ,1},{ 350, 50  ,1},{ 450, 50  ,1},
	{ 50 , 150 ,1},{ 150, 150 ,1},{ 250, 150 ,1},{ 350, 150 ,1},{ 450, 150 ,1},
	{ 50 , 250 ,1},{ 150, 250 ,1},{ 250, 250 ,1},{ 350, 250 ,1},{ 450, 250 ,1},
	{ 50 , 350 ,1},{ 150, 350 ,1},{ 250, 350 ,1},{ 350, 350 ,1},{ 450, 350 ,1},
	{ 50 , 450 ,1},{ 150, 450 ,1},{ 250, 450 ,1},{ 350, 450 ,1},{ 450, 450 ,1},
	{ 0  , 50  ,1},{   0, 100 ,1},{   0, 150 ,1},{   0, 200 ,1},{   0, 250 ,1},
	{ 0  , 300 ,1},{   0, 350 ,1},{   0, 400 ,1},{   0, 450 ,1},{   0, 500 ,1} };
	int labels[trainingValues];
	for (int i = 0; i < trainingValues; ++i) {
		float x = trainingData[i][0];
		float y = trainingData[i][1];
		if (y > 500 * pow((1 / (0.2 * x)) , 0.2))
			labels[i] = 1;
		else
			labels[i] = -1;
	}
	Mat labelsMat(trainingValues, 1, CV_32SC1, labels);
	Mat trainingDataMat(trainingValues, 3, CV_32FC1, trainingData);

	// Train
	Ptr<DTrees> dec_trees = DTrees::create();

	dec_trees->setMaxDepth(10);
	dec_trees->setMinSampleCount(2);
	dec_trees->setCVFolds(0);
	Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	dec_trees->train(td);

	//Test Data
	Mat test_sample;
	float testingData[1][3] = { { 50, 320, 1 } };
	Mat testingDataMat = Mat(1, 3, CV_32FC1, testingData);
	float result = dec_trees->predict(testingDataMat);


	// Visualize
	int thickness = -1;
	int lineType = 8;
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);
	for (int i = 0; i < trainingValues; ++i) {
		if (labels[i] < 0)
			circle(image, Point((int)trainingData[i][0], (int)trainingData[i][1]), 6, Scalar(0, 255, 0), thickness, lineType);
		else
			circle(image, Point((int)trainingData[i][0], (int)trainingData[i][1]), 6, Scalar(0, 0, 255), thickness, lineType);
	}

	for (int i = 0; i < 1; ++i) {
		if (result < 0.5)
			circle(image, Point((int)testingData[i][0], (int)testingData[i][1]), 6, Scalar(0, 155, 0), thickness, lineType);
		else
			circle(image, Point((int)testingData[i][0], (int)testingData[i][1]), 6, Scalar(0, 0, 155), thickness, lineType);
	}
	
	imshow("Simple Example", image); // show it to the user

	cv::waitKey(0);
}