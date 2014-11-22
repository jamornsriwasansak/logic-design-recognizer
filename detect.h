#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "hungarian.h"

#include <fstream>
#include <iostream>
using namespace cv;
using namespace std;


#define AND_GATE_TRAINING_AMOUNT 4
#define OR_GATE_TRAINING_AMOUNT 4
#define NOT_GATE_TRAINING_AMOUNT 4
#define SAMPLE_RANDOM_POINTS_AMOUNT 185

#define AND_GATE 101
#define OR_GATE 102
#define NOT_GATE 103
#define INF 1000000000000

namespace detect {
	void displayFloatMatrix(float** f, int size);

	Mat normalizeGrayscale(Mat mat);

	void yolo(Mat mat);

	Mat slimLine(Mat mat);
	
	Mat cannyInner(Mat mat);

	vector<Point> randomPointsFromMat(Mat mat, int amount);

	float fastSqrt(float x);

	float euclideanDistance(Point p1, Point p2);

	void displayFloatMatrix(float** f, int size);
	
	void displayIntMatrix(int** x, int size);

	/*vector<Histogram> calculateHistogramFromPoints(vector<Point> points) {
	vector<Histogram> h;
	for (int i = 0; i < points.size(); i++) {
	Histogram g;
	g.histogram = malloc(sizeof(float) * )
	for (int j = 0; j < points.size(); i++) {
	}
	}
	}*/

	float** calculateCostMatrixFromPoints(vector<Point> pointsMat1, vector<Point> pointsMat2);

	Mat plotDotsVector(Mat mat, vector<Point> vp);

	Mat plotLinesVector(Mat mat, vector<Point> from, vector<Point> to);

	int** float2DArrayToInt2DArray(float** a, int row, int col);

	float hungarianCalculateSumCost2(hungarian_t* prob, float **costMatrix);

	vector<Point> andgate_training[AND_GATE_TRAINING_AMOUNT + 1];
	vector<Point> orgate_training[OR_GATE_TRAINING_AMOUNT + 1];
	vector<Point> notgate_training[NOT_GATE_TRAINING_AMOUNT + 1];

	// Unimplemented
	vector<Point> readRandomPointsFromFILEPointer(FILE* fp) {
		vector<Point> p;
		return p;
	}

	// Unimplemented
	bool writeRandomPointsWithFILEPointer(FILE* fp, vector<Point> p) {
		return false;
	}

	void printVectorPoint(vector<Point> a) {
		printf("SIZE : %d[", a.size());
		for (int i = 0; i < a.size(); i++) {
			cout << a[i] << endl;
		}
		printf("]", a.size());
	}

	Mat smartResize(Mat input, int bgVal) {
		Mat mat;
		int uppest = 0;
		int lowest = 0;
		int rightest = 0;
		int leftest = 0;
		for (int i = 0; i < input.rows; i++) {
			for (int j = 0; j < input.cols; j++) {
				if (input.at<uchar>(i, j) != bgVal) {
					lowest = i;
				}
			}
		}

		for (int i = input.rows - 1; i >= 0; i--) {
			for (int j = 0; j < input.cols; j++) {
				if (input.at<uchar>(i, j) != bgVal) {
					uppest = i;
				}
			}
		}

		for (int j = 0; j < input.cols; j++) {
			for (int i = 0; i < input.rows; i++) {
				if (input.at<uchar>(i, j) != bgVal) {
					leftest = j;
				}
			}
		}

		for (int j = input.cols - 1; j >= 0; j--) {
			for (int i = input.rows - 1; i >= 0; i--) {
				if (input.at<uchar>(i, j) != bgVal) {
					rightest = j;
				}
			}
		}

		Rect roi(rightest, uppest, -rightest + leftest, -uppest + lowest);
		mat = input(roi).clone();
		resize(mat, mat, Size(500, 500));
		copyMakeBorder(mat, mat, 50, 50, 50, 50, BORDER_CONSTANT, Scalar(255));

		return mat;
	}

	bool initialProgram() {
		char *path = (char*)malloc(sizeof(char)* 100);
		FILE *fp;
		//load gate and
		printf("Initial : Reading And gate\n");
		for (int i = 1; i <= AND_GATE_TRAINING_AMOUNT; i++) {
			sprintf(path, "training/and/%d.dat", i);
			fp = fopen(path, "r");
			if (fp == NULL) {
				//create gate info
				sprintf(path, "training/and/%d.png", i);
				Mat mat = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
				resize(mat, mat, Size(500, 500));
				mat = normalizeGrayscale(mat);
				mat = slimLine(mat);
				mat = smartResize(mat, 255);
				mat = cannyInner(mat);
				andgate_training[i - 1] = randomPointsFromMat(mat, SAMPLE_RANDOM_POINTS_AMOUNT);
				//writeRandomPointsWithFILEPointer(fp, andgate_training[i - 1]);
			}
			else {
				andgate_training[i - 1] = readRandomPointsFromFILEPointer(fp);
			}
		}

		printf("Initial : Reading Or gate\n");
		//load gate or
		for (int i = 1; i <= OR_GATE_TRAINING_AMOUNT; i++) {
			sprintf(path, "training/or/%d.dat", i);
			fp = fopen(path, "r");
			if (fp == NULL) {
				//create gate info
				sprintf(path, "training/or/%d.png", i);
				Mat mat = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
				resize(mat, mat, Size(500, 500));
				mat = normalizeGrayscale(mat);
				mat = slimLine(mat);
				mat = smartResize(mat, 255);
				mat = cannyInner(mat);
				orgate_training[i - 1] = randomPointsFromMat(mat, SAMPLE_RANDOM_POINTS_AMOUNT);
				//writeRandomPointsWithFILEPointer(fp, orgate_training[i - 1]);
			}
			else {
				orgate_training[i - 1] = readRandomPointsFromFILEPointer(fp);
			}
		}

		printf("Initial : Reading Not gate\n");
		//load gate not
		for (int i = 1; i <= NOT_GATE_TRAINING_AMOUNT; i++) {
			sprintf(path, "training/not/%d.dat", i);
			fp = fopen(path, "r");
			if (fp == NULL) {
				//create gate info
				sprintf(path, "training/not/%d.png", i);
				Mat mat = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
				resize(mat, mat, Size(500, 500));
				mat = normalizeGrayscale(mat);
				mat = slimLine(mat);
				mat = smartResize(mat, 255);
				mat = cannyInner(mat);
				notgate_training[i - 1] = randomPointsFromMat(mat, SAMPLE_RANDOM_POINTS_AMOUNT);
				//writeRandomPointsWithFILEPointer(fp, notgate_training[i - 1]);
			}
			else {
				notgate_training[i - 1] = readRandomPointsFromFILEPointer(fp);
			}
		}
		return true;
	}
	float minimumCostVariance(vector<Point> a, vector<Point> b) {
		int size = a.size();
		float **costMatrix = calculateCostMatrixFromPoints(a, b);
		int **costIntMatrix = float2DArrayToInt2DArray(costMatrix, size, size);
		hungarian_t prob;
		hungarian_init(&prob, costIntMatrix, size, size, HUNGARIAN_MIN);
		hungarian_solve(&prob);
		//hungarian_print_assignment(&prob);

		float sumCost = hungarianCalculateSumCost2(&prob, costMatrix);
		return sumCost;
	}

	int checkGate(Mat mat);

	int main(int, char**) {
		initialProgram();
		Mat mat = imread("training/or/1.png", CV_LOAD_IMAGE_GRAYSCALE);
		printf("\n RESULT %d", checkGate(mat));
		int n;
		waitKey(0);
		scanf("%d", &n);
		return 0;
	}
}