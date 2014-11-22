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



void displayFloatMatrix(float** f, int size);

Mat normalizeGrayscale(Mat mat) {
	Scalar avgGrayscaleValue = mean(mat);
	double multiplier = 0.80;
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			float grayscaleValue = ((int)mat.at<uchar>(i, j)) * 1.0;
			if (grayscaleValue < avgGrayscaleValue[0] * multiplier) {
				mat.at<uchar>(i, j) = 0;
			}
			else {
				mat.at<uchar>(i, j) = 255;
			}
		}
	}
	return mat;
}

int erosion_size = 5;
int pre_dilation_size = 2;
int dilation_size = 2;
int const max_kernel_size = 21;

void yolo(Mat mat) {
	imshow("YOLO", mat);
	waitKey(0);
}

Mat slimLine(Mat mat) {
	copyMakeBorder(mat, mat, 50, 50, 50, 50, BORDER_CONSTANT, Scalar(255));
	Mat smoothed;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * pre_dilation_size + 1, 2 * pre_dilation_size + 1), Point(pre_dilation_size, pre_dilation_size));
	dilate(mat, smoothed, element);
	element = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
	erode(smoothed, smoothed, element);
	element = getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
	//copyMakeBorder(smoothed, smoothed, -20, -20, -20, -20, BORDER_REPLICATE, Scalar(0));
	return smoothed;
}

Mat cannyInner(Mat mat) {
	Mat element;
	Mat smoothed;
	element = getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
	dilate(mat, smoothed, element);
	Canny(smoothed, smoothed, 100, 300, 3);
	//yolo(smoothed);

	element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
	dilate(smoothed, smoothed, element);
	bool stop = false;
	for (int i = 0; i < smoothed.rows; i++) {
		for (int j = 0; j < smoothed.cols; j++) {
			//printf("%d ", mat.at<uchar>(i, j));
			if (mat.at<uchar>(i, j) == 0) {
				floodFill(smoothed, Point(i, j), Scalar(255));
				floodFill(smoothed, Point(i, j), Scalar(0));
				stop = true;
				break;
			}
		}
		//printf("\n");
		if (stop)
			break;
	}
	erode(smoothed, smoothed, element);
	//copyMakeBorder(smoothed, smoothed, -20, -20, -20, -20, BORDER_REPLICATE, Scalar(0));
	return smoothed;
}

vector<Point> randomPointsFromMat(Mat mat, int amount) {
	vector<Point> points;
	Mat mat2;
	cvtColor(mat, mat2, CV_GRAY2RGB);
	int amountOfWhiteDots = 0;
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			if (mat.at<uchar>(i, j) == 255) {
				amountOfWhiteDots++;
			}
		}
	}
	int pointSelected = 0;
	int error = amount / 5;
	bool stop = false;
	while (!stop) {
		for (int i = 0; i < mat.rows; i++) {
			for (int j = 0; j < mat.cols; j++) {
				if (mat.at<uchar>(i, j) == 255) {
					//probability calculation
					int randVal = rand() % (amountOfWhiteDots + 1);
					if (randVal < amount - error) {
						points.push_back(Point(j, i));
						if (++pointSelected == amount) {
							stop = true;
							break;
						}
					}
				}
				if (stop)
					break;
			}
			if (stop)
				break;
		}
	}
	return points;
}

float fastSqrt(float x) {
	unsigned int i = *(unsigned int*)&x;

	// adjust bias
	i += 127 << 23;
	// approximation of square root
	i >>= 1;

	return *(float*)&i;
}

float euclideanDistance(Point p1, Point p2) {
	Point vec = p1 - p2;
	float dist2 = vec.ddot(vec);
	//return fastSqrt(dist2);
	return sqrt(dist2);
}

void displayFloatMatrix(float** f, int size) {
	printf("Matrix Display : \n");
	for (int i = 0; i < size; i++) {
		printf("[ ");
		for (int j = 0; j < size; j++) {
			printf("%7.2f", f[i][j]);
		}
		printf("]\n");
	}
	return;
}

void displayIntMatrix(int** x, int size) {
	printf("Int Matrix Display : \n");
	for (int i = 0; i < size; i++) {
		printf("[ ");
		for (int j = 0; j < size; j++) {
			printf("%7d", x[i][j]);
		}
		printf("]\n");
	}
	return;
}

/*vector<Histogram> calculateHistogramFromPoints(vector<Point> points) {
	vector<Histogram> h;
	for (int i = 0; i < points.size(); i++) {
		Histogram g;
		g.histogram = malloc(sizeof(float) * )
		for (int j = 0; j < points.size(); i++) {
		}
	}
}*/

float** calculateCostMatrixFromPoints(vector<Point> pointsMat1, vector<Point> pointsMat2) {
	float ** a;
	a = new float*[pointsMat1.size()];
	for (int i = 0; i < pointsMat1.size(); i++) {
		a[i] = new float[pointsMat1.size()];
	}
	
	for (int i = 0; i < pointsMat1.size(); i++) {
		for (int j = 0; j < pointsMat1.size(); j++) {
			a[i][j] = euclideanDistance(pointsMat1[i], pointsMat2[j]);
		}
	}
	return a;
}

Mat plotDotsVector(Mat mat, vector<Point> vp) {
	Mat matout;
	mat.copyTo(matout);
	for (int i = 0; i < vp.size(); i++) {
		circle(matout, Point(vp[i].x, vp[i].y), 4, Scalar(255), -1);
	}
	return matout;
}

Mat plotLinesVector(Mat mat, vector<Point> from, vector<Point> to) {
	Mat matout;
	mat.copyTo(matout);
	for (int i = 0; i < from.size(); i++) {
		line(matout, from[i], to[i], Scalar(255), 5);
	}
	return matout;
}

int** float2DArrayToInt2DArray(float** a, int row, int col) {
	int **d;
	d = (int **)malloc(sizeof(int *) * row);
	for (int i = 0; i < row; i++){
		d[i] = (int *)malloc(sizeof(int)* col);
		for (int j = 0; j < col; j++) {
			d[i][j] = (int)(a[i][j] * 10000);
		}
	}
	return d;
}

float hungarianCalculateSumCost2(hungarian_t* prob, float **costMatrix)
{
	float sum = 0;
	int i, j;
	for (i = 0; i<prob->m; i++)
	{
		for (j = 0; j < prob->n; j++) {
			sum += (j == prob->a[i]) ? costMatrix[i][j] * costMatrix[i][j] : 0;
		}
	}
	return sum;
}

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

	Rect roi(rightest, uppest, - rightest + leftest, - uppest + lowest);
	mat = input(roi).clone();
	resize(mat, mat, Size(500, 500));
	copyMakeBorder(mat, mat, 50, 50, 50, 50, BORDER_CONSTANT, Scalar(255));

	return mat;
}

bool initialProgram() {
	char *path = (char*) malloc(sizeof(char) * 100);
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
		} else {
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
		} else {
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
		} else {
			notgate_training[i - 1] = readRandomPointsFromFILEPointer(fp);
		}
	}
	return true;
}

#define AND_GATE 101
#define OR_GATE 102
#define NOT_GATE 103
#define INF 1000000000000

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

int checkGate(Mat mat) {
	resize(mat, mat, Size(500, 500));
	//printf("YOLO1");
	mat = normalizeGrayscale(mat);
	yolo(mat);
	mat = slimLine(mat);
	yolo(mat);
	mat = smartResize(mat, 255);
	yolo(mat);
	mat = cannyInner(mat);
	yolo(mat);
	printf("CheckGate : Randoming Points from mats\n");
	vector<Point> matPoints = randomPointsFromMat(mat, SAMPLE_RANDOM_POINTS_AMOUNT);
	
	float minCost = INF;
	int gate = 0;

	printf("CheckGate : Checking with And gate\n");
	//printf("YOLO1");
	//Check And
	for (int i = 0; i < AND_GATE_TRAINING_AMOUNT; i++) {
		//printf("YOLO2");
		float cost = minimumCostVariance(matPoints, andgate_training[i]);
		printf("AND : %f\n", cost);
		if (cost < minCost) {
			minCost = cost;
			gate = AND_GATE;
		}
	}

	printf("CheckGate : Checking with OR gate\n");
	//Check OR
	for (int i = 0; i < OR_GATE_TRAINING_AMOUNT; i++) {
		float cost = minimumCostVariance(matPoints, orgate_training[i]);
		printf("OR : %f\n", cost);
		if (cost < minCost) {
			minCost = cost;
			gate = OR_GATE;
		}
	}

	printf("CheckGate : Checking with Not gate\n");
	//Check Not
	for (int i = 0; i < NOT_GATE_TRAINING_AMOUNT; i++) {
		float cost = minimumCostVariance(matPoints, notgate_training[i]);
		printf("Not : %f\n", cost);
		if (cost < minCost) {
			minCost = cost;
			gate = NOT_GATE;
		}
	}

	printf("%f", minCost);
	
	return gate;
}

int main(int, char**) {
	initialProgram();
	Mat mat = imread("training/or/1.png", CV_LOAD_IMAGE_GRAYSCALE);
	printf("\n RESULT %d", checkGate(mat));
	int n;
	waitKey(0);
	scanf("%d", &n);
	return 0;
}