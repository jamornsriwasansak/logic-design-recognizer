#include <codecvt>
#include "application.h"
#include "view.h"
#include "method_dispatcher.h"
#include <Awesomium/WebCore.h>
#include <Awesomium/STLHelpers.h>
#ifdef _WIN32
#include <Windows.h>
#endif

// OpenCV part
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

#include "hungarian.h"

#include <fstream>
#include <iostream>
using namespace cv;
using namespace std;


#define AND_GATE_TRAINING_AMOUNT 4
#define OR_GATE_TRAINING_AMOUNT 4
#define NOT_GATE_TRAINING_AMOUNT 4
#define AND_TEXT "and"
#define OR_TEXT "or"
#define NOT_TEXT "not"
#define SAMPLE_RANDOM_POINTS_AMOUNT 185
#define MIN_NORMALIZE_GRAYSCALE 100
#define SHADOW_LOWEST_VAL 200
#define PI 3.14159265

void displayFloatMatrix(float** f, int size);
Mat rotation(Mat src, vector<Point2f> axis);
vector<Point2f> majorAxis(vector<Point> feature);
vector<Point> featurePoint(Mat src_in, int size);
Mat cannyInnerForSlimEdge(Mat mat);

typedef struct wire {
	bool components_indexes[100];
}Wire;

vector<Wire*> wires;

typedef struct component {
	int gate_type;
	int in_use = 0;
	Wire* in_1;
	Wire* in_2;
	float rotation;

	Wire* out_1;
	Point center;
	cv::Rect border;
	Scalar color;
} Component;

vector<Component*> components;


void yolo(Mat mat) {
	imshow("YOLO", mat);
	waitKey(0);
}

void yolo2(Mat mat) {
	Mat mat2;
	resize(mat, mat2, Size(800, 450));
	imshow("YOLO", mat2);
	waitKey(0);
}

void yolo3(char *c, Mat mat) {
	imshow(c, mat);
	waitKey(0);
}

Mat thresholdPass(Mat mat) {
	Scalar avgGrayscaleValue = mean(mat);
	avgGrayscaleValue[0] = (avgGrayscaleValue[0] > SHADOW_LOWEST_VAL) ? SHADOW_LOWEST_VAL : avgGrayscaleValue[0];
	//cout << "Threshold Pass : " << avgGrayscaleValue << endl;
	double multiplier = 0.80;
	threshold(mat, mat, avgGrayscaleValue[0] * multiplier, 255, CV_THRESH_BINARY);
	return mat;
}

Mat normalizeGrayscale(Mat mat) {
	int rows = mat.rows;
	int cols = mat.cols;
	if (cols > MIN_NORMALIZE_GRAYSCALE) {
		//cout << "Enter cols exceeds" << endl;
		Mat res;
		Mat img = mat(cv::Rect(0, 0, MIN_NORMALIZE_GRAYSCALE, rows));
		thresholdPass(img);
		Mat img2 = mat(cv::Rect(MIN_NORMALIZE_GRAYSCALE, 0, cols - MIN_NORMALIZE_GRAYSCALE, rows));
		img2 = normalizeGrayscale(img2);
		hconcat(img, img2, res);
		return res;
	} if (rows > MIN_NORMALIZE_GRAYSCALE) {
		//cout << "Enter rows exceeds" << endl;
		Mat res;
		Mat img = mat(cv::Rect(0, 0, cols, MIN_NORMALIZE_GRAYSCALE));
		thresholdPass(img);
		Mat img2 = mat(cv::Rect(0, MIN_NORMALIZE_GRAYSCALE, cols, rows - MIN_NORMALIZE_GRAYSCALE));
		img2 = normalizeGrayscale(img2);
		vconcat(img, img2, res);
		return res;
	}
	else {
		//cout << "Enter normal op" << endl;
		return thresholdPass(mat);
	}
	return mat;
}

Mat contrastGrayscale(Mat mat) {
	Scalar avgGrayscaleValue = mean(mat);
	double multiplier = 0.70;
	double alpha = 0.8;
	Mat bg = Mat::ones(mat.rows, mat.cols, mat.type()) * avgGrayscaleValue[0] * multiplier, diff;
	subtract(mat, bg, diff);
	addWeighted(diff, alpha, bg, 1 - alpha, 0, mat);
	return diff;
}

int erosion_size = 5;
int pre_dilation_size = 2;
int dilation_size = 2;
int const max_kernel_size = 21;
int region_border_size = 10;


Mat slimLineWithOutResize(Mat mat) {
	Mat smoothed;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * pre_dilation_size + 1, 2 * pre_dilation_size + 1), Point(pre_dilation_size, pre_dilation_size));
	dilate(mat, smoothed, element);
	element = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
	erode(smoothed, smoothed, element);
	element = getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
	//copyMakeBorder(smoothed, smoothed, -20, -20, -20, -20, BORDER_REPLICATE, Scalar(0));
	return smoothed;
}

Mat slimLine(Mat mat) {
	copyMakeBorder(mat, mat, 50, 50, 50, 50, BORDER_CONSTANT, Scalar(255));
	return slimLineWithOutResize(mat);
}

Mat cannyInner(Mat mat) {
	Mat element;
	Mat smoothed;
	element = getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
	dilate(mat, smoothed, element);
	Canny(smoothed, smoothed, 100, 300, 3);

	element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
	dilate(smoothed, smoothed, element);
	bool stop = false;
	floodFill(smoothed, Point(0, 0), Scalar(255));
	floodFill(smoothed, Point(0, 0), Scalar(0));
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
}//

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
	d = (int **)malloc(sizeof(int *)* row);
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
	for (i = 0; i < prob->m; i++)
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

	cv::Rect roi(rightest, uppest, -rightest + leftest + 1, -uppest + lowest + 1);
	mat = input(roi).clone();
	resize(mat, mat, Size(500, 500));
	copyMakeBorder(mat, mat, 50, 50, 50, 50, BORDER_CONSTANT, Scalar(bgVal));

	return mat;
}

bool initialProgram() {
	char *path = (char*)malloc(sizeof(char)* 100);

	//load gate and
	printf("Initial : Reading And gate\n");
	//#pragma omp parallel for shared(andgate_training)
	for (int i = 1; i <= AND_GATE_TRAINING_AMOUNT; i++) {
		/*FILE *fp;
		sprintf(path, "training/and/%d.dat", i);
		fp = fopen(path, "r");
		if (fp == NULL) {*/
		//create gate info
		sprintf(path, "training/and/%d.png", i);
		Mat mat = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		resize(mat, mat, Size(500, 500));
		mat = contrastGrayscale(mat);
		mat = normalizeGrayscale(mat);
		mat = normalizeGrayscale(mat);
		mat = slimLine(mat);
		mat = smartResize(mat, 255);
		mat = cannyInner(mat);
		bitwise_not(mat, mat);

		mat = contrastGrayscale(mat);
		mat = normalizeGrayscale(mat);
		mat = smartResize(mat, 255);
		mat = normalizeGrayscale(mat);
		mat = cannyInnerForSlimEdge(mat);
		andgate_training[i - 1] = randomPointsFromMat(mat, SAMPLE_RANDOM_POINTS_AMOUNT);
		//writeRandomPointsWithFILEPointer(fp, andgate_training[i - 1]);
		/*}
		else {
		andgate_training[i - 1] = readRandomPointsFromFILEPointer(fp);
		}
		fclose(fp);*/
	}

	printf("Initial : Reading Or gate\n");
	//load gate or
	//#pragma omp parallel for shared(orgate_training)
	for (int i = 1; i <= OR_GATE_TRAINING_AMOUNT; i++) {
		/*FILE *fp;
		sprintf(path, "training/or/%d.dat", i);
		fp = fopen(path, "r");
		if (fp == NULL) {*/
		//create gate info
		sprintf(path, "training/or/%d.png", i);
		Mat mat = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		resize(mat, mat, Size(500, 500));
		mat = contrastGrayscale(mat);
		mat = normalizeGrayscale(mat);
		mat = normalizeGrayscale(mat);
		mat = slimLine(mat);
		mat = smartResize(mat, 255);
		mat = cannyInner(mat);
		bitwise_not(mat, mat);

		mat = contrastGrayscale(mat);
		mat = normalizeGrayscale(mat);
		mat = smartResize(mat, 255);
		mat = normalizeGrayscale(mat);
		mat = cannyInnerForSlimEdge(mat);
		orgate_training[i - 1] = randomPointsFromMat(mat, SAMPLE_RANDOM_POINTS_AMOUNT);
		//writeRandomPointsWithFILEPointer(fp, orgate_training[i - 1]);
		/*}
		else {
		orgate_training[i - 1] = readRandomPointsFromFILEPointer(fp);
		}*
		fclose(fp);*/
	}

	printf("Initial : Reading Not gate\n");
	//load gate not
	//#pragma omp parallel for shared(notgate_training)
	for (int i = 1; i <= NOT_GATE_TRAINING_AMOUNT; i++) {
		/*FILE *fp;
		sprintf(path, "training/not/%d.dat", i);
		fp = fopen(path, "r");
		if (fp == NULL) {*/
		//create gate info
		sprintf(path, "training/not/%d.png", i);
		Mat mat = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		resize(mat, mat, Size(500, 500));
		mat = contrastGrayscale(mat);
		mat = normalizeGrayscale(mat);
		mat = normalizeGrayscale(mat);
		mat = slimLine(mat);
		mat = smartResize(mat, 255);
		mat = cannyInner(mat);
		bitwise_not(mat, mat);

		mat = contrastGrayscale(mat);
		mat = normalizeGrayscale(mat);
		mat = smartResize(mat, 255);
		mat = normalizeGrayscale(mat);
		mat = cannyInnerForSlimEdge(mat);
		notgate_training[i - 1] = randomPointsFromMat(mat, SAMPLE_RANDOM_POINTS_AMOUNT);
		//writeRandomPointsWithFILEPointer(fp, notgate_training[i - 1]);
		/*}
		else {
		notgate_training[i - 1] = readRandomPointsFromFILEPointer(fp);
		}
		fclose(fp);*/
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

Mat cannyInnerForSlimEdge(Mat mat) {
	Mat element;
	Mat smoothed;
	int dilation_size = 4;
	element = getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
	erode(mat, smoothed, element);
	Canny(smoothed, smoothed, 100, 300, 3);
	//yolo(smoothed);

	element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
	dilate(smoothed, smoothed, element);
	bool stop = false;
	floodFill(smoothed, Point(0, 0), Scalar(255));
	floodFill(smoothed, Point(0, 0), Scalar(0));
	erode(smoothed, smoothed, element);
	//copyMakeBorder(smoothed, smoothed, -20, -20, -20, -20, BORDER_REPLICATE, Scalar(0));
	return smoothed;
}

bool isSmallMat(Mat mat, float ratioin) {
	int matArea = mat.rows * mat.cols;
	int pixel = 0;
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			int color = mat.at<uchar>(i, j);
			if (color == 255){
				pixel++;
			}
		}
	}
	float ratio = (pixel * 1.0f) / (matArea * 1.0f);
	printf("Ration : %f\n", ratio);
	return ratio < ratioin;
}

float rotate_temp = 0;
float rotate_tan_temp = 0;

Mat smartRotate(Mat mat, Mat gateMask) {
	Mat result = mat.clone();
	Mat inv_mat;
	Mat inv_gateMask;
	bitwise_not(mat, inv_mat);
	bitwise_not(gateMask, inv_gateMask);

	Mat inv_wireOnly;
	bitwise_and(inv_mat, inv_gateMask, inv_wireOnly);
	bitwise_xor(inv_mat, inv_wireOnly, inv_wireOnly);

	Mat wireOnlyColor;
	cvtColor(inv_wireOnly, wireOnlyColor, CV_GRAY2BGR);

	Mat inv_wireOnly_Raped;
	vector<Scalar> colorWire;
	int colorEncoded = 200;
	int colorStep = 150;
	inv_wireOnly.copyTo(inv_wireOnly_Raped);
	int wireAmount = 0;

	vector<Mat> wireDiffes;
	for (int i = 0; i < inv_wireOnly.rows; i++) {
		for (int j = 0; j < inv_wireOnly.cols; j++) {
			if (inv_wireOnly_Raped.at<uchar>(i, j) != 0) {
				floodFill(inv_wireOnly_Raped, Point(j, i), Scalar(0));
				Mat wireDiff;
				inv_wireOnly.copyTo(wireDiff);
				floodFill(wireDiff, Point(j, i), Scalar(0));
				bitwise_xor(wireDiff, inv_wireOnly, wireDiff);
				if (isSmallMat(wireDiff, 0.0015)) {
					bitwise_xor(wireDiff, inv_wireOnly, inv_wireOnly);
				}
				else {
					wireDiffes.push_back(wireDiff);
					wireAmount++;
				}
			}
		}
	}

	Vec4f linee;
	vector<float> angleFloat;
	float sumAngle = 0, max = -PI / 2, min = PI / 2;
	Mat featurePoint = Mat::zeros(mat.rows, mat.cols, CV_8UC3), gate3;
	for (int i = 0; i < wireDiffes.size(); i++) {
		vector<Point> points = randomPointsFromMat(wireDiffes[i], 200);
		fitLine(points, linee, CV_DIST_L2, 0, 0.01, 0.01);
		//printf("%f %f %f %f\n", linee[0], linee[1], linee[2], linee[3]);
		Mat lineM = Mat::zeros(mat.rows, mat.cols, CV_8UC3), lineN, lineO;
		line(lineM, Point(0, (int)linee[3] - linee[2] * linee[1] / linee[0]), Point(mat.cols, (int)linee[3] - (linee[2] - 500) * linee[1] / linee[0]), Scalar(255, 255, 255), 3, 8, 0);
		bitwise_and(wireOnlyColor, lineM, lineN);
		erode(lineN, lineN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(lineM, lineM, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(lineN, lineN, getStructuringElement(MORPH_ELLIPSE, Size(50, 50)));
		Canny(lineN, lineO, 100, 200);
		merge({ lineO, lineO, lineO }, lineN);
		bitwise_and(lineM, lineN, lineN);
		bitwise_or(featurePoint, lineN, featurePoint);
		float angle = atan2(linee[1], linee[0]);
		angleFloat.push_back(angle);
		if (angle > max) max = angle;
		if (angle < min) min = angle;
	}
	if (max - min > 3 * PI / 4) for (int i = 0; i < angleFloat.size(); i++) if (angleFloat[i] < 0) angleFloat[i] += 2 * PI;
	for (int i = 0; i < angleFloat.size(); i++) sumAngle += angleFloat[i];
	sumAngle /= wireDiffes.size();
	merge({ gateMask, gateMask, gateMask }, gate3);
	bitwise_not(featurePoint, featurePoint);
	bitwise_or(gate3, featurePoint, featurePoint);
	bitwise_not(featurePoint, featurePoint);

	//yolo3("fp", featurePoint);

	sumAngle /= wireDiffes.size();
	sumAngle += PI;
	sumAngle = fmodf(sumAngle, PI);
	sumAngle -= PI / 2;



	Mat outSide = Mat::zeros(mat.rows, mat.cols, CV_8U), inSide;
	Point poly[1][4];
	if (sumAngle >= -PI / 4 && sumAngle <= PI / 4) {
		poly[0][0] = Point(0, mat.rows);
		poly[0][1] = Point(mat.cols, mat.rows);
		poly[0][2] = Point(mat.cols, (int)(mat.rows / 2 - mat.cols / 2 * tan(sumAngle)));
		poly[0][3] = Point(0, (int)(mat.rows / 2 + mat.cols / 2 * tan(sumAngle)));
	}
	else {
		poly[0][0] = Point(mat.cols, 0);
		poly[0][1] = Point(mat.cols, mat.rows);
		poly[0][2] = Point((int)(mat.cols / 2 + mat.rows / 2 / tan(sumAngle)), mat.rows);
		poly[0][3] = Point((int)(mat.cols / 2 - mat.rows / 2 / tan(sumAngle)), 0);
	}
	const Point* ppt[1] = { poly[0] };
	int npt[] = { 4 };
	fillPoly(outSide, ppt, npt, 1, Scalar(255));
	bitwise_not(outSide, inSide);

	Scalar r0 = mean(mat, outSide);
	Scalar r1 = mean(mat, inSide);
	if (r0[0] > r1[0]) {
		bitwise_not(outSide, outSide);
		bitwise_not(inSide, inSide);
	}

	vector<Point> pin, pout;
	for (int i = 0; i < featurePoint.rows; i++) for (int j = 0; j < featurePoint.cols; j++) if (featurePoint.at<uchar>(i, j) != 0) {
		if (outSide.at<uchar>(i, j) != 0) pout.push_back(Point(j, i));
		else pin.push_back(Point(j, i));
	}
	if (pin.size() > 0 && pout.size() > 0) {
		sumAngle = 0;
		for (int i = 0; i < pin.size(); i++) for (int j = 0; j < pout.size(); j++) {
			float angle = atan2(pout[j].y - pin[i].y, pout[j].x - pin[j].x);
			sumAngle += angle;
		}
		sumAngle /= (pin.size() * pout.size());
		sumAngle += PI / 2;
	}
	sumAngle = fmodf(sumAngle, PI);
	sumAngle -= PI / 2;
	if (inSide.at<uchar>(mat.cols - 1, mat.rows / 2) != 0) sumAngle += PI;

	rotate_tan_temp = sumAngle;
	rotate_temp = sumAngle / PI * 180;
	/*yolo3("Inv_wireOnly Clean", inv_wireOnly);

	colorStep = 255 * 255 * 255 / wireAmount;

	cvtColor(inv_wireOnly, wireOnlyColor, CV_GRAY2BGR);
	for (int i = 0; i < wireOnlyColor.rows; i++) {
	for (int j = 0; j < wireOnlyColor.cols; j++){
	Vec3b colorPixel = wireOnlyColor.at<Vec3b>(i, j);
	if (colorPixel == Vec3b(255, 255, 255)) {
	int r = colorEncoded % 255;
	int g = (colorEncoded / 255) % 255;
	int b = (colorEncoded / 255 / 255);
	Scalar color(r, g, b);
	cout << color << endl;
	floodFill(wireOnlyColor, Point(j, i), color);
	colorEncoded += colorStep;
	colorWire.push_back(color);
	}
	}
	}

	//dilate(wireOnlyColor, wireOnlyColor, getStructuringElement(MORPH_ELLIPSE, Size(10, 5)));

	//yolo3("Inv_wireOnly Clean", inv_wireOnly);
	yolo3("Inv_wireOnly color Clean", wireOnlyColor);*/

	Mat rotation_matrix = getRotationMatrix2D(Point2f(mat.cols / 2, mat.rows / 2), 90 - sumAngle * 180 / PI, 1.0);
	warpAffine(result, result, rotation_matrix, Size(mat.rows, mat.cols), INTER_LINEAR, BORDER_CONSTANT, Scalar(255));

	return result;
}

int checkGate(Mat mat, Mat gateMask, int componentID) {
	//printf("YOLO1");
	//yolo3("CheckGate", mat);
	int gate = 0;

	resize(mat, mat, Size(500, 500));
	resize(gateMask, gateMask, Size(500, 500));

	gateMask = contrastGrayscale(gateMask);
	gateMask = normalizeGrayscale(gateMask);
	erode(gateMask, gateMask, getStructuringElement(MORPH_ELLIPSE, Size(30, 30)));

	mat = contrastGrayscale(mat);
	mat = normalizeGrayscale(mat);
	mat = slimLineWithOutResize(mat);
	mat = smartRotate(mat, gateMask);
	mat = smartResize(mat, 255);
	mat = cannyInner(mat);
	bitwise_not(mat, mat);

	mat = contrastGrayscale(mat);
	mat = normalizeGrayscale(mat);
	mat = smartResize(mat, 255);
	mat = normalizeGrayscale(mat);
	mat = cannyInnerForSlimEdge(mat);

	printf("CheckGate : Randoming Points from mats\n");
	vector<Point> matPoints = randomPointsFromMat(mat, SAMPLE_RANDOM_POINTS_AMOUNT);

	float minCost = INF;

	printf("CheckGate : Checking with And gate\n");
	//printf("YOLO1");
	//Check And
#pragma omp parallel for shared(gate, minCost, matPoints)
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
#pragma omp parallel for shared(gate, minCost, matPoints)
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
#pragma omp parallel for shared(gate, minCost, matPoints)
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

vector<cv::Rect> boundary(Mat src) {
	Mat bin = src.clone();
	vector<vector<Point>> contours;
	findContours(bin, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point());
	vector<vector<Point>> contours_poly(contours.size());
	vector<cv::Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}
	for (int i = contours.size() - 1; i >= 0; i--)
	if (boundRect[i].width < 10 || boundRect[i].height < 10) boundRect.erase(boundRect.begin() + i);
	else if (boundRect[i].width > 2 * src.rows / 3 || boundRect[i].height > 2 * src.cols / 3) boundRect.erase(boundRect.begin() + i);
	else if (boundRect[i].x < 0 || boundRect[i].y < 0 || boundRect[i].x + boundRect[i].height > src.cols || boundRect[i].y + boundRect[i].width > src.rows) boundRect.erase(boundRect.begin() + i);
	return boundRect;
}

void expand(cv::Rect *src, Size *border, int size) {
	Point sub(size, size);
	Size add(2 * size, 2 * size);
	if ((*src).x < size) {
		sub.x -= size - (*src).x;
		add.height -= size - (*src).x;
	}
	if ((*src).y < size) {
		sub.y -= size - (*src).y;
		add.width -= size - (*src).y;
	}
	if ((*src).y + (*src).height + size >(*border).width) add.height -= (*src).y + (*src).height + size - (*border).width;
	if ((*src).x + (*src).width + size >(*border).height) add.width -= (*src).x + (*src).width + size - (*border).height;
	*src -= sub;
	*src += add;
}

Mat cannyInnerForSlimEdgeMainImage(Mat mat) {
	Mat element;
	Mat smoothed;
	mat.copyTo(smoothed);
	element = getStructuringElement(MORPH_RECT, Size(2, 2), Point(1, 1));
	erode(smoothed, smoothed, element);
	Canny(smoothed, smoothed, 100, 300, 3);

	//yolo(smoothed);

	element = getStructuringElement(MORPH_RECT, Size(2, 2), Point(1, 1));
	dilate(smoothed, smoothed, element);
	bool stop = false;
	floodFill(smoothed, Point(0, 0), Scalar(255));
	floodFill(smoothed, Point(0, 0), Scalar(0));
	erode(smoothed, smoothed, element);
	//copyMakeBorder(smoothed, smoothed, -20, -20, -20, -20, BORDER_REPLICATE, Scalar(0));
	return smoothed;
}

typedef struct regionTuple{
	vector<Mat> regionsMat;
	vector<cv::Rect> regionsRect;
	Mat colorMask;
}RegionsTuple;

regionTuple regions(Mat src, Mat bin, int size) {
	RegionsTuple rt;
	Mat map = Mat::zeros(src.rows, src.cols, CV_8U), temp;
	rt.colorMask = Mat::zeros(src.rows, src.cols, CV_8U), temp;
	cvtColor(rt.colorMask, rt.colorMask, CV_GRAY2BGR);

	vector<cv::Rect> roi = boundary(bin);
	int sideLen = (src.rows > src.cols) ? src.cols : src.rows;
	cout << sideLen;
	size = sideLen * 0.02;
	//yolo3("Bin", src);
	for (int i = 0; i < roi.size(); i++) {
		expand(&roi[i], &Size(src.rows, src.cols), size);
		rectangle(map, roi[i].tl(), roi[i].br(), Scalar(255), CV_FILLED, 8, 0);
	}
	roi = boundary(map);

	vector<Mat> dst(roi.size());
	int allPossibleColor = 255 * 255 * 255;
	int stepColor = allPossibleColor / roi.size();
	int colorEncoded = stepColor;
	for (int i = 0; i < roi.size(); i++) {
		expand(&roi[i], &Size(src.rows, src.cols), size);
		dst[i] = src(roi[i]);
		int r = colorEncoded % 255;
		int g = (colorEncoded / 255) % 255;
		int b = (colorEncoded / 255 / 255);
		rectangle(rt.colorMask, roi[i].tl(), roi[i].br(), Scalar(r, g, b), CV_FILLED, 8, 0);
		Component* component = (Component*)malloc(sizeof(Component));
		component->border = roi[i];
		component->in_use = 0;
		components.push_back(component);
		components[i]->color = Scalar(r, g, b);
		colorEncoded += stepColor;
	}

	rt.regionsMat = dst;
	rt.regionsRect = roi;
	return rt;
}

vector<Mat> invWiresMatFromWireMatInv(Mat mat) {
	vector<Mat> output;
	Mat wireMatInv;
	mat.copyTo(wireMatInv);
	dilate(wireMatInv, wireMatInv, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	//yolo3("wireMatInv", wireMatInv);
	for (int i = 0; i < wireMatInv.rows; i++) {
		for (int j = 0; j < wireMatInv.cols; j++)  {
			int color = wireMatInv.at<uchar>(i, j);
			if (color == 255) {
				Mat wireMatInvTemp;
				wireMatInv.copyTo(wireMatInvTemp);
				floodFill(wireMatInv, Point(j, i), Scalar(0));

				Mat res;
				bitwise_xor(wireMatInvTemp, wireMatInv, res);

				//yolo3("WireMatInv", wireMatInv);
				//yolo3("WireMatInvTemp", wireMatInvTemp);
				if (!isSmallMat(res, 0.00020f)) {
					output.push_back(res);
				}
			}
		}
	}
	return output;
}

//////////////////////////////////Implement Verilog//////////////////////////////////////////////////

const char *outPicture;
const char *outVerilog;

void verilog() {
	vector<Wire*> in_wire, con_wire, out_wire;

	for (int i = 0; i < wires.size(); i++) {
		bool in = false, out = false;
		for (int j = 0; j < components.size(); j++) if (wires[i]->components_indexes[j]) {
			if (components[j]->out_1 == wires[i]) out = true;
			else if (components[j]->in_1 == wires[i]) in = true;
			else if (components[j]->in_2 == wires[i]) in = true;
			if (in && out) break;
		}
		if (in && out) con_wire.push_back(wires[i]);
		else if (out) out_wire.push_back(wires[i]);
		else in_wire.push_back(wires[i]);
	}

	char buf[2000];

	FILE *fwrite = fopen(outVerilog, "w");

	int c_in = in_wire.size();
	int c_out = out_wire.size();
	int c_wire = con_wire.size();

	fprintf(fwrite, "module combination_logic (");
	for (int i = 0; i < c_in; i++) fprintf(fwrite, "i%d, ", i);
	for (int i = 0; i < c_out; i++) {
		if (i != c_out - 1)
			fprintf(fwrite, "o%d, ", i);
		else
			fprintf(fwrite, "o%d", i);
	}
	fprintf(fwrite, ");\n");

	fprintf(fwrite, "\tinput");
	for (int i = 0; i < c_in; i++) {
		if (i != c_in - 1)
			fprintf(fwrite, " i%d,", i);
		else
			fprintf(fwrite, " i%d", i);
	}
	fprintf(fwrite, ";\n\toutput");
	for (int i = 0; i < c_out; i++) {
		if (i != c_out - 1)
			fprintf(fwrite, " o%d,", i);
		else
			fprintf(fwrite, " o%d", i);
	}
	fprintf(fwrite, ";\n\twire");
	for (int i = 0; i < c_wire; i++) {
		if (i != c_wire - 1)
			fprintf(fwrite, " w%d,", i);
		else
			fprintf(fwrite, " w%d", i);
	}
	fprintf(fwrite, ";\n\n");

	int c_and = 0, c_or = 0, c_not = 0;

	for (int i = 0; i < components.size(); i++) {
		fprintf(fwrite, "\t");
		switch (components[i]->gate_type) {
		case AND_GATE:
			fprintf(fwrite, "and an%d(", c_and);
			c_and++;
			break;
		case OR_GATE:
			fprintf(fwrite, "or or%d(", c_or);
			c_or++;
			break;
		case NOT_GATE:
			fprintf(fwrite, "not no%d(", c_not);
			c_not++;
		}

		Wire* k = components[i]->out_1;
		bool found = false;
		for (int j = 0; !found && j < out_wire.size(); j++) if (k == out_wire[j]) {
			fprintf(fwrite, "o%d", j);
			found = true;
		}
		for (int j = 0; !found && j < con_wire.size(); j++) if (k == con_wire[j]) {
			fprintf(fwrite, "w%d", j);
			found = true;
		}

		k = components[i]->in_1;
		found = false;
		for (int j = 0; !found && j < in_wire.size(); j++) if (k == in_wire[j]) {
			fprintf(fwrite, ", i%d", j);
			found = true;
		}
		for (int j = 0; !found && j < con_wire.size(); j++) if (k == con_wire[j]) {
			fprintf(fwrite, ", w%d", j);
			found = true;
		}

		if (components[i]->gate_type == NOT_GATE) printf(");\n");
		else {
			k = components[i]->in_2;
			found = false;
			for (int j = 0; !found && j < in_wire.size(); j++) if (k == in_wire[j]) {
				fprintf(fwrite, ", i%d);\n", j);
				found = true;
			}
			for (int j = 0; !found && j < con_wire.size(); j++) if (k == con_wire[j]) {
				fprintf(fwrite, ", w%d);\n", j);
				found = true;
			}
		}
	}

	fprintf(fwrite, "endmodule\n");
	fclose(fwrite);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
Application *app;

void controller(const char *recognizeFile) {
	Mat mat = imread(recognizeFile, CV_LOAD_IMAGE_GRAYSCALE);
	imshow(recognizeFile, mat);
	app->ShowMessage("Click OK and press any key to continue");
	waitKey(0);
	//clean Mat to gates
	Mat bin = mat.clone();
	bin = normalizeGrayscale(bin);
	Mat cleanMat;
	bin.copyTo(cleanMat);
	dilate(bin, bin, getStructuringElement(MORPH_ELLIPSE, Size(1.5, 1.5)));
	erode(bin, bin, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	bin = cannyInnerForSlimEdgeMainImage(bin);

	Mat cleanMatInv;
	bitwise_not(cleanMat, cleanMatInv);
	Mat cleanGate;
	Mat cleanGateInv;
	bin.copyTo(cleanGateInv);
	dilate(cleanGateInv, cleanGateInv, getStructuringElement(MORPH_ELLIPSE, Size(15, 15)));
	bitwise_not(cleanGateInv, cleanGate);

	Mat cleanWireInv;
	bitwise_and(cleanMat, cleanGate, cleanWireInv);
	bitwise_or(cleanWireInv, cleanGateInv, cleanWireInv);
	bitwise_not(cleanWireInv, cleanWireInv);

	Mat mainCut;
	cleanMatInv.copyTo(mainCut);
	RegionsTuple rt = regions(mat, bin, region_border_size);
	Mat colorMask = rt.colorMask;
	Mat grayScaleMask;
	colorMask.copyTo(grayScaleMask);
	cvtColor(grayScaleMask, grayScaleMask, CV_BGR2GRAY);
	grayScaleMask = grayScaleMask > 0;

	//yolo3("Color Mask", rt.colorMask);
	//yolo3("grayScaleMask", grayScaleMask);

	vector<cv::Rect> cropRect = rt.regionsRect;
	for (int i = 0; i < cropRect.size(); i++) {
		//cout << mainCut.size() << endl;
		Point c = ((components[i]->border.tl() + components[i]->border.br())) * 0.5;
		cout << c.x << endl;
		components[i]->center = c;
		circle(mainCut, c, 2, Scalar(255, 0, 0));

		rectangle(mainCut, components[i]->border.tl(), components[i]->border.br(), Scalar(255));
	}
	//circle(mainCut, Point(100, 100), 400, Scalar(255, 0, 0));
	imshow(recognizeFile, mainCut);
	app->ShowMessage("Click OK and press any key to continue");
	waitKey(0);

	initialProgram();
	//Check Gate
	for (int i = 0; i < rt.regionsMat.size(); i++) {
		//yolo3("GATE", crop[i]);
		Mat gate = cleanMatInv(cropRect[i]);
		bitwise_not(gate, gate);
		int gateNum = checkGate(mat(cropRect[i]), cleanGate(cropRect[i]), i);
		components[i]->gate_type = gateNum;
		components[i]->rotation = rotate_tan_temp;

		float angle = components[i]->rotation + PI;
		float tann = tan(angle);
		printf("tann %f", tann);
		Point p1 = components[i]->border.tl();
		Point p2 = components[i]->border.br();

		Point c = (p1 + p2) * 0.5;
		Point c2(c.x + 100, c.y - tann * c.x + tann * (c.x + 100));


		//cout << "C1" << c << endl;
		//cout << "C2" << c2 << endl;

		line(mainCut, c, c2, Scalar(255));

		printf("Angle : %f\n", rotate_temp);
		//printf("\n RESULT %d", gateNum);
		int textSize = 1;
		if (gateNum == AND_GATE) {
			putText(mainCut, AND_TEXT, cropRect[i].tl(), CV_FONT_HERSHEY_COMPLEX, textSize, Scalar(150));
		}
		else if (gateNum == OR_GATE) {
			putText(mainCut, OR_TEXT, cropRect[i].tl(), CV_FONT_HERSHEY_COMPLEX, textSize, Scalar(150));
		}
		else if (gateNum == NOT_GATE) {
			putText(mainCut, NOT_TEXT, cropRect[i].tl(), CV_FONT_HERSHEY_COMPLEX, textSize, Scalar(150));
		}
	}

	imshow(recognizeFile, mainCut);
	imwrite(outPicture, mainCut);
	app->ShowMessage("Click OK and press any key to continue");
	waitKey(0);

	vector<Mat> wiresInvMat = invWiresMatFromWireMatInv(cleanWireInv);

	for (int i = 0; i < wiresInvMat.size(); i++){
		cout << "New Wire" << endl;
		//dilate(wiresInvMat[i], wiresInvMat[i], getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		Wire* wiree = (Wire*)malloc(sizeof(Wire));
		/*init wire*/
		for (int j = 0; j < 100; j++) {
			wiree->components_indexes[j] = false;
		}
		Mat wiresInvMatColor;
		Mat wireConnect;

		cvtColor(wiresInvMat[i], wiresInvMatColor, CV_GRAY2BGR);
		bitwise_and(wiresInvMatColor, colorMask, wireConnect);
		for (int j = 0; j < wireConnect.rows; j++){
			for (int k = 0; k < wireConnect.cols; k++) {
				Vec3b color3b = wireConnect.at<Vec3b>(j, k);
				Scalar colorScalar(color3b[0], color3b[1], color3b[2]);
				for (int l = 0; l < components.size(); l++) {
					if (components[l]->color == colorScalar && !wiree->components_indexes[l]) {
						float compAngle = components[l]->rotation + PI;
						cout << "CompAngle" << components[l]->rotation << endl;
						Point headVector(cos(compAngle) * 10000, sin(compAngle) * 10000);
						cout << "Cos " << cos(compAngle) << "Sin " << sin(compAngle) << endl;
						Point vectorr = Point(k, j) - components[l]->center;
						cout << "Center" << components[l]->center << endl;
						cout << "Vectorr" << vectorr << endl;
						cout << "HeadVector" << headVector << endl;
						float coss = headVector.ddot(vectorr);
						cout << "COSS" << coss << endl;
						if (coss <= 0) {
							components[l]->out_1 = wiree;
							cout << "Connect Wire to Output of Components" << l << endl;
						}
						else {
							cout << "Connect Wire to Input of Components" << l << endl;
							if (components[l]->in_use == 0) {
								components[l]->in_1 = wiree;
								components[l]->in_use++;
							}
							else {
								components[l]->in_2 = wiree;
							}
						}
						//float c1 = components[l]->
						wiree->components_indexes[l] = true;
						floodFill(wireConnect, Point(k, j), Scalar(0));
						break;
					}
				}
			}
		}

		printf("[");
		for (int j = 0; j < components.size(); j++) {
			if (wiree->components_indexes[j]) {
				printf("%d, ", j);
			}
		}

		Mat tmp;
		bitwise_or(wiresInvMatColor, colorMask, tmp);
		//yolo3("Mat", tmp);
		wires.push_back(wiree);
	}

	cout << endl << endl << endl;
	app->ShowMessage("Processing is done. Press Any Key to continue with verilog");
	waitKey(0);
	verilog();
	app->ShowMessage("End");
	exit(0);
}

using namespace Awesomium;

class TutorialApp : public Application::Listener {
	Application* app_;
	View* view_;
	MethodDispatcher method_dispatcher_;
public:
	TutorialApp()
		: app_(Application::Create()),
		view_(0) {
		app_->set_listener(this);
	}

	virtual ~TutorialApp() {
		if (view_)
			app_->DestroyView(view_);
		if (app_)
			delete app_;
	}

	void Run() {
		app_->Run();
	}

	// Inherited from Application::Listener
	virtual void OnLoaded() {
		view_ = View::Create(800, 300);
		WebView *web_view = view_->web_view();
		BindMethods(web_view);
		WebURL url(WSLit("file:///app.html"));
		view_->web_view()->LoadURL(url);
	}

	// Inherited from Application::Listener
	virtual void OnUpdate() {
	}

	// Inherited from Application::Listener
	virtual void OnShutdown() {
	}

	void BindMethods(WebView *web_view) {
		JSValue result = web_view->CreateGlobalJavascriptObject(WSLit("app"));

		if (result.IsObject()) {
			// Bind our custom method to it.
			JSObject& app_object = result.ToObject();
			method_dispatcher_.Bind(app_object,
				WSLit("sayHello"),
				JSDelegate(this, &TutorialApp::OnSayHello));
		}

		// Bind our method dispatcher to the WebView
		web_view->set_js_method_handler(&method_dispatcher_);
	}

	void OnSayHello(WebView* caller, const JSArray& args) {
		JSValue filedirValue = caller->ExecuteJavascriptWithResult(WSLit("document.getElementById('filedir').value"), WSLit(""));
		JSValue picoutValue = caller->ExecuteJavascriptWithResult(WSLit("document.getElementById('picoutname').value"), WSLit(""));
		JSValue verilogOutValue = caller->ExecuteJavascriptWithResult(WSLit("document.getElementById('verilogoutname').value"), WSLit(""));
		app = app_;
		std::string filedirString = ToString(filedirValue.ToString());
		//app_->ShowMessage(filedirString.c_str());

		std::string picoutFileName = ToString(picoutValue.ToString());
		outPicture = picoutFileName.c_str();

		std::string verilogFileName = ToString(verilogOutValue.ToString());
		outVerilog = verilogFileName.c_str();

		controller(filedirString.c_str());
	}
};

#ifdef _WIN32
int APIENTRY wWinMain(HINSTANCE hInstance, HINSTANCE, wchar_t*,
	int nCmdShow) {
#else
int main() {
#endif

	TutorialApp app;
	app.Run();

	return 0;
}
