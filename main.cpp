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
#define SAMPLE_RANDOM_POINTS_AMOUNT 185
#define MIN_NORMALIZE_GRAYSCALE 100
#define SHADOW_LOWEST_VAL 200

void displayFloatMatrix(float** f, int size);
Mat rotation(Mat src, vector<Point2f> axis);
vector<Point2f> majorAxis(vector<Point> feature);
vector<Point> featurePoint(Mat src_in, int size);
Mat cannyInnerForSlimEdge(Mat mat);

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
	cout << "Threshold Pass : " << avgGrayscaleValue << endl;
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
		Mat img = mat(Rect(0, 0, MIN_NORMALIZE_GRAYSCALE, rows));
		thresholdPass(img);
		Mat img2 = mat(Rect(MIN_NORMALIZE_GRAYSCALE, 0, cols - MIN_NORMALIZE_GRAYSCALE, rows));
		img2 = normalizeGrayscale(img2);
		hconcat(img, img2, res);
		return res;
	} if (rows > MIN_NORMALIZE_GRAYSCALE) {
		//cout << "Enter rows exceeds" << endl;
		Mat res;
		Mat img = mat(Rect(0, 0, cols, MIN_NORMALIZE_GRAYSCALE));
		thresholdPass(img);
		Mat img2 = mat(Rect(0, MIN_NORMALIZE_GRAYSCALE, cols, rows - MIN_NORMALIZE_GRAYSCALE));
		img2 = normalizeGrayscale(img2);
		vconcat(img, img2, res);
		return res;
	} else {
		cout << "Enter normal op" << endl;
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

	Rect roi(rightest, uppest, -rightest + leftest + 1, -uppest + lowest + 1);
	mat = input(roi).clone();
	resize(mat, mat, Size(500, 500));
	copyMakeBorder(mat, mat, 50, 50, 50, 50, BORDER_CONSTANT, Scalar(bgVal));

	return mat;
}

bool initialProgram() {
	char *path = (char*)malloc(sizeof(char)* 100);
	FILE *fp;
	//load gate and
	printf("Initial : Reading And gate\n");

#pragma omp parallel for shared(andgate_training)
	for (int i = 1; i <= AND_GATE_TRAINING_AMOUNT; i++) {
		sprintf(path, "training/and/%d.dat", i);
		fp = fopen(path, "r");
		if (fp == NULL) {
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
		}
		else {
			andgate_training[i - 1] = readRandomPointsFromFILEPointer(fp);
		}
	}

	printf("Initial : Reading Or gate\n");
	//load gate or
#pragma omp parallel for shared(orgate_training)
	for (int i = 1; i <= OR_GATE_TRAINING_AMOUNT; i++) {
		sprintf(path, "training/or/%d.dat", i);
		fp = fopen(path, "r");
		if (fp == NULL) {
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
		}
		else {
			orgate_training[i - 1] = readRandomPointsFromFILEPointer(fp);
		}
	}

	printf("Initial : Reading Not gate\n");
	//load gate not
#pragma omp parallel for shared(notgate_training)
	for (int i = 1; i <= NOT_GATE_TRAINING_AMOUNT; i++) {
		sprintf(path, "training/not/%d.dat", i);
		fp = fopen(path, "r");
		if (fp == NULL) {
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
		}
		else {
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

int countRegions(Mat src) {
	int r = src.rows, c = src.cols, out = 0;
	Mat bin;
	src.convertTo(bin, CV_8U);
	uchar* data = bin.data;
	for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) if (data[i * c + j] == 255) {
		out++;
		floodFill(bin, Point(j, i), Scalar(out));
	}
	return out;
}

vector<Point> featurePoint(Mat src_in, int size) {
	int pre_dilation_size = 7;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * pre_dilation_size + 1, 2 * pre_dilation_size + 1), Point(pre_dilation_size, pre_dilation_size));

	Mat src;
	src_in.copyTo(src);
	dilate(src, src, element);

	int r = src.rows, c = src.cols, count = 0;
	Mat dst = Mat::zeros(r, c, CV_8U);
	uchar *out = dst.data;

	cout << "from i = " << 0 << " to " << r - size << endl;
	#pragma omp parallel for shared(dst)
	for (int i = 0; i <= r - size; i++) {
		cout << "i = " << i << endl;
		#pragma omp parallel for shared(dst)
		for (int j = 0; j <= c - size; j++) {
			Rect roi(j, i, size, size);
			Mat p = src(roi);
			int rc = countRegions(p);
			if (rc > 2) out[(i + size / 2) * c + j + size / 2] = 255;
		}
	}
	erode(dst, dst, getStructuringElement(MORPH_CROSS, Size(size / 3, size / 3)));
	yolo(dst);


	for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) if (out[i * c + j] == 255) {
		count++;
		floodFill(dst, Point(j, i), Scalar(count));
	}


	vector<Point> fp(count);
	vector<double> fpc(count);
	for (int i = 0; i < count; i++) {
		fp[i] = Point(0, 0);
		fpc[i] = 0;
	}
	for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) if (out[i * c + j] > 0) {
		uchar index = out[i * c + j] - 1;
		fp[index] += Point(j, i);
		fpc[index]++;
	}
	for (int i = 0; i < count; i++) fp[i] *= 1 / fpc[i];
	return fp;
}

vector<Point2f> majorAxis(vector<Point> feature) {
	Mat points(feature.size(), 2, CV_32F), label, centers;
	for (int i = 0; i < feature.size(); i++) {
		points.at<float>(i, 0) = feature[i].x;
		points.at<float>(i, 1) = feature[i].y;
	}
	kmeans(points, 2, label, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
	int count[2] = { 0, 0 }, max = 0;
	for (int i = 0; i < feature.size(); i++) {
		int t = label.at<int>(i);
		count[t] ++;
	}
	vector<Point2f> dst(2);
	if (count[0] < count[1]) max = 1;
	dst[0] = Point2f(centers.at<float>(1 - max, 0), centers.at<float>(1 - max, 1));
	dst[1] = Point2f(centers.at<float>(max, 0), centers.at<float>(max, 1));
	return dst;
}

Mat rotation(Mat src, vector<Point2f> axis) {
	double angle = atan2(axis[1].y - axis[0].y, axis[1].x - axis[0].x) * 180 / 3.14159265;
	Mat rt = getRotationMatrix2D(Point2f(src.rows / 2, src.cols / 2), 270 - angle, 1.0);
	warpAffine(src, src, rt, Size(src.rows, src.cols), INTER_CUBIC, BORDER_CONSTANT, Scalar(255, 255, 255));
	return src;
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

int checkGate(Mat mat) {
	//printf("YOLO1");
	int gate = 0;
	resize(mat, mat, Size(500, 500));
	mat = contrastGrayscale(mat);
	mat = normalizeGrayscale(mat);
	mat = slimLine(mat);
	vector<Point> fp = featurePoint(mat, 100);
	vector<Point2f> ma = majorAxis(fp);
	mat = rotation(mat, ma);
	mat = smartResize(mat, 255);
	mat = cannyInner(mat);
	bitwise_not(mat, mat);

	mat = contrastGrayscale(mat);
	mat = normalizeGrayscale(mat);
	mat = smartResize(mat, 255);
	mat = normalizeGrayscale(mat);
	mat = cannyInnerForSlimEdge(mat);

	yolo3("Checking This gate", mat);
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

vector<Rect> boundary(Mat src) {
	Mat bin = src.clone();
	vector<vector<Point>> contours;
	findContours(bin, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point());
	vector<vector<Point>> contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}
	for (int i = contours.size() - 1; i >= 0; i --)
		if (boundRect[i].width < 10 || boundRect[i].height < 10) boundRect.erase(boundRect.begin() + i);
		else if (boundRect[i].width > 2 * src.rows / 3 || boundRect[i].height > 2 * src.cols / 3) boundRect.erase(boundRect.begin() + i);
		else if (boundRect[i].x < 0 || boundRect[i].y < 0 || boundRect[i].x + boundRect[i].height > src.cols || boundRect[i].y + boundRect[i].width > src.rows) boundRect.erase(boundRect.begin() + i);
	return boundRect;
}

void expand(Rect *src, Size *border, int size) {
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
	if ((*src).y + (*src).height + size > (*border).width) add.height -= (*src).y + (*src).height + size - (*border).width;
	if ((*src).x + (*src).width + size > (*border).height) add.width -= (*src).x + (*src).width + size - (*border).height;
	*src -= sub;
	*src += add;
}

vector<Mat> regions(Mat src, int size) {
	Mat bin = src.clone(), map = Mat::zeros(src.rows, src.cols, CV_8U), temp;
	yolo3("Main", bin);
	bin = normalizeGrayscale(bin);
	dilate(bin, bin, getStructuringElement(MORPH_ELLIPSE, Size(1.5, 1.5)));
	erode(bin, bin, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	//bin = slimLineWithOutResize(bin);
	bin = cannyInnerForSlimEdge(bin);
	yolo3("Main", bin);
	vector<Rect> roi = boundary(bin);
	for (int i = 0; i < roi.size(); i++) {
		expand(&roi[i], &Size(src.rows, src.cols), size);
		rectangle(map, roi[i].tl(), roi[i].br(), Scalar(255), CV_FILLED, 8, 0);
	}
	roi = boundary(map);
	vector<Mat> dst(roi.size());
	for (int i = 0; i < roi.size(); i++) {
		expand(&roi[i], &Size(src.rows, src.cols), size);
		dst[i] = src(roi[i]);
	}
	return dst;
}

int main2() {
	initialProgram();
	Mat mat = imread("test5.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	vector<Mat> crop = regions(mat, region_border_size);
	for (int i = 0; i < crop.size(); i++) {
		yolo3("GATE", crop[i]);
		printf("\n RESULT %d", checkGate(crop[i]));
	}
	return 0;
}

int main() {
	main2();
	/*Mat a = imread("test4.jpg"), aa, b;
	GaussianBlur(a, a, Size(3, 3), 3);
	GaussianBlur(a, aa, Size(0, 0), 3);
	addWeighted(a, 1.5, aa, -0.5, 0, a);
	int size = 10;
	cvtColor(a, aa, CV_BGR2GRAY);
	threshold(aa, aa, 128, 255, THRESH_BINARY);
	vector<Mat> crop = regions(aa, size);
	for (int i = 0; i < crop.size(); i++) {
		Mat mask = featurePoint(crop[i], size);
		Mat aaa[3];
		aaa[0] = crop[i] & ~mask;
		aaa[1] = aaa[0];
		aaa[2] = crop[i] | mask;
		merge(aaa, 3, crop[i]);
		char p[20];
		sprintf(p, "Crop[%d]", i);
		imshow(p, crop[i]);
	}
	waitKey(0);*/
	return 0;
}