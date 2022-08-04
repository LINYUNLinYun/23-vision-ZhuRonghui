#include "preimg.h"

Mat Pre_image::findOrange(const Mat input){
    if (input.empty()) {
		cout << "Something is wrong!";
	}
	const int LowH = 11;         
	const int HighH = 34;
	const int LowS = 43;
	const int HighS = 255;
	const int LowV = 46;
	const int HighV = 255;
	Mat hsv, binary;		//HSV和二值图
    //bgr转hsv转binary图
	cvtColor(input, hsv,COLOR_BGR2HSV);		
	inRange(hsv, Scalar(LowH, LowS, LowV), Scalar(HighH, HighS, HighV), binary);	
	//3*3卷积核
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(binary, binary, element, Point(-1, -1), 1);
	dilate(binary, binary, element, Point(-1, -1), 1);
	
    return binary;
}
Mat Pre_image::findBlue(const Mat input){
    if (input.empty()) {
		cout << "Something is wrong!";
	}
	Mat hsv, binary;		//HSV和二值图
    //bgr转hsv转binary图
	cvtColor(input, hsv,COLOR_BGR2HSV);		
	inRange(hsv, Scalar(78, 43, 46), Scalar(124, 255, 255), binary);
	//3*3卷积核
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	//erode(binary, binary, element, Point(-1, -1), 1);
	dilate(binary, binary, element, Point(-1, -1), 1);
	
    return binary;
}

Mat Pre_image::whiteBlurBlue(const Mat input)
{
    vector<Mat> temp;
    split(input, temp);//颜色通道分离
    Mat output;
    output = 2 * temp[0];//进行通道加减
    output -= (temp[2] * 0.5 + temp[1] * 0.5);
    threshold(output, output, 140, 255, THRESH_BINARY);//阈值处理
    return output;
}

Mat Pre_image::whiteBlurRed(const Mat input)
{
    vector<Mat> temp;
    split(input, temp);//颜色通道分离
    Mat output;
    output = (temp[2] * 0.5 + temp[1] * 0.5) * 1.3;//进行通道加减
    output -= temp[0];
    threshold(output, output, 160, 255, THRESH_BINARY);//阈值处理
    return output;
}


void Pre_image::line(Mat inputArray,RotatedRect inputRotateRect){
	Point2f anglePoints[4];
	inputRotateRect.points(anglePoints);
	for (int i = 0; i < 4; i++)
	{
		cv::line(inputArray, anglePoints[i], anglePoints[(i + 1) % 4], Scalar(0, 255, 0), 3);
	}
}
/**
 * @brief 重载
 * 
 * @param inputArray 
 * @param ptr 
 */
void Pre_image::line(Mat inputArray,Point2f ptr[]){
	for (int i = 0; i < 2; i++)
	{
		cv::line(inputArray, ptr[i], ptr[(i + 2) % 4], Scalar(0, 0, 255), 2);
	}

}

bool Pre_image::rightAngle(Point2f a,Point2f b,Point2f c){
	bool flag = false;
	float ab = sqrtf(powf((a.x - b.x),2) + powf((a.y - b.y),2));
	float bc = sqrtf(powf((b.x - c.x),2) + powf((b.y - c.y),2));
	float ac = sqrtf(powf((a.x - c.x),2) + powf((a.y - c.y),2));
	float cosAngleB = (ab*ab + bc*bc - ac*ac)/(2*ab*bc);
	//cout<<"      cangle "<< cosAngleB <<endl;
	double angleB = abs(acosf(cosAngleB)*180/3.1415926);
	//cout<<"      angle "<<angleB<<endl;
	if(angleB > 80 && angleB<100){
		flag = true;
	}
	return flag;
}	

