#pragma once

#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#define HALF_HEIGHT 26.0
#define HALF_WIDTH 69.0

using namespace std;
using namespace cv;

class SolvePNP{
public:
    SolvePNP(){
        
    }
    /**
     * @brief 求解位姿函数
     * 
     * @param inputPtr 
     */
    void solution(Point2f inputPtr[]);
    //位姿
    double x = 0;
    double y = 0;
    double z = 0;

    double thetax = 0;
    double thetaz = 0;
    double thetay = 0;

    //偏航
    double yaw = 0;
    //俯仰
    double pitch = 0;

private:
    
    //世界坐标系点
    vector<Point3f> _worldPoints = vector<Point3f>{
        //这里选择x轴正方向水平向右，y轴正方形竖直向下，z轴成右手系的世界坐标系
        Point3f(-HALF_WIDTH, HALF_HEIGHT, 0),
        Point3f(-HALF_WIDTH, -HALF_HEIGHT, 0),
        Point3f(HALF_WIDTH, -HALF_HEIGHT, 0),
        Point3f(HALF_WIDTH, HALF_HEIGHT, 0)
    };
    //图像坐标系点
    vector<Point2f> _imgPoints;
    //旋转矩阵
    Mat _rVec = Mat::zeros(3, 1, CV_64FC1);//init rvec
    //平移矩阵
    Mat _tVec = Mat::zeros(3, 1, CV_64FC1);//init tvec
    //相机内参
    Mat _camera_matrix = (Mat_<double>(3,3) << 1.2853517927598091e+03, 0., 3.1944768628958542e+02, 0.,
       1.2792339468697937e+03, 2.3929354061292258e+02, 0., 0., 1. );
    //畸变矩阵
    Mat _distortion_coefficients = (Mat_<double>(5,1)<<-6.3687295852461456e-01, -1.9748008790347320e+00,
       3.0970703651800782e-02, 2.1944646842516919e-03, 0.);

};