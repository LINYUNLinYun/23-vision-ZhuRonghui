#pragma once

#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class My_kalman{
public:
    /**
     * @brief 启动卡尔曼预测器的初始化函数
     * 
     * @return true 
     * @return false 
     */
    void kalman_init();
    /**
     * @brief 预测函数,对当前位置点进行下一帧的预测，返回预测的坐标点
     * 
     * @param target_centre 
     * @return Point2f 
     */
    Point2f kalman_predict(Point2f target_centre);

private:
    //当前状态值6×1向量(x,y)
    const int stateNum = 6;	
    //测量值6×1向量(x,y)
    const int measureNum = 6;                               	
    //KalmanFilter KF(stateNum, measureNum, 0);		//实例化卡尔曼滤波类
    //卡尔曼滤波参考系下的时间
    const int T = 1;
    //基于卡尔曼滤波预测的预测点
    Point2f predict_point;
    //卡尔曼滤波参考系下的时间的总和（用于算法）
    int sumTime = 0;
    //当前帧、上一帧、上上一帧的时间
    int T_time[3];						
    int t_count = 0;					//计数
    Point2f llastPoint, lastPoint, nowPoint;				//储存当前帧、上一帧、上上一帧的点信息

    KalmanFilter KF = KalmanFilter(stateNum,measureNum,0);		//实例化卡尔曼滤波类
    Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
    //Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
};