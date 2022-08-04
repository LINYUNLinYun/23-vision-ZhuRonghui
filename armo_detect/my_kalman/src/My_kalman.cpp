#include "My_kalman.h"
/**********************************卡尔曼滤波模块开始********************************************/
// const int stateNum = 6;					//当前状态值6×1向量(x,y)
// const int measureNum = 6;                               //测量值6×1向量(x,y)	
// KalmanFilter KF(stateNum, measureNum, 0);		//实例化卡尔曼滤波类
// int T = 1;						//卡尔曼滤波参考系下的时间
// Point2f predict_point;					//基于卡尔曼滤波预测的预测点
// int sumT = 0;						//卡尔曼滤波参考系下的时间的总和（用于算法）
// int T_time[3];						//当前帧、上一帧、上上一帧的时间
// int t_count = 0;					//计数
// Point2f llastp, lastp, nowp;				//储存当前帧、上一帧、上上一帧的点信息
// Mat measurement = Mat::zeros(measureNum, 1, CV_32F);



void My_kalman::kalman_init() {
	
	KF.transitionMatrix = (Mat_<float>(stateNum, measureNum) <<
		1, 0, T, 0, 1 / 2 * T * T, 0,
		0, 1, 0, T, 0, 1 / 2 * T * T,
		0, 0, 1, 0, T, 0,
		0, 0, 0, 1, 0, T,
		0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 1);  //转移矩阵A

	setIdentity(KF.measurementMatrix);                                  //设置测量矩阵H
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));                 //设置系统噪声方差矩阵Q
	setIdentity(KF.measurementNoiseCov, Scalar::all(3e-1));             //设置测量噪声方差矩阵R
	setIdentity(KF.errorCovPost, Scalar::all(1));                       //设置后验错误估计协方差矩阵P
	
}


Point2f My_kalman::kalman_predict(Point2f target_centre) {
	Mat prediction = KF.predict();			                //预测
	//Mat measurement = Mat::zeros(measureNum, 1, CV_32F);	
	llastPoint = lastPoint;					                //在新的一帧里，更新点的信息
	lastPoint = nowPoint;
	nowPoint = target_centre;
	predict_point = Point(prediction.at<float>(0), prediction.at<float>(1));//获取预测值(x',y')
	measurement.at<float>(0) = target_centre.x;			//更新测量值
	measurement.at<float>(1) = target_centre.y;
	measurement.at<float>(2) = (nowPoint.x - lastPoint.x)/T;                     //速度
	measurement.at<float>(3) = (nowPoint.y - lastPoint.y)/T;
	measurement.at<float>(4) = ((nowPoint.x - lastPoint.x) - (lastPoint.x - llastPoint.x))/T;   //加速度
	measurement.at<float>(5) = ((nowPoint.y - lastPoint.y) - (lastPoint.y - llastPoint.y))/T;
	KF.correct(measurement);			                        //根据测量值修正协方差矩阵
	return predict_point;				                        //返回预测的点
}

