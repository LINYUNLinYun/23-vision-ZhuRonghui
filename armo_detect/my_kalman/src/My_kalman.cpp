#include "My_kalman.h"

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

