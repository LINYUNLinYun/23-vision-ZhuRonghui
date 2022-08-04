#include "ros/ros.h"
#include "visualization_msgs/Marker.h"
#include "tf2/LinearMath/Quaternion.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "preimg.h"
#include "Lightbar.h"
#include "armourplate.h"
#include "SolvePNP.h"
#include "geometry_msgs/TransformStamped.h"
#include "tf2_ros/transform_broadcaster.h"
//#include "cv_bridge/cv_bridge.h"
//#include "image_transport/image_transport.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv ){
    Mat frame;
	char key = 0;
    VideoCapture capture;
    SolvePNP pnp;
    capture.open("./2.avi");
    if(!capture.isOpened()){
        cerr<<"wrong open!";
    }

	ros::init(argc,argv,"basic_shapes");
	ros::NodeHandle nh;
	ros::Rate rate(1);
	//创建一个marker类型发布对象
	ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);
	//
  	uint32_t shape = visualization_msgs::Marker::CUBE;

	//
	//image_transport::ImageTransport it(nh);
	//图像发布对象
	//image_transport::Publisher pub = it.advertise("camera/image", 1);

    while(ros::ok()){
    	capture>>frame;
    	Mat binaryInput = frame.clone();
    	Pre_image pImg;
    	if(frame.empty()){
    	    cerr<<"frame wrong!"<<endl;
    	    break;
    	}
	
    	Mat binary = pImg.findOrange(binaryInput);

    	vector<Lightbar> lights;

    	lights = Lightbar::screenToContous(binary);
	
    	lights = Lightbar::screenToRotate(lights);

    	vector<Armour> armour1;
    	for(int i = 0;i<lights.size();i++){
    	    for(int j = i + 1;j<lights.size();j++){
    	        Armour armour(lights[i],lights[j]);
    	        armour1.push_back(Armour(lights[i],lights[j]));    
    	    }
    	}
    	armour1 = Armour::screen(armour1);
    	for(int i = 0;i<armour1.size();i++){
    	    pnp.solution(armour1[i].armourPoints);

    	    Pre_image::line(frame,armour1[i].armourPoints);
    	}
		//创建一个marker对象
		visualization_msgs::Marker marker;
		marker.header.frame_id = "image";
		//marker.
		//设置时间戳
    	marker.header.stamp = ros::Time(0);
		//命名空间的设置
    	marker.ns = "basic_shapes";
    	//设置id,同一个id的marker会被覆盖
    	marker.id = 0;
		//cube
		marker.type = shape;
		//设置mk的类型为新增
		marker.action = visualization_msgs::Marker::ADD;
		//设置位姿
		marker.pose.position.x = pnp.x/1000;
    	marker.pose.position.y = pnp.y/1000;
    	marker.pose.position.z = pnp.z/1000;
		//四元数
		tf2::Quaternion qtn;
		qtn.setRPY(pnp.thetax,pnp.thetay,pnp.thetaz);
    	marker.pose.orientation.x = qtn.getX();
    	marker.pose.orientation.y = qtn.getY();
    	marker.pose.orientation.z = qtn.getZ();
    	marker.pose.orientation.w = qtn.getW();
		//设置大小
		marker.scale.x = 0.138;
    	marker.scale.y = 0.052;
    	marker.scale.z = 0.01;
		//设置颜色
		marker.color.r = 1.0f;
    	marker.color.g = 0.0f;
    	marker.color.b = 1.0f;
    	marker.color.a = 1;
		//持续时间为永久
		marker.lifetime = ros::Duration();
		//等待订阅
		while (marker_pub.getNumSubscribers() < 1)
    	{
    	  if (!ros::ok())
    	  {
    	    return 0;
    	  }
    	  ROS_WARN_ONCE("Please create a subscriber to the marker");
    	  sleep(1);
    	}
		marker_pub.publish(marker);
		//sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    	key = waitKey(33);
		
		//pub.publish(msg);
		//回旋函数
		//ros::spinOnce();
		if(key == 27){
			break;
		}
    }
    capture.release();
    return 0;
}
