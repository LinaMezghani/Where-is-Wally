#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

string source_file = "../../images/pos_0.jpg" ;

// detection cercles
int main(void)
 {
 Mat src, src_gray;
 
 /// Read the image
 src = imread( source_file );
 
 if( !src.data )
 { return -1; }
 
 /// Convert it to gray
 cvtColor( src, src_gray, CV_BGR2GRAY );
 
 /// Reduce the noise so we avoid false circle detection
 GaussianBlur( src_gray, src_gray, Size(3, 3), 0, 0 );
 namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
 imshow("Hough Circle Transform Demo", src_gray );waitKey();
 
 vector<Vec3f> circles;
 
 /// Apply the Hough Transform to find the circles
 HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, 2, 100, 14, 0, 8 );
 
 /// Draw the circles detected
 for( size_t i = 0; i < circles.size(); i++ )
 {
 Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
 int radius = cvRound(circles[i][2]);
 // circle center
 circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
 // circle outline
 circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
 }
 
 /// Show your results
 
 imshow( "Hough Circle Transform Demo", src );
 
 waitKey(0);
 return 0;
 }


