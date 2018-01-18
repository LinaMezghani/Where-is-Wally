#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(int begin_x, int begin_y, int window_size_x, int window_size_y);
void applyClassifier(Mat frame, Point topLeft) ;

/** Global variables */
//Cascade Classifier
String face_cascade_name = "../../cascade-4000.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
int window_size_x = 160 ;   //taille des frames utilisées pour la base de donnée des négatifs
int window_size_y = 200 ;

//Image source (enigme charlie)
string source_file = "../../Test_Images/test2.jpg" ;
Mat source = imread(source_file) ;
//Résultat (engime charlie résolue)
Mat result = Mat::zeros(source.rows,source.cols,CV_8UC3) ;

//Indique si on utilise le détecteur de rouge
bool red_filter = false ;

String window_name = "Trouver Charlie";


void modify_mask(Mat source, Mat mask, Rect rect){
    //IN_PLACE : Pour mettre en lumière le charlie trouvé
    Mat roi = source(rect).clone();
    roi.copyTo(mask(rect)) ;
}

cv::Vec3f BGRtoHSV(const cv::Vec3b& bgr)
{
    //convertit BGR en HSV
    cv::Mat3f bgrMat(static_cast<cv::Vec3f>(bgr));

    bgrMat *= 1./255.;

    cv::Mat3f hsvMat;
    cv::cvtColor(bgrMat, hsvMat, CV_BGR2HSV);

    cv::Vec3f hsv = hsvMat(0,0);

    return hsv;
}

bool isRed(Vec3f hsv){
    //renvoie true si le code HSV code une teinte de rouge
    float h =  hsv[0] ;
    float s =  hsv[1] ;
    float v =  hsv[2] ;
    return s>=0.5&&v>=0.6&&(h>=320||h<=15) ;
}

bool contains_red(Mat image){
    //retourne true si l'image contient suffisamment de rouge

    Mat img = image.clone() ;

    GaussianBlur(img,img, Size(3, 3), 0, 0 ); //filtre gaussien pour lisser les couleurs

    //conversion des données pour appliquer k_mean
    Mat samples(img.rows * img.cols, 3, CV_32F);
    for( int y = 0; y < img.rows; y++ )
        for( int x = 0; x < img.cols; x++ )
            for( int z = 0; z < 3; z++)
                samples.at<float>(y + x*img.rows, z) = img.at<Vec3b>(y,x)[z];

    int clusterCount = 20; //nb de couleurs majoritaires que l'on veut
    Mat labels;
    int attempts = 5;
    Mat centers;

    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

    for(int i =0 ; i<centers.rows ; i++){ //pour chaque couleur dominante trouvée
        Vec3b bgr(centers.at<float>(i,0),centers.at<float>(i,1),centers.at<float>(i,2)) ;
        if (isRed(BGRtoHSV(bgr))) return true ; //si c'est du rouge, on renvoie true
    }
    return false ; //sinon, c'est qu'il n'y a pas suffisamment de rouge
}


//detectAndDisplay : applique le classifier sur chaque rectangle (de taille window_size_x,window_size_y) de l'image source
void detectAndDisplay(int begin_x, int begin_y, int window_size_x, int window_size_y){

    for(int i = begin_x ; i<source.cols-window_size_x ; i+= window_size_x){
        for(int j = begin_y ; j<source.rows-window_size_y ; j+= window_size_y){

            Point topLeft = Point(i,j) ;
            Mat frame  = source(Rect(i,j,window_size_x,window_size_y)) ;

            //Load the cascades
            if(!face_cascade.load(face_cascade_name)){
                printf("--(!)Error loading face cascade\n");
                return;
            }

            if(frame.empty())printf(" --(!) No captured frame -- Break!");

            //Apply the classifier to the frame
            applyClassifier(frame, topLeft);
        }

    }
}

void applyClassifier(Mat frame, Point topLeft)
{
    vector<Rect> faces;
    Mat frame_gray;

    //la cascade s'applique sur des images NB
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist(frame_gray, frame_gray);

    //Detect charlie
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(8, 10));

    for( size_t i = 0; i < faces.size(); i++ ) //pour tous les positifs détectés
    {
        Point topLeft_face(topLeft.x+ faces[i].x,topLeft.y+faces[i].y);
        Mat positive = frame(faces[i]) ;

        if (!red_filter || contains_red(positive)){ //application du détecteur de rouge
            modify_mask(source, result, Rect(topLeft_face.x,topLeft_face.y,faces[i].width,faces[i].height)) ; //on illumine le positif détecté dans result
        }
    }
}

//Main function

int main( void )
{
    addWeighted(source, 0.25, result, 0.75, 0,result) ;

    //On teste 4 fois la cascade au cas où le découpage de l'image source couperait charlie
    detectAndDisplay(0,0,window_size_x,window_size_y);
    detectAndDisplay(0,window_size_y/2,window_size_x,window_size_y);
    detectAndDisplay(window_size_x/2,0,window_size_x,window_size_y);
    detectAndDisplay(window_size_x/2,window_size_y/2,window_size_x,window_size_y);

    resize(result,result, Size(result.cols/3,result.rows/3)) ;

    imshow( window_name,result );waitKey();

    return 0;
}

