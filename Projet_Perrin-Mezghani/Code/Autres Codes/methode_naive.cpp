#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

vector<Point> charlie_naif(Mat source, Mat charlie, Mat alpha){
    //Méthode naive
    
    Mat result;
    float seuil = 0.805;
    cout<<"seuil : "<<seuil ;
    
    matchTemplate(source,charlie,result,TM_CCORR_NORMED,alpha); //renvoie la matrice de corrélation
    
    vector<Point> topLeft ;
    
    for (int i = 0 ; i<result.rows ; i++){
        for (int j =0; j<result.cols ; j++){
            if(result.at<float>(i,j) > seuil){
                topLeft.push_back(Point(j,i)) ;
            }
        }
    }
    return topLeft ;
}

void modify_mask(Mat source, Mat mask, Point topLeft, int m, int n){
    //IN_PLACE : Pour mettre en lumière le charlie trouvé
    Mat roi = source(Rect(topLeft.x,topLeft.y,n,m)).clone();
    roi.copyTo(mask(Rect(topLeft.x,topLeft.y,n,m))) ;
}

Mat mask_v2(Mat source, vector<Point> topLeft, int m, int n){
    //éclaire les points de coin supérieur gauche dans topLeft
    Mat mask = Mat::zeros(source.rows,source.cols,CV_8UC3) ;
    addWeighted(source, 0.25, mask, 0.75, 0,mask) ;
    for(int i = 0 ; i<topLeft.size();i++)
    {
        if(topLeft.at(i).x+n<source.cols && topLeft.at(i).y + m<source.rows)
            modify_mask(source,mask,topLeft.at(i),m,n) ;
    }
    return mask ;
}

int main()
{
    Mat source=imread("../../images/test7.jpg");
    Mat modele_charlie=imread("../../images/modele_charlie.png",IMREAD_UNCHANGED);
    
    //Pour différencier les canaux RGB et le canal alpha
    vector<Mat> mv(4);
    split(modele_charlie,mv) ;
    Mat modele_charlie_rgb ;
    vector<Mat> rgb ;
    rgb.push_back(mv[0]) ; rgb.push_back(mv[1]) ; rgb.push_back(mv[2]) ;
    merge(rgb,modele_charlie_rgb) ;
    Mat alpha;
    alpha=mv[3] ;
    cvtColor(alpha,alpha,CV_GRAY2RGB);
    
    //Affiche le résultat
    vector<Point> topLeft = charlie_naif(source,modele_charlie_rgb,alpha) ;
    Mat result = mask_v2(source,topLeft,modele_charlie_rgb.rows,modele_charlie_rgb.cols) ;
    resize(result,result,Size(result.cols/3,result.rows/3)) ;
    imshow("Methode Naive",result);waitKey();
    
	return 0;
}
