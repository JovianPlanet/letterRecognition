#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cvblobs/BlobResult.h>

#include <iostream>
#include <string>
#include <cstdio>
#include <vector>

using namespace std;
using namespace cv;



CBlobResult blobs; 


std::string folderpath="/media/hoh-6/Datos/David/U/ai/database/DataBase_Letras/";
std::stringstream path;
Mat frame;

FILE *fp;

Mat momentosHu(Mat imagenGris){
	
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Point> contour;

	findContours( imagenGris, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	
	/// Get the moments
	vector<Moments> mu(contours.size() );
	int countMayor=0;
	int mayor;
	Mat huM;
	Mat hu;
	

	for( int i = 0; i < contours.size() ; i++ )
	{ 
		approxPolyDP(cv::Mat(contours[i]), contour, 0.025, true);
		cv::Rect rect = cv::boundingRect(contour);
		mu[i] = moments( contours[i], false ); 
		
		HuMoments(mu[i],hu);

		/*const cv::Point* p = &contour[0];
        int n = (int)contour.size();
        polylines(imagen, &p, &n, 1, true, cv::Scalar(0, 255, 0), 1, CV_AA);*/
 
        cv::Point pt1, pt2;
        pt1.x = rect.x;
        pt1.y = rect.y;
        pt2.x = rect.x + rect.width;
        pt2.x = rect.y + rect.height;
        cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 1);
		
		if(countMayor<contours[i].size()){
			mayor=i;
			countMayor=contours[i].size();
		}
	}

	mu[mayor] = moments( contours[mayor], true ); 
		
	HuMoments(mu[mayor],huM);

	cout<<huM<<endl;

	/*const cv::Point* p = &contour[mayor];
        int n = (int)contour.size();
        polylines(imagen, &p, &n, 1, true, cv::Scalar(0, 255, 0), 1, CV_AA);

	cv::FileStorage fs( "hu Data s", cv::FileStorage::WRITE );
    fs << "hu Moments" << huM;*/

	return huM;
		

}

int main(void)
{
	
	//if((fp=fopen("datosEntrenamientoLetras.data", "w+"))==NULL) { 
	if((fp=fopen("miletra.data", "w+"))==NULL) {
		printf("Cannot open file.\n"); 
		exit(1); 
	} 
	namedWindow("window",1);

	for(int i=1; i<6;i++){
		for(int j=1; j<21;j++){
			std::stringstream path;
			path<<folderpath<<char(i+64)<<"_4_("<<j<<").jpg"; //i+1

			//load images
			//frame= imread(path.str().c_str());
			frame= imread("b.jpg");
			Mat frameGris;
			cvtColor(frame,frameGris,CV_BGR2GRAY);
			GaussianBlur(frameGris, frameGris, Size(3, 3), sqrt(2), sqrt(2), 0);
			threshold(frameGris,frameGris,128,255,CV_THRESH_OTSU);
			frameGris= frameGris(Range(5,frameGris.rows-5), Range(5,frameGris.cols-5));			
						
			IplImage IplimagenUmbral = frameGris;
			IplImage Iplimagen = frame;

			blobs = CBlobResult( &IplimagenUmbral,NULL,255);

			CBlob *currentBlob;	

			blobs.Filter( blobs, B_INCLUDE, CBlobGetArea(), B_GREATER,  200);

			int numBlobs=blobs.GetNumBlobs();

			Mat frameEsc;

			int cont=0;

			if(numBlobs==1){
				for (int ii = 0; ii < blobs.GetNumBlobs(); ii++ )
				{
					currentBlob = blobs.GetBlob(ii);
					currentBlob->FillBlob(&Iplimagen, CV_RGB( 0, 0, 255));
					
					Mat feature=momentosHu(frameGris);
					fprintf(fp,"%c",char(i+64)); 
					for(int jj=0; jj<feature.rows; jj++){
							fprintf(fp,", %f",feature.at<double>(jj,0)); 
					}
					fprintf(fp,"\n");
				/*imshow("window",frame);    
				waitKey(1000);*/
				}
			}


			/*imshow("window",frameGris);    
			waitKey(1000);*/
		}
	}
	
    return 0;
}