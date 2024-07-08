#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
                   CascadeClassifier& nestedCascade,
                   double scale);

int main(int argc, const char** argv)
{
    VideoCapture capture;
    Mat frame;

    CascadeClassifier cascade, nestedCascade;
    double scale = 1.0;

    nestedCascade.load("../../haarcascade_eye_tree_eyeglasses.xml");
    cascade.load("../../haarcascade_frontalcatface.xml");

    capture.open(0);
    if (capture.isOpened())
    {
        cout << "Face Detection Started...." << endl;
        while (1)
        {
            capture >> frame;
            if (frame.empty())
                break;

            Mat frame1 = frame.clone();
            detectAndDraw(frame1, cascade, nestedCascade, scale);

            char c = (char)waitKey(10);
            if (c == 27 || c == 'q' || c == 'Q')
                break;
        }
    }
    else
    {
        cout << "Could not Open Camera" << endl;
    }

    return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
                   CascadeClassifier& nestedCascade,
                   double scale)
{
    Mat gray, smallImg;
    vector<Rect> faces;

    cvtColor(img, gray, COLOR_BGR2GRAY);
    double fx = 1 / scale;
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];
        Point topLeft(cvRound(r.x * scale), cvRound(r.y * scale));
        Point bottomRight(cvRound((r.x + r.width - 1) * scale), cvRound((r.y + r.height - 1) * scale));
        rectangle(img, topLeft, bottomRight, Scalar(255, 0, 0), 3, LINE_8, 0);

        Mat smallImgROI = smallImg(r);
        vector<Rect> nestedObjects;
        nestedCascade.detectMultiScale(smallImgROI, nestedObjects, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        for (size_t j = 0; j < nestedObjects.size(); j++)
        {
            Rect nr = nestedObjects[j];
            Point center(cvRound((r.x + nr.x + nr.width * 0.5) * scale), cvRound((r.y + nr.y + nr.height * 0.5) * scale));
            int radius = cvRound((nr.width + nr.height) * 0.25 * scale);
            circle(img, center, radius, Scalar(255, 0, 0), 3, LINE_8, 0);
        }
    }

    imshow("Face Detection", img);
}
