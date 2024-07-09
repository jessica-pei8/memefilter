//test2
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

enum FilterType { NO_FILTER, FACE_DETECTION, EYE_DETECTION, BOW_AND_TEAR, EYEBROW_LINES };

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
                   CascadeClassifier& nestedCascade,
                   double scale, FilterType filterType, Mat& bowImg, Mat& tearImg);

int main(int argc, const char** argv)
{
    VideoCapture capture;
    Mat frame;

    CascadeClassifier cascade, nestedCascade;
    double scale = 1.0;

    nestedCascade.load("/Users/jessi/OneDrive/Desktop/projects/opencv/sources/data/haarcascades_cuda/haarcascade_eye_tree_eyeglasses.xml");
    cascade.load("/Users/jessi/OneDrive/Desktop/projects/opencv/build/etc/haarcascades/haarcascade_frontalcatface.xml");

    Mat bowImg = imread("/Users/jessi/OneDrive/Desktop/projects/memefilter/src/bow.png", IMREAD_UNCHANGED);
    Mat tearImg = imread("/Users/jessi/OneDrive/Desktop/projects/memefilter/src/tear.png", IMREAD_UNCHANGED);

    if (bowImg.empty() || tearImg.empty()) {
        cerr << "Error: Could not load image files for bow or tear." << endl;
        return -1;
    }

    FilterType filterType = NO_FILTER;

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
            detectAndDraw(frame1, cascade, nestedCascade, scale, filterType, bowImg, tearImg);

            imshow("Meme Filter", frame1);
            char c = (char)waitKey(10);
            if (c == 27 || c == 'q' || c == 'Q')
                break;
            else if (c == '0')
                filterType = NO_FILTER;
            else if (c == '1')
                filterType = FACE_DETECTION;
            else if (c == '2')
                filterType = EYE_DETECTION;
            else if (c == '3')
                filterType = BOW_AND_TEAR;
            else if (c == '4')
                filterType = EYEBROW_LINES;
        }
    }
    else
    {
        cout << "Could not Open Camera" << endl;
    }

    return 0;
}

void drawEyebrowLines(Mat& img, Rect r)
{
    // Diagonal line slanted northwest to southeast for left eyebrow
    line(img, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 255, 0), 2);

    // Draw caret symbol for right eyebrow
    Point p1(r.x + r.width, r.y);
    Point p2(r.x + r.width + 20, r.y - 20);
    Point p3(r.x + r.width + 40, r.y);
    Point p4(r.x + r.width + 60, r.y - 20);
    line(img, p1, p2, Scalar(0, 255, 0), 2);
    line(img, p3, p4, Scalar(0, 255, 0), 2);
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
                   CascadeClassifier& nestedCascade,
                   double scale, FilterType filterType, Mat& bowImg, Mat& tearImg)
{
    if (filterType == NO_FILTER) {
        flip(img, img, 1);
        return;
    }

    Mat gray, smallImg;
    vector<Rect> objects;

    cvtColor(img, gray, COLOR_BGR2GRAY);
    double fx = 1 / scale;
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    if (filterType == FACE_DETECTION) {
        cascade.detectMultiScale(smallImg, objects, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    } else if (filterType == EYE_DETECTION) {
        nestedCascade.detectMultiScale(smallImg, objects, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    }

    for (size_t i = 0; i < objects.size(); i++)
    {
        Rect r = objects[i];
        Point topLeft(cvRound(r.x * scale), cvRound(r.y * scale));
        Point bottomRight(cvRound((r.x + r.width - 1) * scale), cvRound((r.y + r.height - 1) * scale));
        rectangle(img, topLeft, bottomRight, Scalar(255, 0, 0), 3, LINE_8, 0);

        if (filterType == BOW_AND_TEAR) {
            // Apply bow on the top left of the head
            Point bowPosition(topLeft.x, topLeft.y - bowImg.rows);
            if (bowPosition.y > 0 && bowPosition.x > 0 &&
                bowPosition.x + bowImg.cols < img.cols &&
                bowPosition.y + bowImg.rows < img.rows) {
                Mat roi = img(Rect(bowPosition, Size(bowImg.cols, bowImg.rows)));
                bowImg.copyTo(roi, bowImg);
            }

            // Apply tear under the right eye
            Point tearPosition(topLeft.x + r.width, topLeft.y + r.height * 0.5);
            if (tearPosition.x > 0 && tearPosition.y > 0 &&
                tearPosition.x + tearImg.cols < img.cols &&
                tearPosition.y + tearImg.rows < img.rows) {
                Mat roi = img(Rect(tearPosition, Size(tearImg.cols, tearImg.rows)));
                tearImg.copyTo(roi, tearImg);
            }
        }

        if (filterType == EYEBROW_LINES) {
            drawEyebrowLines(img, r);
        }
    }

    flip(img, img, 1);
}
