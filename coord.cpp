#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

// Global variables
cv::Mat g_frame;
cv::Mat g_thresholded;
cv::Mat g_contourImage;
cv::Mat g_outlinesImage;
int g_thresholdValue = 132;
int g_blurSize = 7;
int g_fgMaskBlurSize = 13;
int g_minContourSize = 50;
int g_maxContourSize = 10000;
int g_centroidID = 1;
int g_mouseX = 0;
int g_mouseY = 0;

// Background Subtraction variables
cv::Ptr<cv::BackgroundSubtractor> g_backgroundSubtractor;

// Log file
std::ofstream g_logFile;

// Function to threshold the image and find contours
void processImage()
{
    cv::Mat blurred;
    cv::GaussianBlur(g_frame, blurred, cv::Size(g_blurSize, g_blurSize), 0);

    cv::threshold(blurred, g_thresholded, g_thresholdValue, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(g_thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    g_contourImage = cv::Mat::zeros(g_frame.size(), CV_8UC3);
    cv::Scalar contourColor = cv::Scalar(0, 165, 255);
    cv::Scalar centroidColor = cv::Scalar(0, 0, 255);

    cv::cvtColor(g_frame, g_outlinesImage, cv::COLOR_GRAY2BGR);

    for (const auto& contour : contours) {
        if ((cv::contourArea(contour) > g_minContourSize) && (cv::contourArea(contour) < g_maxContourSize)) {
            cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
            cv::drawContours(g_contourImage, std::vector<std::vector<cv::Point>>{contour}, -1, color, cv::FILLED);

            cv::drawContours(g_outlinesImage, std::vector<std::vector<cv::Point>>{contour}, -1, contourColor, 2);

            cv::Moments moments = cv::moments(contour);
            cv::Point centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

            cv::circle(g_outlinesImage, centroid, 3, centroidColor, cv::FILLED);

            if (g_logFile.is_open()) {
                double timestamp = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
                int minutes = static_cast<int>(timestamp / 60);
                int seconds = static_cast<int>(timestamp) % 60;
                int milliseconds = static_cast<int>((timestamp - static_cast<int>(timestamp)) * 1000);

                g_logFile << g_centroidID << ", " << centroid.x << ", " << centroid.y << ", "
                          << minutes << ":" << seconds << "." << milliseconds << std::endl;

                std::cout << "Centroid logged: ID=" << g_centroidID << ", X=" << centroid.x << ", Y=" << centroid.y
                          << ", Time=" << minutes << ":" << seconds << "." << milliseconds << std::endl;

                g_centroidID++;
            }
        }
    }

    cv::Mat result;
    cv::cvtColor(g_frame, result, cv::COLOR_GRAY2BGR);
    cv::addWeighted(result, 0.5, g_contourImage, 0.5, 0.0, result);

    cv::imshow("Segmented Image", result);
    cv::imshow("Threshold", g_thresholded);
    cv::imshow("marked", g_outlinesImage);
}

// Callback function for the threshold trackbar
void onThresholdChange(int, void*)
{
    processImage();
}

// Callback function for the blur trackbar
void onBlurSizeChange(int, void*)
{
    if (g_blurSize % 2 == 0)
        ++g_blurSize;

    if (g_blurSize < 3)
        g_blurSize = 3;

    cv::setTrackbarPos("Blur Size", "Segmented Image", g_blurSize);
    processImage();
}

// Callback function for the foreground mask blur trackbar
void onFgMaskBlurSizeChange(int, void*)
{
    if (g_fgMaskBlurSize % 2 == 0)
        ++g_fgMaskBlurSize;

    if (g_fgMaskBlurSize < 3)
        g_fgMaskBlurSize = 3;

    cv::setTrackbarPos("Foreground Mask Blur Size", "Segmented Image", g_fgMaskBlurSize);
    processImage();
}

// Callback function for the minimum contour size trackbar
void onMinContourSizeChange(int, void*)
{
    processImage();
}

void onMaxContourSize(int, void*)
{
    processImage();
}

// Mouse callback function
void onMouse(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_MOUSEMOVE) {
        g_mouseX = x;
        g_mouseY = y;
    }
}

// Callback function for key events
void onKey(int key)
{
    if (key == 's') {
        if (g_logFile.is_open()) {
            g_logFile.close();
            std::cout << "Log file closed." << std::endl;
        } else {
            g_logFile.open("logg.csv", std::ios::app);
            if (g_logFile.is_open()) {
                std::cout << "Log file opened." << std::endl;
                g_logFile << "ID, X, Y, Time" << std::endl;
            } else {
                std::cout << "Error opening log file!" << std::endl;
            }
        }
    }
}

int main()
{
    g_backgroundSubtractor = cv::createBackgroundSubtractorMOG2();
    cv::VideoCapture video("./peak_procedure.mov");

    if (!video.isOpened()) {
        std::cout << "Error opening video file!" << std::endl;
        return -1;
    }

    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Segmented Image", cv::WINDOW_NORMAL);

    cv::createTrackbar("Threshold", "Segmented Image", &g_thresholdValue, 255, onThresholdChange);
    cv::createTrackbar("Blur Size", "Segmented Image", &g_blurSize, 15, onBlurSizeChange);
    cv::createTrackbar("Foreground Mask Blur Size", "Segmented Image", &g_fgMaskBlurSize, 15,
                       onFgMaskBlurSizeChange);
    cv::createTrackbar("Minimum Contour Size", "Segmented Image", &g_minContourSize, 500, onMinContourSizeChange);
    cv::createTrackbar("Maximum Contour Size", "Segmented Image", &g_maxContourSize, 40000, onMaxContourSize);

    g_backgroundSubtractor = cv::createBackgroundSubtractorMOG2();

    cv::setMouseCallback("Segmented Image", onMouse);

    while (true) {
        if (!video.read(g_frame))
            break;

        cv::cvtColor(g_frame, g_frame, cv::COLOR_BGR2GRAY);

        processImage();

        cv::imshow("Video", g_frame);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) // ESC key
            break;

        onKey(key);
    }

    video.release();
    cv::destroyAllWindows();

    return 0;
}
