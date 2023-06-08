#include <iostream>
#include <opencv2/opencv.hpp>

// Global variables
cv::Mat g_frame;
cv::Mat g_fgMask;
cv::Mat g_thresholded;
cv::Mat g_contourImage;
cv::Mat g_outlinesimage;
int g_thresholdValue = 132;
int g_blurSize = 7;
int g_fgMaskBlurSize = 13;
int g_minContourSize = 50;
int g_maxContourSize = 10000;
int g_aspectRatioThreshold = 80; // Added aspect ratio threshold variable


double g_aspectRatioThresholdDouble = (double)g_aspectRatioThreshold / 100.0;
// Background Subtraction variables
cv::Ptr<cv::BackgroundSubtractor> g_backgroundSubtractor;

// Function to calculate the aspect ratio of a contour
double calculateAspectRatio(const std::vector<cv::Point>& contour);

// Calculate aspect ratio of a contour
double calculateAspectRatio(const std::vector<cv::Point>& contour)
{
    cv::RotatedRect boundingBox = cv::minAreaRect(contour);
    cv::Size2f size = boundingBox.size;
    double aspectRatio = size.width / size.height;
    return aspectRatio;
}

// Function to threshold the image and find contours
void processImage()
{
    cv::Mat blurred;
    cv::GaussianBlur(g_frame, blurred, cv::Size(g_blurSize, g_blurSize), 0);
    cv::threshold(blurred, g_thresholded, g_thresholdValue, 255, cv::THRESH_BINARY_INV);

    // Find contours in the thresholded image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(g_thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    g_contourImage = cv::Mat::zeros(g_frame.size(), CV_8UC3);
    cv::Scalar contourColor = cv::Scalar(0, 165, 255); // Orange color (BGR format)
    cv::Scalar centroidColor = cv::Scalar(0, 0, 255); // Red color (BGR format)

    cv::cvtColor(g_frame, g_outlinesimage, cv::COLOR_GRAY2BGR);

    // Filter contours based on size and aspect ratio, and draw them on the contour image
    for (const auto& contour : contours)
    {
        double aspectRatio = calculateAspectRatio(contour);
        if ((cv::contourArea(contour) > g_minContourSize) &&
            (cv::contourArea(contour) < g_maxContourSize) &&
            (aspectRatio < g_aspectRatioThreshold) ) // Check aspect ratio
        {
            cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
            cv::drawContours(g_contourImage, std::vector<std::vector<cv::Point>>{contour}, -1, color, cv::FILLED);

            // Draw orange line along the contour on g_outlinesimage
            cv::drawContours(g_outlinesimage, std::vector<std::vector<cv::Point>>{contour}, -1, contourColor, 2);

            // Find centroid of the contour
            cv::Moments moments = cv::moments(contour);
            cv::Point centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

            // Draw red dot as the centroid on g_outlinesimage
            cv::circle(g_outlinesimage, centroid, 3, centroidColor, cv::FILLED);
        }
    }

    cv::Mat result;
    cv::cvtColor(g_frame, result, cv::COLOR_GRAY2BGR);
    cv::addWeighted(result, 0.5, g_contourImage, 0.5, 0.0, result);

    cv::imshow("Segmented Image", result);
    cv::imshow("Threshold", g_thresholded);
    cv::imshow("marked", g_outlinesimage);
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

// Callback function for the maximum contour size trackbar
void onMaxContourSizeChange(int, void*)
{
    processImage();
}

// Callback function for the aspect ratio threshold trackbar
void onAspectRatioThresholdChange(int, void*)
{
    processImage();
}

int main()
{
    g_backgroundSubtractor = cv::createBackgroundSubtractorMOG2();
    cv::VideoCapture video("./peak_procedure.mov");
    if (!video.isOpened())
    {
        std::cout << "Error opening video file!" << std::endl;
        return -1;
    }

    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Segmented Image", cv::WINDOW_NORMAL);

    cv::createTrackbar("Threshold", "Segmented Image", &g_thresholdValue, 255, onThresholdChange);
    cv::createTrackbar("Blur Size", "Segmented Image", &g_blurSize, 15, onBlurSizeChange);
    cv::createTrackbar("Foreground Mask Blur Size", "Segmented Image", &g_fgMaskBlurSize, 15, onFgMaskBlurSizeChange);
    cv::createTrackbar("Minimum Contour Size", "Segmented Image", &g_minContourSize, 500, onMinContourSizeChange);
    cv::createTrackbar("Maximum Contour Size", "Segmented Image", &g_maxContourSize, 40000, onMaxContourSizeChange);
    cv::createTrackbar("Aspect Ratio Threshold", "Segmented Image", &g_aspectRatioThreshold, 120, onAspectRatioThresholdChange);

    while (true)
    {
        if (!video.read(g_frame))
            break;

        cv::cvtColor(g_frame, g_frame, cv::COLOR_BGR2GRAY);
        processImage();

        cv::imshow("Video", g_frame);
        int key = cv::waitKey(30);
        if (key == 27)
            break;
    }

    video.release();
    cv::destroyAllWindows();
    return 0;
}
