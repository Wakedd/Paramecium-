#include <opencv2/opencv.hpp>

// Global variables
cv::Mat g_frame;
cv::Mat g_fgMask  ;
cv::Mat g_thresholded;
cv::Mat g_contourImage;
cv::Mat g_outlinesimage; 
int g_thresholdValue = 132;
int g_blurSize = 7;
int g_fgMaskBlurSize = 13;
int g_minContourSize = 50;
int g_maxContourSize = 10000;

// Background Subtraction variables
cv::Ptr<cv::BackgroundSubtractor> g_backgroundSubtractor;

// Function to threshold the image and find contours
void processImage()
{
    cv::Mat blurred;
    cv::GaussianBlur(g_frame, blurred, cv::Size(g_blurSize, g_blurSize), 0);

    // Apply background subtraction
    //g_backgroundSubtractor->apply(blurred, g_fgMask);

    // Blur the foreground mask
    //if (g_fgMaskBlurSize > 1)
      //  cv::GaussianBlur(g_fgMask, g_fgMask, cv::Size(g_fgMaskBlurSize, g_fgMaskBlurSize), 0);

    cv::threshold(blurred, g_thresholded, g_thresholdValue, 255, cv::THRESH_BINARY_INV);

    // Find contours in the thresholded image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(g_thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Create a new image for drawing contours
g_contourImage = cv::Mat::zeros(g_frame.size(), CV_8UC3);


cv::Scalar contourColor = cv::Scalar(0, 165, 255); // Orange color (BGR format)
cv::Scalar centroidColor = cv::Scalar(0, 0, 255); // Red color (BGR format)

cv::cvtColor(g_frame, g_outlinesimage, cv::COLOR_GRAY2BGR); 
// Filter contours based on size and draw them on the contour image
for (const auto& contour : contours) {
    if ((cv::contourArea(contour) > g_minContourSize) && (cv::contourArea(contour) < g_maxContourSize)) {
         
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
 


    // Draw the filtered contours on top of the original grayscale image
cv::Mat result;
cv::cvtColor(g_frame, result, cv::COLOR_GRAY2BGR);

// Combine the grayscale image with the contours image
cv::addWeighted(result, 0.5, g_contourImage, 0.5, 0.0, result);

// Display the result images
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
    if (g_blurSize % 2 == 0)  // If even value is selected, increment by 1
        ++g_blurSize;

    if (g_blurSize < 3)  // Minimum blur size should be 3
        g_blurSize = 3;

    cv::setTrackbarPos("Blur Size", "Segmented Image", g_blurSize);  // Update the trackbar position

    processImage();
}

// Callback function for the foreground mask blur trackbar
void onFgMaskBlurSizeChange(int, void*)
{
    if (g_fgMaskBlurSize % 2 == 0)  // If even value is selected, increment by 1
        ++g_fgMaskBlurSize;

    if (g_fgMaskBlurSize < 3)  // Minimum blur size should be 3
        g_fgMaskBlurSize = 3;

    cv::setTrackbarPos("Foreground Mask Blur Size", "Segmented Image", g_fgMaskBlurSize);  // Update the trackbar position

    processImage();
}

// Callback function for the minimum contour size trackbar
void onMinContourSizeChange(int, void*)
{
    processImage();
}
void onmaxContourSize (int, void*)
{
    processImage();
}


int main() {

        g_backgroundSubtractor = cv::createBackgroundSubtractorMOG2();
    // Open the video file
    cv::VideoCapture video("./peak_procedure.mov");

    // Check if the video file was opened successfully
    if (!video.isOpened()) {
        std::cout << "Error opening video file!" << std::endl;
        return -1;
    }

    // Create windows to display the video frames and segmented image
    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Segmented Image", cv::WINDOW_NORMAL);

    // Create a trackbar/slider to control the threshold value
    cv::createTrackbar("Threshold", "Segmented Image", &g_thresholdValue, 255, onThresholdChange);

    // Create a trackbar/slider to control the blur size
    cv::createTrackbar("Blur Size", "Segmented Image", &g_blurSize, 15, onBlurSizeChange);

    // Create a trackbar/slider to control the foreground mask blur size
    cv::createTrackbar("Foreground Mask Blur Size", "Segmented Image", &g_fgMaskBlurSize, 15, onFgMaskBlurSizeChange);

    // Create a trackbar/slider to control the minimum contour size
    cv::createTrackbar("Minimum Contour Size", "Segmented Image", &g_minContourSize, 500, onMinContourSizeChange);


    // Create a trackbar/slider to control the maximum contour size
    cv::createTrackbar("maximum Contour Size", "Segmented Image", &g_maxContourSize, 40000, onmaxContourSize);

    // Initialize the background subtractor
    g_backgroundSubtractor = cv::createBackgroundSubtractorMOG2();

    while (true) {
        // Read a frame from the video file
        if (!video.read(g_frame))
            break;

        // Convert the frame to monochrome (8-bit, single channel)
        cv::cvtColor(g_frame, g_frame, cv::COLOR_BGR2GRAY);

        // Process the image and display the segmented imageA
        processImage();

        // Display the frame in the "Video" window
        cv::imshow("Video", g_frame);

        // Wait for a key press (30ms delay between frames)
        int key = cv::waitKey(30);

        if (key == 27) // 'Esc' key
            break;
    }

    // Release the video file and destroy the windows
    video.release();
    cv::destroyAllWindows();

    return 0;
}