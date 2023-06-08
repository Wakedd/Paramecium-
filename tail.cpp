#include <opencv2/opencv.hpp>
#include <chrono>

// Global variables
cv::Mat g_frame;
cv::Mat g_fgMask;
cv::Mat g_thresholded;
cv::Mat g_contourImage;
int g_thresholdValue = 128;
int g_blurSize = 3;
int g_fgMaskBlurSize = 3;
int g_minContourSize = 100;

// Background Subtraction variables
cv::Ptr<cv::BackgroundSubtractor> g_backgroundSubtractor;

// Tracking variables
std::vector<cv::Point> g_previousPositions;
std::chrono::steady_clock::time_point g_lastMoveTime;

// Function to threshold the image and find contours
void processImage()
{
    cv::Mat blurred;
    cv::GaussianBlur(g_frame, blurred, cv::Size(g_blurSize, g_blurSize), 0);

    // Apply background subtraction
    g_backgroundSubtractor->apply(blurred, g_fgMask);

    // Blur the foreground mask
    if (g_fgMaskBlurSize > 1)
        cv::GaussianBlur(g_fgMask, g_fgMask, cv::Size(g_fgMaskBlurSize, g_fgMaskBlurSize), 0);

    cv::threshold(g_fgMask, g_thresholded, g_thresholdValue, 255, cv::THRESH_BINARY);

    // Perform morphological closing operation to merge nearby regions
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::morphologyEx(g_thresholded, g_thresholded, cv::MORPH_CLOSE, kernel);

    // Find contours in the thresholded image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(g_thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a new image for drawing contours
    g_contourImage = cv::Mat::zeros(g_frame.size(), CV_8UC3);

    // Filter contours based on size and draw them on the contour image
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) > g_minContourSize) {
            cv::Scalar color = cv::Scalar(0, 165, 255); // Orange color
            cv::drawContours(g_contourImage, std::vector<std::vector<cv::Point>>{contour}, -1, color, 2, cv::LINE_AA); // Draw contour with thickness 2

            // Calculate centroid of the contour
            cv::Moments moments = cv::moments(contour);
            double cx = moments.m10 / moments.m00;
            double cy = moments.m01 / moments.m00;

            // Mark centroid with a red dot
            cv::circle(g_contourImage, cv::Point(cx, cy), 3, cv::Scalar(0, 0, 255), -1);

            // Track the red dot
            auto currentTime = std::chrono::steady_clock::now();
            double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - g_lastMoveTime).count() / 1000.0;
            if (elapsedTime <= 4.0) {
                g_previousPositions.push_back(cv::Point(cx, cy));
            } else {
                g_previousPositions.clear();
                g_lastMoveTime = currentTime;
            }

            // Draw the previous positions as a green tail
            for (size_t i = 0; i < g_previousPositions.size(); i++) {
                cv::Scalar tailColor = cv::Scalar(0, 255, 0); // Green color
                cv::circle(g_contourImage, g_previousPositions[i], 2, tailColor, -1);
            }
        }
    }

    // Draw the filtered contours on top of the original grayscale image
    cv::Mat result;
    cv::cvtColor(g_frame, result, cv::COLOR_GRAY2BGR);
    cv::addWeighted(result, 1.0, g_contourImage, 0.5, 0.0, result);

    // Display the result image
    cv::imshow("Segmented Image", result);
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

int main() {
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

    // Initialize the background subtractor
    g_backgroundSubtractor = cv::createBackgroundSubtractorMOG2();

    while (true) {
        // Read a frame from the video file
        if (!video.read(g_frame))
            break;

        // Convert the frame to monochrome (8-bit, single channel)
        cv::cvtColor(g_frame, g_frame, cv::COLOR_BGR2GRAY);

        // Process the image and display the segmented image
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
