#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <start_address>" << std::endl;
        return -1;
    }

    // Convert the command-line argument to a memory address
    void* startAddress;
    sscanf(argv[1], "%p", &startAddress);

    // Create a background subtractor
    Ptr<BackgroundSubtractorMOG2> bgSubtractor = createBackgroundSubtractorMOG2();

    // Open the video file
    VideoCapture cap("Hackathon.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    // Get video properties
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));

    // Define the codec and create a VideoWriter object
    VideoWriter videoWriter("output_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));

    while (true) {
        Mat frame;
        cap >> frame;

        // Break the loop if the video ends
        if (frame.empty())
            break;

        // Apply background subtraction
        Mat fgMask;  // Foreground mask
        bgSubtractor->apply(frame, fgMask);

        // Find contours in the foreground mask
        std::vector<std::vector<Point>> contours;
        findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Draw bounding rectangles around moving objects
        for (const auto& contour : contours) {
            if (contourArea(contour) > 100) {  // Adjust the threshold as needed
                Rect boundingBox = boundingRect(contour);
                rectangle(frame, boundingBox, Scalar(0, 255, 0), 2);
            }
        }

        // Save the frame to the video
        videoWriter.write(frame);

        // Display the processed frame
        imshow("Motion Detection", frame);

        // Press 'Esc' to exit the loop
        if (waitKey(30) == 27)
            break;
    }

    // Release the VideoWriter
    videoWriter.release();

    return 0;
}
