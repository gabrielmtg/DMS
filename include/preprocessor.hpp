#pragma once
#include <opencv2/opencv.hpp>

struct CropConfig {
    float scale = 2.2f;
    float shift_y = 0.1f;
    int target_size = 160;
};

// Recebe o frame original (BGR) e a caixa do MediaPipe, devolve o tensor RGB 160x160
cv::Mat get_driver_crop(const cv::Mat& frame, const cv::Rect& bbox_mp, const CropConfig& cfg = CropConfig());