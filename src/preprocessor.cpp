#include "preprocessor.hpp"
#include <algorithm>
#include <cmath>

cv::Mat get_driver_crop(const cv::Mat& frame, const cv::Rect& bbox_mp, const CropConfig& cfg) {
    float s = std::max(bbox_mp.width, bbox_mp.height);
    float new_s = s * cfg.scale;
    float cx = bbox_mp.x + (bbox_mp.width / 2.0f);
    float cy = bbox_mp.y + (bbox_mp.height / 2.0f) + (cfg.shift_y * s);
    
    int x1 = std::round(cx - (new_s / 2.0f));
    int y1 = std::round(cy - (new_s / 2.0f));
    int x2 = x1 + std::round(new_s);
    int y2 = y1 + std::round(new_s);
    
    int pad_left = std::max(0, -x1);
    int pad_top = std::max(0, -y1);
    int pad_right = std::max(0, x2 - frame.cols);
    int pad_bottom = std::max(0, y2 - frame.rows);
    
    cv::Rect safe_roi(
        std::max(0, x1), std::max(0, y1),
        std::min(x2, frame.cols) - std::max(0, x1),
        std::min(y2, frame.rows) - std::max(0, y1)
    );
    
    cv::Mat cropped = frame(safe_roi);
    cv::Mat padded;
    
    if (pad_left > 0 || pad_top > 0 || pad_right > 0 || pad_bottom > 0) {
        cv::copyMakeBorder(cropped, padded, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else {
        padded = cropped;
    }
    
    cv::Mat resized, final_rgb;
    cv::resize(padded, resized, cv::Size(cfg.target_size, cfg.target_size), 0, 0, cv::INTER_AREA);
    cv::cvtColor(resized, final_rgb, cv::COLOR_BGR2RGB);
    
    return final_rgb;
}