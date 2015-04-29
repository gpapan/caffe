#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  fg_ratio_ = this->layer_param_.mask_param().fg_ratio();
  fg_value_ = this->layer_param_.mask_param().fg_value();
  bg_value_ = this->layer_param_.mask_param().bg_value();
  CHECK(fg_ratio_ >= 0 && fg_ratio_ <= 1);
  if (fg_ratio_ > 0.5) {
    fg_ratio_ = 1 - fg_ratio_;
    const Dtype swap = fg_value_;
    fg_value_ = bg_value_;
    bg_value_ = swap;
  }
  rect_height_ = this->layer_param_.mask_param().height();
  rect_width_ = this->layer_param_.mask_param().width();
  CHECK_GT(rect_height_, 0);
  CHECK_GT(rect_width_, 0);
}

template <typename Dtype>
void MaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Set up the cache for random number generation
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  caffe_set(top[0]->count(), bg_value_, top[0]->mutable_cpu_data());
  int area = height_ * width_;
  for (int n = 0; n < num_; ++n) {
    Dtype* top_data = top[0]->mutable_cpu_data(n);
    int num_fg = 0;
    while (static_cast<float>(num_fg) / area < fg_ratio_) {
      const int h_beg = caffe_rng_rand() % (height_ - rect_height_);
      const int w_beg = caffe_rng_rand() % (width_ - rect_width_);
      for (int h = h_beg; h < h_beg + rect_height_; ++h) {
	for (int w = w_beg; w < w_beg + rect_width_; ++w) {
	  if (top_data[h * width_ + w] == bg_value_) {
	    num_fg++;
	    top_data[h * width_ + w] = fg_value_;
	  }
	}
      }
    }
    for (int c = 1; c < channels_; ++c) {
      caffe_copy(area, top_data, top_data + c * area);
    }
  }
}

INSTANTIATE_CLASS(MaskLayer);
REGISTER_LAYER_CLASS(Mask);

}  // namespace caffe
