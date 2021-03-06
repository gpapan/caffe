#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  for (int i = 0; i < this->layer_param_.loss_param().ignore_label_size(); ++i) {
    ignore_label_.insert(this->layer_param_.loss_param().ignore_label(i));
  }
  normalize_ = this->layer_param_.loss_param().normalize();
  // read the weight for each class
  if (this->layer_param_.softmax_loss_param().has_weight_source()) {
    const string& weight_source = this->layer_param_.softmax_loss_param().weight_source();
    LOG(INFO) << "Opening file " << weight_source;
    std::fstream infile(weight_source.c_str(), std::fstream::in);
    CHECK(infile.is_open());
    Dtype tmp_val;
    while (infile >> tmp_val) {
      CHECK_GE(tmp_val, 0) << "Weights cannot be negative";
      loss_weights_.push_back(tmp_val);
    }
    infile.close();
    CHECK_EQ(loss_weights_.size(), prob_.channels());
  } else {
    LOG(INFO) << "Weight_Loss file is not provided. Assign all one to it.";
    loss_weights_.assign(prob_.channels(), 1.0);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  Dtype loss = 0;
  Dtype batch_weight = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (ignore_label_.count(label_value) != 0) {
        continue;
      }
      CHECK_GE(label_value, 0);
      CHECK_LT(label_value, prob_.shape(softmax_axis_));
      batch_weight += loss_weights_[label_value];
      loss -= loss_weights_[label_value] * log(std::max(prob_data[i * dim +
	 label_value * inner_num_ + j], Dtype(FLT_MIN)));
    }
  }
  if (batch_weight == 0) {
    batch_weight = 1;
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / batch_weight;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    // We need to do *weighted* copy, so we defer that for the loop
    //caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    Dtype batch_weight = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (ignore_label_.count(label_value) != 0) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
	  batch_weight += loss_weights_[label_value];
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] =
	      loss_weights_[label_value] * prob_data[i * dim + c * inner_num_ + j];
          }
	  bottom_diff[i * dim + label_value * inner_num_ + j] -=
	    loss_weights_[label_value];
        }
      }
    }
    // Scale gradient
    if (batch_weight == 0) {
      batch_weight = 1;
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / batch_weight, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
