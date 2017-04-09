#include <caffe/caffe.hpp>
#include <caffe/layer.hpp>
#include <caffe/net.hpp>
//#include <caffe/math_functions.hpp>

//to read image file
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <time.h>
#include <assert.h>
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

struct ImageNode{
	cv::Mat* img;
	struct ImageNode* next;
};

class ImageChain{
	public:
		ImageChain(int max_buffer_size);
		int insert_image(const cv::Mat& img);
		cv::Mat* pop_top(int num);
		int remaining_image_num();
	private:
		int buffer_size_;
		ImageNode* head_;
		ImageNode* tail_;
		int max_buffer_size_;
};

ImageChain::ImageChain(int max_buffer_size){
	max_buffer_size_ = max_buffer_size;
	buffer_size_ = 0;
	head_ = new ImageNode;
	head_->img = NULL;
	head_->next = NULL;
	tail_ = head_;
}

int ImageChain::insert_image(const cv::Mat& img){
	if (buffer_size_ == max_buffer_size_)
		return -1;
	cv::Mat* img_ = new cv::Mat;
	*(img_) = img;
	ImageNode* tmp = new ImageNode;
	tmp->next = NULL;
	tmp->img = img_;
	tail_->next = tmp;
	tail_ = tmp;
	buffer_size_++;
	return 0;
};

cv::Mat* ImageChain::pop_top(int num){
	//todo only one
	assert(num == 1); 

	ImageNode* tmp;
	tmp = head_->next;
	if (head_->next != NULL){
		head_->next = head_->next->next;
		buffer_size_--;
	}
	if (tmp == NULL)
		return NULL;
	return tmp->img;
}

int ImageChain::remaining_image_num(){
	return buffer_size_;
}

const int TEST_CASE = 10000;
class ConvTester {
 public:
  ConvTester(const string& model_file,
             const string& trained_file);

  void Test(const cv::Mat& img);

 private:
  unsigned long convTest(const cv::Mat& img, const int runTimes, const int batchSize);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};

ConvTester::ConvTester(const string& model_file,
                       const string& trained_file) {
  Caffe::set_mode(Caffe::GPU);
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  std::cout<<"good reset"<<std::endl;
  net_->CopyTrainedLayersFrom(trained_file);
  std::cout<<"good copy"<<std::endl;

  Blob<float>* input_layer = net_->input_blobs()[0];
  std::cout<<"good blob"<<std::endl;
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  std::cout<<"good channel"<<std::endl;
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();
}


void ConvTester::Test(const cv::Mat& img) {

  int runTimes = TEST_CASE;
  unsigned long t;

  Caffe::set_mode(Caffe::GPU);
  t = convTest(img, runTimes, 1);
  std::cout<<"Time of convolution forward ("<<TEST_CASE<<" times on GPU):" << t<<"ms"<< std::endl;

  Caffe::set_mode(Caffe::CPU);
  t = convTest(img, runTimes, 1);
  std::cout<<"Time of convolution forward ("<<TEST_CASE<<" times on CPU):" << t<<"ms"<< std::endl;

  Caffe::set_mode(Caffe::GPU);
  t = convTest(img, runTimes, 1);
  std::cout<<"Time of convolution forward ("<<TEST_CASE<<" times on GPU):" << t<<"ms"<< std::endl;

  Caffe::set_mode(Caffe::CPU);
  t = convTest(img, runTimes, 1);
  std::cout<<"Time of convolution forward ("<<TEST_CASE<<" times on CPU):" << t<<"ms"<< std::endl;

  Caffe::set_mode(Caffe::CPU);
  t = convTest(img, runTimes, 1);
  std::cout<<"Time of convolution forward ("<<TEST_CASE<<" times on GPU):" << t<<"ms"<< std::endl;

  Caffe::set_mode(Caffe::GPU);
  t = convTest(img, runTimes, 1);
  std::cout<<"Time of convolution forward ("<<TEST_CASE<<" times on CPU):" << t<<"ms"<< std::endl;

}

unsigned long ConvTester::convTest(const cv::Mat& img, const int runTimes, const int batchSize) {
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  Preprocess(img, &input_channels);

  timeval start_t, end_t;
  unsigned long t;

  //const string conv_name = "conv1";
  //const shared_ptr<Blob<float>>& conv_blob = net_->blob_by_name(conv_name);
  //const shared_ptr< Layer<float>>& conv_layer = net_->layer_by_name(conv_name);
  //const float* conv_out;
  //const vector< shared_ptr< Blob< float > > > & conv_layer_out = conv_layer->blobs();
  //const vector<int>& conv_shape = conv_blob->shape();
  
  //conv_out = conv_blob->cpu_data();
//  std::cout<<"Layer Size:" << conv_layer_out.size() <<std::endl;

  /*
  const vector<int> & conv_layer_shape = conv_layer_out[0]->shape();
  std::cout << conv_layer_shape.size() << std::endl;
  for (int i = 0;i<conv_layer_shape.size();i++)
	  std::cout << conv_layer_shape[i] << std::endl;
  int p=0;
  for (int bs = 0; bs<conv_layer_shape[0] ;bs++){
	  std::cout<<bs<<std::endl;
	  for (int ks = 0; ks<conv_layer_shape[1];ks++){
		  for (int h = 0; h<conv_layer_shape[2]; h++){
			  for (int w = 0; w<conv_layer_shape[3];w++)
				  std::cout << std::setprecision(2)<<conv_layer_out[0]->cpu_data()[p++] <<' ';
			  std::cout<<std::endl;
		  }
	  }
  }
*/

//  std::cout << conv_shape.size() << std::endl;
//  for (int i = 0;i<conv_shape.size();i++)
//	  std::cout << conv_shape[i] << std::endl;

  /*
  int p=0;
  for (int bs = 0; bs<conv_shape[0] ;bs++)
	  for (int ks = 0; ks<conv_shape[1];ks++){
		  std::cout<<ks<<std::endl;
		  for (int h = 0; h<conv_shape[2]; h++){
			  for (int w = 0; w<conv_shape[3];w++)
				  std::cout << conv_out[p++] <<' ';
			  std::cout<<std::endl;
		  }
	  }
 */
  gettimeofday(&start_t, NULL);
  for (int seq = 0; seq < runTimes; ++seq)
    net_->ForwardFromTo(0,1);
  gettimeofday(&end_t, NULL);

  t = 1000*(end_t.tv_sec - start_t.tv_sec) + (end_t.tv_usec - start_t.tv_usec)/1000;

  return t;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void ConvTester::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void ConvTester::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  ConvTester classifier(model_file, trained_file);
  string file = argv[3];

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  
  ImageChain* chain = new ImageChain(10);
  chain->insert_image(img);
  chain->insert_image(img);
  chain->insert_image(img);
  chain->insert_image(img);
  chain->insert_image(img);
  chain->insert_image(img);
  chain->insert_image(img);
  chain->insert_image(img);
  chain->insert_image(img);
  cv::Mat* img4test = (chain->pop_top(1));
  std::cout <<img4test<<std::endl;
  classifier.Test(*img4test);
  img4test = (chain->pop_top(1));
  std::cout <<img4test<<std::endl;
  classifier.Test(*img4test);
  
  img4test = (chain->pop_top(1));
  std::cout <<img4test<<std::endl;
  classifier.Test(*img4test);
  
  img4test = (chain->pop_top(1));
  std::cout <<img4test<<std::endl;
  classifier.Test(*img4test);
  
  img4test = (chain->pop_top(1));
  std::cout <<img4test<<std::endl;
  classifier.Test(*img4test);
  
  img4test = (chain->pop_top(1));
  std::cout <<img4test<<std::endl;
  classifier.Test(*img4test);
  
  img4test = (chain->pop_top(1));
  std::cout <<img4test<<std::endl;
  classifier.Test(*img4test);
  
  img4test = (chain->pop_top(1));
  std::cout <<img4test<<std::endl;
  classifier.Test(*img4test);

  img4test = (chain->pop_top(1));
  std::cout <<img4test<<std::endl;
  classifier.Test(*img4test);
}
