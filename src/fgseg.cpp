/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.cpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "fgseg.hpp"
using namespace std;
using namespace fgseg;

//default constructor
bgs::bgs(double tau, bool rgb, int threshold, double alpha4)
{
	_rgb=rgb;

	_tau=tau;

	//_alpha=alpha;

	_threshold = threshold;

	//_shdalpha = shdalpha;

	//_shdbetha = shdbetha;

	//_taus = taus;

	//_tauh = tauh;

	_alpha4 = alpha4;
}

//default destructor
bgs::~bgs(void)
{
}

//method to initialize bkg (first frame - hot start)
void bgs::init_bkg(cv::Mat Frame)
{

	if (!_rgb){
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); // to work with gray even if input is color

		_bkg = Mat::zeros(Size(Frame.cols,Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
		//ADD YOUR CODE HERE
		//...
		_bkg = Frame;
		//...
	}
	else{
		_bkg = Frame;
	}

	//size of frame
	int width = Frame.cols;
	int height = Frame.rows;
	//blank image to keep counter per pixel for foreground
	if(!_rgb){
		_mean = _bkg;
		_mean.convertTo(_mean, CV_32FC1);
		_std = Mat(cv::Size(width, height), CV_32FC1, cv::Scalar(20));
	}
	else{
		_mean = _bkg;
		_mean.convertTo(_mean, CV_32FC3);
		_std = Mat(cv::Size(width, height), CV_32FC3, cv::Scalar(5, 5, 5));
	}

	//blank image to keep counter per pixel for foreground
	if(_rgb){
		_counter = Mat(cv::Size(width, height), CV_8UC3, cv::Scalar(0, 0, 0));
		_increment = Mat(cv::Size(width, height), CV_8UC3, cv::Scalar(1, 1, 1));
	}
	else{
		_counter = Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(0));
		_increment = Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(1));
	}

}

//method to perform BackGroundSubtraction

void bgs::bkgSubtraction(cv::Mat Frame)
{

	if (!_rgb){
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); // to work with gray even if input is color
		Frame.copyTo(_frame);

		Mat low_bkg = _mean - 4*_std;
		Mat high_bkg = _mean + 4*_std;

		low_bkg.convertTo(low_bkg, CV_8U);
		high_bkg.convertTo(high_bkg, CV_8U);
		//black inside the interval(background) and white outside(foreground)
		_bgsmask = (_frame < low_bkg) | (_frame > high_bkg);
		//_bgsmask = _frame;
		_diff = abs(Frame - _bkg);

	}
	else{
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); // to work with gray even if input is color
		Frame.copyTo(_frame);

	    }
	update_bkg(Frame);

	//cvtColor(_bgsmask, _bgsmask, COLOR_BGR2GRAY);

}
//method to detect and remove shadows in the BGS mask to create FG mask
void bgs::removeShadows()
{/*
	cv::Mat bkg_hsv;
	cv::Mat frame_hsv;
    // Convert the images to float type

	cvtColor(_bkg, bkg_hsv, CV_BGR2HSV);
	cvtColor(_frame, frame_hsv, CV_BGR2HSV);
	cv::Mat channels_bkg[3];
	cv::Mat channels_frame[3];
	split(bkg_hsv, channels_bkg);
	split(frame_hsv, channels_frame);

	//get the hue and convert to float
	//first term
	cv::Mat i_frame = channels_frame[2], i_bkg = channels_bkg[2];
	i_frame.convertTo(i_frame, CV_32F);
	i_bkg.convertTo(i_bkg, CV_32F);

	cv::Mat aux;
	aux = i_frame / i_bkg;
	cv::Mat aux1 = (aux >= _shdalpha) & (aux <= _shdbetha);

	//second term
	cv::Mat aux2 = abs(channels_frame[1] - channels_bkg[1]) <= _taus;

	//D
	//third term
	cv::Mat aux3 = 2*abs(channels_frame[0] - channels_bkg[0]);
	cv::Mat aux4 = 360 - aux3;
	aux3 = min(aux3, aux4) <= _tauh;
	//cout<<aux3;
	_shadowmask = aux1 & aux2 & aux3;
	cv::imshow("shadow", aux3);

//	 init Shadow Mask (currently Shadow Detection not implemented)
//	_bgsmask.copyTo(_shadowmask); // creates the mask (currently with bgs)
*/
	//ADD YOUR CODE HERE
	//...
	absdiff(_bgsmask, _bgsmask, _shadowmask);// currently void function mask=0 (should create shadow mask)
	//...

	absdiff(_bgsmask, _shadowmask, _fgmask); // eliminates shadows from bgsmask0
}

//ADD ADDITIONAL FUNCTIONS HERE
void bgs::update_bkg(cv::Mat Frame)
{


	ghost_remove();
	single_gaussian(Frame);


}

void bgs::single_gaussian(cv::Mat Frame){
	Mat frame_float = Frame;
	frame_float.convertTo(frame_float, CV_32F);
	Mat new_mean = _alpha4 * frame_float + (1 - _alpha4) * _mean;
	Mat aux = frame_float - _mean;
	cv::multiply(aux, aux, aux);
	Mat new_std;
	Mat std_squared;
	cv::multiply(_std, _std, std_squared);
	cv::sqrt(_alpha4 * aux + (1 - _alpha4) * std_squared, new_std);
	//update just for background
	//invert _bgsmask
	Mat mask_f = _bgsmask;
	mask_f.convertTo(mask_f, CV_32F);
	_mean = (_mean & mask_f) + (new_mean & ~mask_f);
	_std = (_std & mask_f) + (new_std & ~mask_f);

}

void bgs::ghost_remove()
{
	//add to counter the foreground pixels(add 1 everywhere because we remove later anyway)
	_counter += _increment;

	//set to zero pixels detected to background (AND with foreground)
	_counter = _counter & _bgsmask;
	//imshow("counter", _counter);
	//find pixels with count greater than threshold
	cv::Mat aboveThreshold;
	threshold( _counter, aboveThreshold, _threshold, 255, THRESH_BINARY );
	//ghost_removeincorporate in _bgsmask
	_bgsmask = _bgsmask - aboveThreshold;

}

