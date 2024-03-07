/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.hpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */


#include <opencv2/opencv.hpp>

#ifndef FGSEG_H_INCLUDE
#define FGSEG_H_INCLUDE

using namespace cv;
using namespace std;

namespace fgseg {


	//Declaration of FGSeg class based on BackGround Subtraction (bgs)
	class bgs{
	public:

		//constructor with parameter "tau"
		bgs(double tau, bool rgb, int threshold_ghosts2, double alpha4);

		//destructor
		~bgs(void);

		//method to initialize bkg (first frame - hot start)
		void init_bkg(cv::Mat Frame);

		//method to perform BackGroundSubtraction
		void bkgSubtraction(cv::Mat Frame);

		//method to detect and remove shadows in the binary BGS mask
		void removeShadows();

		//returns the BG image
		cv::Mat getBG(){return _bkg;};

		//returns the DIFF image
		cv::Mat getDiff(){return _diff;};

		//returns the BGS mask
		cv::Mat getBGSmask(){return _bgsmask;};

		//returns the binary mask with detected shadows
		cv::Mat getShadowMask(){return _shadowmask;};

		//returns the binary FG mask
		cv::Mat getFGmask(){return _fgmask;};


		//ADD ADITIONAL METHODS HERE
		//...
		void update_bkg(cv::Mat Frame);
		void ghost_remove();
		void single_gaussian(cv::Mat Frame);

	private:
		cv::Mat _bkg; //Background model
		cv::Mat	_frame; //current frame
		cv::Mat _diff; //abs diff frame
		cv::Mat _bgsmask; //binary image for bgssub (FG)
		cv::Mat _shadowmask; //binary mask for detected shadows
		cv::Mat _fgmask; //binary image for foreground (FG)

		bool _rgb;

		double _tau;
		//ADD ADITIONAL VARIABLES HERE
		//...
		//double _alpha;
		//double _shdalpha;
		//double _shdbetha;
		//double _taus;
		//double _tauh;
		int _threshold;

		cv::Mat _counter;
		cv::Mat _increment;
		double _alpha4;
		cv::Mat _mean;
		cv::Mat _std;

	};//end of class bgs

}//end of namespace

#endif




