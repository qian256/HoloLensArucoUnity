#include "src/aruco.h"
#include "src/cvdrawingutils.h"
#include <opencv2/imgproc/imgproc.hpp>

extern "C" {
	__declspec(dllexport) int getInt() {
		return 10;
	}

	__declspec(dllexport) int getSize() {
		cv::Mat m(100, 100, CV_8UC3);
		return m.cols * m.rows;
	}

	class arucoController {
	public:
		arucoController() : CamParam(), MDetector() {
			MarkerSize = 0.1;
			MDetector.setThresholdParams(7, 7);
			MDetector.setThresholdParamRange(2, 0);
			rows = 0;
			cols = 0;
			image = NULL;
		}

		void setImage(uchar* imageData) {
			delete image;
			image = new cv::Mat(rows, cols, CV_8UC4, imageData, cols * 4);
			cv::cvtColor(*image, gray, CV_BGRA2GRAY);
		}

		void setSize(int row, int col) {
			rows = row;
			cols = col;
		}

		void detect() {
			try {
				Markers = MDetector.detect(gray, CamParam, MarkerSize);
			}
			catch (std::exception &ex) {
				std::cerr << "Exception :" << ex.what() << endl;
			}
		}

		int getNumMarkers() {
			return (int)(Markers.size());
		}

		uchar* getProcessedImage() {
			if (image == NULL) {
				std::cerr << "Error : image not set" << endl;
				return NULL;
			}
			processedImage = cv::Mat(*image);
			for (unsigned int i = 0; i < Markers.size(); i++) {
				Markers[i].draw(processedImage, cv::Scalar(0, 0, 255), 2);
			}
			return processedImage.data;
		}


		~arucoController() {
			delete image;
		}

		int getCols() {
			return cols;
		}

		int getRows() {
			return rows;
		}

	private:
		aruco::CameraParameters CamParam;
		aruco::MarkerDetector MDetector;
		float MarkerSize;
		cv::Mat * image;
		cv::Mat gray;
		int rows, cols;
		cv::Mat processedImage;
		std::vector<aruco::Marker>  Markers;
	};

	arucoController* ac;
	bool initialized = false;

	__declspec(dllexport) void initArucoController() {
		if (!initialized) {
			ac = new arucoController();
			initialized = true;
		}
	}

	__declspec(dllexport) void destroyArucoController() {
		if (initialized)
			delete ac;
		initialized = false;
	}

	__declspec(dllexport) int isInitialized() {
		if (initialized)
			return 1;
		return 0;
	}

	__declspec(dllexport) void newImage(uchar* imageData) {
		if (initialized) {
			ac->setImage(imageData);
		}
	}

	__declspec(dllexport) void setImageSize(int row, int col) {
		if (initialized)
			ac->setSize(row, col);
	}

	__declspec(dllexport) void detect() {
		if (initialized)
			ac->detect();
	}

	__declspec(dllexport) int getNumMarkers() {
		if (initialized)
			return ac->getNumMarkers();
		else
			return -2;
	}

	__declspec(dllexport) int getRows() {
		if (initialized)
			return ac->getRows();
		else
			return -2;
	}

	__declspec(dllexport) int getCols() {
		if (initialized)
			return ac->getCols();
		else
			return -2;
	}

	__declspec(dllexport) uchar* getProcessedImage() {
		if (initialized)
			return ac->getProcessedImage();
		return NULL;
	}


}

