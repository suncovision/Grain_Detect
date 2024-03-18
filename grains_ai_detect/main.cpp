#include "detector.h"
#include "test.h"
#include <vector>
#include <fstream>


int main()
{
	system("color 3F");

	std::vector<std::string> class_names = LoadNames("./weights/coco.names");   // 源文件为  ../weights/coco.names
	if (class_names.empty()) {
		return -1;
	}
	bool view_img = 1;
	torch::DeviceType device_type = torch::kCUDA;
	//std::string weights = "./weights/best.torchscript.pt";
	std::string weights = "E:\\PyCharm 2023.1\\project\\yolov5-6.0_modify\\runs\\train\\exp88\\weights\\best.torchscript.pt"; //best model is exp83
	auto detector = Detector(weights, device_type);
	
	cv::String path0 = "./images";
	std::vector<cv::String> files0;
	cv::glob(path0, files0, false);
	std::vector<cv::Mat> images0;
	for (int i = 0; i < files0.size(); i++)
	{
		cv::Mat src = cv::imread(files0[i]);
		images0.push_back(src);
	}
	
	infos info;
	std::vector<std::vector<infos>> result_infos;   //这里包含了每张图像中每个缺陷的检测信息
	float conf_thres = 0.2;  // 置信度阈值，域值越大，检测结果越少
	int thre_roi = 30;
	std::cout << "初始化开始..." << std::endl;
	detect1(images0, info, result_infos, conf_thres, thre_roi, class_names, detector, view_img);
	std::cout << "检测开始" << std::endl;
	cv::String path = "C:\\Users\\Lenovo\\Desktop\\6";
	std::vector<cv::String> files;
	cv::glob(path, files, false);
	
	std::vector<cv::Mat> images;
	for (int i = 0; i < files.size(); i++)
	{
		cv::Mat src = cv::imread(files[i]);
		images.push_back(src);
	}
	std::cout << "image size: " << images.size() << std::endl;
	detect(images, info, result_infos, conf_thres, thre_roi, class_names, detector, view_img);

	/*std::ofstream oFile;
	oFile.open("C:\\Users\\Lenovo\\Desktop\\9\\0\\grains.csv", std::ios::out | std::ios::trunc);
	oFile << "x" << "," << "y" << "," << "w" << "," << "h" << std::endl;*/

	for (int i = 0; i < result_infos.size(); i++)
	{
		std::cout << "\n===========================" << std::endl;
		for (int j = 0; j < result_infos[i].size(); j++)
		{
			std::cout << result_infos[i][j].location << std::endl;  // (左上角x，左上角y，宽，高)
			std::cout << result_infos[i][j].cls << std::endl;       // 类别
			std::cout << result_infos[i][j].conf << std::endl;      // 置信度
			/*oFile << result_infos[i][j].location[0] << ","
				  << result_infos[i][j].location[1] << ","
				  << result_infos[i][j].location[2] << ","
				  << result_infos[i][j].location[3] << std::endl;*/
		}
	}
	//oFile.close();
	std::ofstream oFile;
	oFile.open("C:\\Users\\Lenovo\\Desktop\\9\\grains.csv", std::ios::out | std::ios::trunc);
	

	std::vector<int> nums;
	int c = 1;
	for (int k = 0; k < result_infos.size(); k++)
	{
		std::cout << result_infos[k].size() << std::endl;
		if (result_infos[k].size() != 0)
		{
			nums.push_back(result_infos[k].size());
			int num = result_infos[k].size();
			oFile << "第" + std::to_string(c) + "张图" << "," << num << std::endl;

			c += 1;
		}
	}
	
	auto minElement = std::min_element(nums.begin(), nums.end());
	auto maxElement = std::max_element(nums.begin(), nums.end());

	std::cout << "min: " << *minElement << std::endl;
	std::cout << "max: " << *maxElement << std::endl;
	std::cout << "diff: " << *maxElement - *minElement << std::endl;

	return 0;
}