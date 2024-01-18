#pragma once

#include <iostream>
#include <HalconCpp.h>
#include <opencv.hpp> 
#include <vector> 
#include <Dense>

using namespace HalconCpp;
using std::vector;
#ifdef EIGEN_MAJOR_VERSION
using Eigen::MatrixXd;
#endif


namespace my {
	template<typename T>
	concept CVPoint2 = std::is_same_v<T, cv::Point2i> || std::is_same_v<T, cv::Point2f> \
		|| std::is_same_v<T, cv::Point2d>;

	const std::vector<int> CVPNG_NO_COMPRESSION{ cv::IMWRITE_PNG_COMPRESSION ,0 };
	void imshow_ha(std::string winname, const HalconCpp::HObject& Hobj);
	void imshow_ha(std::string winname, const HalconCpp::HImage& image);
	wchar_t* wGetFileName(const char* title, const wchar_t* defualtPath);
	template<typename T> requires CVPoint2<T>
	auto Hxld2CVpt(const HObject & Hxld) -> std::vector<T>;
	template<typename T> requires std::is_same_v<T, double> || std::is_same_v<T, float>
	auto Hxld2CArray(const HObject& Hxld) -> std::vector<std::vector<T>>;
	template<typename T> requires std::floating_point<T>
	auto EllipticParasToRotRec(T* EllipticParameters) -> cv::RotatedRect;
	void RotRec2param(cv::RotatedRect& rect, float* params);
	auto myfit(cv::InputArray _points, int iterations = 1) -> cv::RotatedRect;
#ifdef EIGEN_MAJOR_VERSION
	Eigen::VectorXd ReweightedLeastSquares(Eigen::MatrixXd& A, Eigen::VectorXd& B, Eigen::VectorXd& vectorW);
	Eigen::VectorXd IterativeReweightedLeastSquares(Eigen::MatrixXd A, Eigen::VectorXd B, double p, int kk);
#endif


	class hwindow {
	public:
		hwindow(const HTuple wwidth, const HTuple wheight, const char* title = "window title");
		hwindow();
		//hwindow(int width, int height, const char* title = "window title");
		hwindow(HalconCpp::HObject& image, const char* title = "window title");

		void setwindowparam(const char* name, const char* param);
		void show(const HalconCpp::HObject& obj) const;
		void show() const;
		void click() const;
		void clearwindow();

		HTuple windowHight;
		HTuple windowWidth;
		HalconCpp::HWindow *w;
		HObject* object{ nullptr };

		~hwindow();
	};

	class LSEllipse
	{
	public:
		LSEllipse(void);
		~LSEllipse(void);
		auto getEllipseparGauss(std::vector<cv::Point2f>& vec_point) -> std::vector<double>;
		void cvFitEllipse2f(float* arrayx, float* arrayy, int n, float* box);
	private:
		int SVD(float* a, int m, int n, float b[], float x[], float esp);
		int gmiv(float a[], int m, int n, float b[], float x[], float aa[], float eps, float u[], float v[], int ka);
		int ginv(float a[], int m, int n, float aa[], float eps, float u[], float v[], int ka);
		int muav(float a[], int m, int n, float u[], float v[], float eps, int ka);
	};

	struct EliipsePara {
		cv::Point2f c;
		float A{ -1 }, B{ -1 }, C{ -1 }, D{ -1 }, E{ -1 }, theta;
		static auto getEliipsePara(cv::RotatedRect& ellipse) -> EliipsePara;
	};

}

template<typename T> requires my::CVPoint2<T>
auto my::Hxld2CVpt(const HObject& Hxld) ->std::vector<T>
{
	std::vector<T> cvContour;
	HTuple HxldROW, HxldCOL;
	GetContourXld(Hxld, &HxldROW, &HxldCOL);
	int num_points = HxldROW.Length();
	cvContour.reserve(num_points);
	for (int j = 0; j < num_points; j++)
	{
		cvContour.push_back(cv::Point2f(HxldCOL[j].D(), HxldROW[j].D()));
	}
	return std::move(cvContour);
}

template<typename T> requires std::is_same_v<T, double> || std::is_same_v<T, float>
auto my::Hxld2CArray(const HObject& Hxld)->std::vector<std::vector<T>>
{
	HTuple HxldROW, HxldCOL;
	GetContourXld(Hxld, &HxldROW, &HxldCOL);
	int num_points = HxldROW.Length();
	std::vector<std::vector<T>> xy;
	std::vector<T> x, y;
	x.reserve(num_points);
	y.reserve(num_points);
	for (int j = 0; j < num_points; j++)
	{
		x.push_back(HxldCOL[j].D());
		y.push_back(HxldROW[j].D());
	}
	xy.push_back(std::move(x));
	xy.push_back(std::move(y));

	return xy;
}

template<typename T> requires std::floating_point<T>
auto my::EllipticParasToRotRec(T* EllipticParameters) -> cv::RotatedRect
{
	cv::RotatedRect r;
	T A = EllipticParameters[0], B = EllipticParameters[1], C = EllipticParameters[2], \
		D = EllipticParameters[3], E = EllipticParameters[4];
	auto a = 2 * std::sqrtf(2 * (A * E * C + B * D * C - C * C - A * E * E - B * D * D) * \
		(A + B + std::sqrt((A - B) * (A - B) + C * C)) / ((A - B) * (A - B) * C));
	auto b = 2 * std::sqrtf(2 * (A * E * C + B * D * C - C * C - A * E * E - B * D * D) * \
		(A + B - std::sqrt((A - B) * (A - B) + C * C)) / ((A - B) * (A - B) * C));
	/*r.center.x = (2 * B * C - D * E) / (D * D - 4 * A * B);
	r.center.y = (2 * A * E - D * C) / (D * D - 4 * A * B);*/
	//r.center.x = (float)EllipticParameters.[5];
	//r.center.y = (float)EllipticParameters.[6];
	r.angle = -0.5 * atan2(C, B - A);
	r.angle =r.angle* 180. / CV_PI;
	if (r.angle < -180)
		r.angle += 360;
	if (r.angle > 360)
		r.angle -= 360;
	if (a < b)
		std::swap(a, b);
	r.size.width = b;
	r.size.height = a;

	return std::move(r);
}

//二维点坐标
struct P2d
{
	double x = 0.0;//横坐标
	double y = 0.0;//纵坐标
	//double angle = 0.0;//角度
	//int number = 0;//序号
	//int pp = 0;//分段
	//double di2 = 0;//点到圆心的距离平方
	//double Dhxy = 0;//点的共焦双曲线距离
};
//椭圆参数
struct Parameter_ell
{
	double xc = 0.0;
	double yc = 0.0;
	double ae = 0.0;
	double be = 0.0;
	double angle = 0.0;
};

#ifdef EIGEN_MAJOR_VERSION
//最小二乘平差椭圆拟合
class Ellispefitting_error_adjustment
{
public:
	Ellispefitting_error_adjustment() = default;
	~Ellispefitting_error_adjustment() = default;

	bool fitting(vector<P2d> input, Parameter_ell& Final_ellispe_par);//椭圆拟合总函数，包含椭圆拟合流程
	void Cal_LSadj(vector<P2d> input, Parameter_ell& Final_ellispe_par);//最小二乘法平差
	void Cal_Y0(vector<P2d> input);//计算初始A,B,C,D,E,F

	void SetX(vector<P2d> input);
	void SetY0(vector<P2d> input);
	Eigen::MatrixXd Dt_Y = MatrixXd::Zero(5, 1);//dltY
	static auto CV2P2d(cv::Point2f) -> P2d;
private:
	MatrixXd Y0 = MatrixXd::Zero(5, 1);//初始A,B,C,D,E,F
	MatrixXd X = MatrixXd::Zero(pointsize, 5);
	int pointsize = 0;//点的数量
};


inline auto Ellispefitting_error_adjustment::CV2P2d(cv::Point2f cvp) -> P2d
{
	P2d p;
	p.x = cvp.x;
	p.y = cvp.y;
	return p;
}
#endif



