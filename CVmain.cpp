//#include "imgproc_M.hpp"
#include <iostream>
#include <opencv.hpp>
#include <HalconCpp.h>
#include <tchar.h>
#include <chrono>
#include "imgalgm.h"
#include <windows.h>

using namespace HalconCpp;

using Himg = HImage;

int main()
{
    try
    {
        HObject  ho_OriginImg, ho_GrayImage, ho_binImg;
        HObject  ho_Contours, ho_ContEllipse;

        ReadImage(&ho_OriginImg, my::wGetFileName("ѡ��ͼ��", L"D:\\TestSet\\"));
        my::imshow_ha("tuxiang", ho_OriginImg);

        Rgb1ToGray(ho_OriginImg, &ho_GrayImage);

        Threshold(ho_GrayImage, &ho_binImg, 128, 255);

        GenContourRegionXld(ho_binImg, &ho_Contours, "border");

        HTuple width, height;
        HalconCpp::GetImageSize(ho_OriginImg, &width, &height);  

        //cv::Mat srcImage = cv::imread("D:\\TestSet\\zh\\2wb.png");
        //cv::Mat srcImage = cv::imread("D:\\TestSet\\zh\\4wbm29i4n0.png");
        cv::Mat srcImage = cv::imread("D:\\TestSet\\zh\\2wb_p1_i1.png");
        //cv::Mat srcImage = cv::imread("D:\\TestSet\\zh\\1wbp1.png");
        //cv::Mat srcImage = cv::imread("D:\\TestSet\\zh\\AA.png");
        cv::imshow("src", srcImage);
        cv::waitKey();

        std::vector<std::vector<cv::Point2f>> xldContours;
        std::vector<std::vector<std::vector<float>>> xldCountours2;
        HTuple numContours;
        CountObj(ho_Contours, &numContours);
        for (int i = 1; i <= numContours.I(); i++)
        {
            HTuple  hv_Row, hv_Column;
            HTuple  hv_Phi, hv_Radius1, hv_Radius2, hv_StartPhi, hv_EndPhi, hv_PointOrder;
            HObject singleContour;
            SelectObj(ho_Contours, &singleContour, i);     
            xldContours.push_back(my::Hxld2CVpt<cv::Point2f>(singleContour));
            xldCountours2.push_back(my::Hxld2CArray<float>(singleContour));
            //std::cout << xldContours[i] << std::endl;
            //my::hwindow xldw1{ ho_OriginImg ,"xld region_current" };
            //xldw1.show(singleContour);
            //xldw1.click();
#ifdef EIGEN_MAJOR_VERSION
            //std::vector<P2d> p2dpts;
            //for (const auto& iter : my::Hxld2CVpt<cv::Point2f>(singleContour))
            //{
            //    p2dpts.push_back(Ellispefitting_error_adjustment::CV2P2d(iter));
            //}
            //Parameter_ell elpm;
            //Ellispefitting_error_adjustment fitelp;
            //fitelp.fitting(p2dpts, elpm);
            //std::cout << "x,y,a,b,angle= " << elpm.xc << "," << elpm.yc << "," << elpm.ae << "," << elpm.be << "," << elpm.angle*180/3.1415926 << std::endl;
            //getchar();
            //cv::RotatedRect epl;
            //epl.center.x = elpm.xc;
            //epl.center.y = elpm.yc;
            //epl.angle = elpm.angle;
            //epl.size.width = elpm.ae * 2;
            //epl.size.height = elpm.be * 2;
            //cv::ellipse(srcImage, epl, cv::Scalar(255, 0, 255), 1, cv::LineTypes::LINE_AA);
            //cv::imshow("reslt1", srcImage);
            //cv::waitKey();
#endif // EIGEN_MAJOR_VERSION

            int iters = 5;
            HTuple row, col;
            GetContourXld(singleContour, &row, &col);
            if (col.Length() < 30)
                continue;
            std::cout << "[" << i << "]��������=" << col.Length() << std::endl;
            cv::RotatedRect ellipse1;
            cv::Mat points = static_cast<cv::_InputArray>(xldContours[i - 1]).getMat();
            float coffs[7]{ -1 }, rcoffs[7]{ -1 };
            //auto xyAarray = my::Hxld2CArray<float>(singleContour);
            //my::LSEllipse fitobj;
            auto t1_st = std::chrono::high_resolution_clock::now();
            //ellipse1 = cv::fitEllipse(xldContours[i-1]);
#ifdef MY_OPENCV_IMGPROC_HPP
            cv::myfitellipse(points, ellipse1);
            auto t1_ed = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = t1_ed - t1_st;
            std::cout << "call from cv:" << duration << std::endl;
            cv::ellipse(srcImage, ellipse1, cv::Scalar(0, 0, 255), 1, cv::LineTypes::LINE_AA);
            cv::_EPms<float> ellipseParas;
#endif // MY_OPENCV_IMGPROC_HPP



            //cv::ellipse(srcImage,ellipse1.center,)
            //cv::imshow("cv_rslt", srcImage);
            //cv::waitKey();

            FitEllipseContourXld(singleContour, "ftukey", -1, 2, 0, 200, 1, 2, &hv_Row, &hv_Column,
                &hv_Phi, &hv_Radius1, &hv_Radius2, &hv_StartPhi, &hv_EndPhi, &hv_PointOrder);
            GenEllipseContourXld(&ho_ContEllipse, hv_Row, hv_Column, hv_Phi, hv_Radius1, hv_Radius2,
                hv_StartPhi, hv_EndPhi, hv_PointOrder, 0.7);
            for (const auto& iter : my::Hxld2CVpt<cv::Point2f>(ho_ContEllipse))
            {
                srcImage.at<cv::Vec3b>(iter) = cv::Vec3b{ 0,255,255 };
            }
             
            auto t1_st3 = std::chrono::high_resolution_clock::now();
            FitEllipseContourXld(singleContour, "ftukey", -1, 2, 0, 200, iters, 2, &hv_Row, &hv_Column,
                &hv_Phi, &hv_Radius1, &hv_Radius2, &hv_StartPhi, &hv_EndPhi, &hv_PointOrder);
            auto t1_ed3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration3 = t1_ed3 - t1_st3;
            std::cout << "call from halcon:" << duration3 << std::endl;
            GenEllipseContourXld(&ho_ContEllipse, hv_Row, hv_Column, hv_Phi, hv_Radius1, hv_Radius2,
                hv_StartPhi, hv_EndPhi, hv_PointOrder, 0.7);
            //my::hwindow ellipsew{ ho_OriginImg ,"hal_rslt" };
            //ellipsew.show(ho_ContEllipse);
            //ellipsew.click();
            //std::cout << "CV time / HALCON time=" << (duration / duration2) << std::endl << std::endl;

            for (const auto& iter : my::Hxld2CVpt<cv::Point2f>(ho_ContEllipse))
            {
                srcImage.at<cv::Vec3b>(iter) = cv::Vec3b{ 255,0,0 };
            }
            //cv::imshow("cv_hal_rslt", srcImage);
            //cv::waitKey();
            //cv::imshow("jg", srcImage);
            //cv::waitKey();

            auto t2_st = std::chrono::high_resolution_clock::now();
            //ellipse1 = cv::SVDfitEllipse(points, iters);
            ellipse1 = my::myfit(points, iters);
            auto t2_ed = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration2 = t2_ed - t2_st;
            std::cout << "call from cv_it:" << duration2 << std::endl;
            //ellipse1.center.x = hv_Column.ToDArr()[0];
            //ellipse1.center.y = hv_Row.ToDArr()[0];
            //ellipse1.angle = -hv_Phi.ToDArr()[0] * 180. / 3.1415926 + 90;
            cv::ellipse(srcImage, ellipse1, cv::Scalar(0, 255, 0), 1, cv::LineTypes::LINE_AA);
        }

        //// Find the largest contour
        //auto largestContour = xldContours[0];
        //for (const auto& iter: xldContours)
        //{
        //    if (iter.size()>largestContour.size())
        //        largestContour = iter;
        //}

        //cv::Mat frmH(srcImage.rows, srcImage.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        //std::vector<cv::Point2f> xldEllipsecont = my::Hxld2CVpt<cv::Point2f>(ho_ContEllipse);
        //for (const auto& iter : my::Hxld2CVpt<cv::Point2f>(ho_ContEllipse))
        //{
        //    srcImage.at<cv::Vec3b>(iter) = cv::Vec3b{ 255,0,0 };
        //}
        cv::destroyAllWindows();
        cv::imshow("cv_hal_rslt", srcImage);
        cv::waitKey();
        cv::imwrite("D:\\TestSet\\SPCrslt\\Out_cv+hal_resualt_17.png", srcImage, my::CVPNG_NO_COMPRESSION);

        //imshow_ha("jieguo", ho_ContEllipse);
    }
    catch (HException& HDevExpDefaultException)
    {
        printf("%s\n", HDevExpDefaultException.ErrorMessage().Text());
    }
    
}
