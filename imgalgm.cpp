#include "imgalgm.h"
#include <cmath>
#include <tchar.h>
#include <shlobj.h>

using std::vector, cv::Mat, cv::Point2f;

namespace my {

#define __imshow_ha__(winname,image)                  \
{                                                     \
	HTuple width, height;                             \
	                                                  \
	HalconCpp::GetImageSize(image, &width, &height);  \
	HWindow w(0, 0, width, height);                   \
	w.SetWindowParam("window_title", winname.c_str());\
	                                                  \
	image.DispObj(w);                                 \
	w.Click();                                        \
	/*w.ClearWindow();*/                                  \
}

    void imshow_ha(std::string winname, const HalconCpp::HObject& image)
    {
        __imshow_ha__(winname, image);
    }
    void imshow_ha(std::string winname, const HalconCpp::HImage& image)
    {
        __imshow_ha__(winname, image);
    }

    void CharToTchar(const char* _char, TCHAR* tchar)
    {
        int iLength;

        iLength = MultiByteToWideChar(CP_ACP, 0, _char, strlen(_char) + 1, NULL, 0);
        MultiByteToWideChar(CP_ACP, 0, _char, strlen(_char) + 1, tchar, iLength);
    }

    wchar_t* wGetFileName(const char* title, const wchar_t* defualtPath)
    {
        OPENFILENAME ofn;
        wchar_t* fileName = new wchar_t[200];
        wchar_t* wtitle = new TCHAR[200](0);
        CharToTchar(title, wtitle);
        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lpstrTitle = L"选择文件";
        ofn.lStructSize = sizeof(OPENFILENAME);
        ofn.lpstrFilter = L"图像文件\0*.bmp;*.jpg;*.png;*.tif;*.gif;*.jpeg;*.jpe;*.jfif\0All Files\0*.*\0";
        ofn.lpstrInitialDir = defualtPath;//默认的文件路径 
        ofn.lpstrFile = fileName;
        ofn.nMaxFile = MAX_PATH;
        ofn.lpstrTitle = wtitle;
        ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_ALLOWMULTISELECT;
        if (GetOpenFileName(&ofn))
        {
            return fileName;
        }
        return nullptr;
    }

    std::string Wchar_tToString(wchar_t* wchar)
    {
        wchar_t* wText = wchar;
        BOOL B = FALSE;
        LPBOOL LB = &B;
        DWORD dwNum = WideCharToMultiByte(CP_OEMCP, NULL, wText, -1, NULL, 0, NULL, LB);
        char* psText = new char[dwNum]; 
        WideCharToMultiByte(CP_OEMCP, NULL, wText, -1, psText, dwNum, NULL, LB);
        std::string szDst = psText;
        delete[] psText;

        return szDst;
    }
    std::string TCHAR2STRING(wchar_t* str)
    {
        std::string strstr;
            int iLen = WideCharToMultiByte(CP_ACP, 0, str, -1, NULL, 0, NULL, NULL);
            char* chRtn = new char[iLen * sizeof(char)];
            WideCharToMultiByte(CP_ACP, 0, str, -1, chRtn, iLen, NULL, NULL);
            strstr = chRtn;
     
        return strstr;
    }

    HWND _hwndf;
    //可包含中文
    std::string GetFileName(const char* title, const wchar_t* defualtPath)
    {
        TCHAR* szBuffer = new TCHAR[200](0);
        TCHAR* wtitle = new TCHAR[200](0);
        CharToTchar(title, wtitle);
        OPENFILENAME filepath{ 0 };
        filepath.lStructSize = sizeof(filepath);
        filepath.hwndOwner = _hwndf;
        filepath.lpstrFilter = _T("jpg文件(*.jpg)\0*.jpg\0png文件(*.png)\0*.png\0所有文件(*.*)\0*.*\0");
        filepath.lpstrInitialDir = defualtPath;//默认的文件路径 
        filepath.lpstrFile = szBuffer;//存放文件的缓冲区 
        filepath.nMaxFile = 200;
        filepath.nFilterIndex = 0;
        filepath.lpstrTitle = wtitle;
        filepath.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_EXPLORER | OFN_ALLOWMULTISELECT;
        BOOL bSel = GetOpenFileName(&filepath);

        std::string filename = Wchar_tToString(szBuffer);
        for (auto& i : filename)
        {
            if (i == '\\')
                i = '/';
        }

        delete[] szBuffer;
        szBuffer = nullptr;
        return filename;
    }

    std::string GetSaveName(const char* title, const wchar_t* defualtPath)
    {
        TCHAR* szBuffer = new TCHAR[200](0);
        TCHAR* wtitle = new TCHAR[200](0);
        CharToTchar(title, wtitle);
        OPENFILENAME filepath{ 0 };
        filepath.lStructSize = sizeof(filepath);
        filepath.hwndOwner = _hwndf; 
        filepath.lpstrFilter = _T("png文件(*.png)\0*.png\0jpg文件(*.jpg)\0*.jpg\0");
        filepath.lpstrInitialDir = defualtPath;//默认的文件路径 
        filepath.lpstrFile = szBuffer;//存放文件的缓冲区 
        filepath.nMaxFile = 200;
        filepath.nFilterIndex = 0;
        filepath.lpstrTitle = wtitle;
        filepath.Flags =  OFN_EXPLORER ;
        BOOL bSel = GetSaveFileName(&filepath);

        std::string filename = Wchar_tToString(szBuffer);
        //for (auto& i : filename)
        //{
        //    if (i == '\\')
        //        i = '/';
        //}

        delete[] szBuffer;
        return filename;
    }

    std::string GetFolderName(const char* title)
    {
        TCHAR* path = new TCHAR[260](0);
        TCHAR* wtitle = new TCHAR[260](0);
        CharToTchar(title, wtitle);
        std::string dirname;
        BROWSEINFO bi;
        bi.hwndOwner = _hwndf;
        bi.pidlRoot = NULL;
        bi.pszDisplayName = path;
        bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_USENEWUI | BIF_UAHINT;
        bi.lpfn = NULL;
        bi.lpszTitle = wtitle;
        LPITEMIDLIST pidl = SHBrowseForFolder(&bi);

        SHGetPathFromIDList(pidl, path);
        dirname = my::TCHAR2STRING(path);
        for (auto& i : dirname)
        {
            if (i == '\\')
                i = '/';
        }

        delete[] path;
        return dirname;
    }

    template<class T>
    static auto weight(double* dd, double* wd, int n, int coef)
    {
        std::unique_ptr<double[]> dsp = std::make_unique<double[]>(n);
        double* dsd = dsp.get();
        Mat ds(n, 1, CV_64F, dsd);
        Mat d(n, 1, CV_64F, dd);
        T median{ 0 }, tao{ 0 };
        cv::sort(d, ds, cv::SORT_ASCENDING);

        if (n % 2)
        {
            median = dsd[n/2] + dsd[n / 2 - 1];
            median /= (2. * 0.6745);
        }
        else
        {
            median = dsd[n / 2] / 0.6745;
        }
        tao = coef * median;
        //std::cout << "\nmedian=" << median << " ,n=" << d.rows << " ,tao=" << tao << std::endl;
        //getchar();

        double tao2 = tao * tao;
        for (int i = 0; i < n; ++i)
        {
            if (std::fabs(dd[i]) > tao)
                wd[i] = 1e-8;
            else
            {
                double wi = (1 - (dd[i] * dd[i] / tao2));
                wi *= wi;
                wd[i] = isinf(wi) ? 1.0 : wi;
            }
            //std::cout << "\nw[" << i << "] = " << w.at<T>(cv::Point(i, 1)) << std::endl;
        }
    }
    auto myfit(cv::InputArray _points, int iterations) -> cv::RotatedRect
    {
        const double eps = 1e-12;
        auto points = _points.getMat();
        int i, n = points.checkVector(2);
        std::unique_ptr<double[]> pDatas = std::make_unique<double[]>(n * 5 + n + n + n);
        double* Md = pDatas.get(), * fd = Md + 5*n, * dd = fd + n, * wd = dd + n;
        Mat M(n, 5, CV_64F, Md);
        Mat f(n, 1, CV_64F, fd);
        Mat x(5, 1, CV_64F);
        Mat Q(2, 2, CV_64F);
        Mat N(2, 2, CV_64F);
        Mat d(n, 1, CV_64F, dd);
        Mat w(n, 1, CV_64F, wd);
        Point2f* const ptsf = points.ptr<Point2f>();
        std::unique_ptr<Point2f[]> ptsptr = std::make_unique < Point2f[]>(n);
        Point2f* ptsfc = ptsptr.get();
        cv::Point2f c0(0, 0), c(0, 0);
        double A, B, C, D, E;
        cv::RotatedRect box;
        cv::Point2f p;

        for (i = 0; i < n; ++i)
        {
            c0 += ptsf[i];
        }
        c0 /= n;

#pragma region first fit
        for (i = 0; i < n; ++i)
        {
            ptsfc[i] = ptsf[i] - c0;
            p = ptsfc[i];
            fd[i] = 1;
            Md[5 * i + 0] = (double)ptsfc[i].x * ptsfc[i].x;
            Md[5 * i + 1] = (double)ptsfc[i].x * ptsfc[i].y;
            Md[5 * i + 2] = (double)ptsfc[i].y * ptsfc[i].y;
            Md[5 * i + 3] = (double)ptsfc[i].x;
            Md[5 * i + 4] = (double)ptsfc[i].y;
        }
        cv::solve(M, f, x, cv::DecompTypes::DECOMP_SVD);
        A = x.ptr<double>(0)[0];
        B = x.ptr<double>(1)[0];
        C = x.ptr<double>(2)[0];
        D = x.ptr<double>(3)[0];
        E = x.ptr<double>(4)[0];
        
        x = Mat(2, 1, CV_64F);
        Q.ptr<double>(0)[0] = -2 * A;
        Q.ptr<double>(0)[1] = Q.ptr<double>(1)[0] = -B;
        Q.ptr<double>(1)[1] = -2 * C;
        Q.ptr<double>(0)[0] = D;
        N.ptr<double>(1)[0] = E;
        cv::solve(Q, N, x, cv::DecompTypes::DECOMP_SVD);
        c = cv::Point2f(x.ptr<double>(0)[0], x.ptr<double>(1)[0]);
        auto xc = c.x, yc = c.y;

        M = Mat(n, 3, CV_64F, Md);
        x = Mat(3, 1, CV_64F);
        for (i = 0; i < n; ++i)
        {
            p = ptsfc[i] - c;
            Md[3 * i + 0] = (double)p.x * p.x;
            Md[3 * i + 1] = (double)p.x * p.y;
            Md[3 * i + 2] = (double)p.y * p.y;
        }
        cv::solve(M, f, x, cv::DecompTypes::DECOMP_SVD);
        A = x.ptr<double>(0)[0];
        B = x.ptr<double>(1)[0];
        C = x.ptr<double>(2)[0];
#pragma endregion        

#pragma region Iterations
        for (int iters = 1; iters < iterations; ++iters)
        {
            
            for (i = 0; i < n; ++i)
            {
                p = ptsfc[i] - c;
                d.ptr<double>(i)[0] = fabs(A * p.x * p.x + B * p.x * p.y + C * p.y * p.y + D * p.x + E * p.y - 1);
            }
            weight<double>(dd, wd, n, 2);
            M = Mat(n, 5, CV_64F, Md);
            x = Mat(5, 1, CV_64F);
            for (i = 0; i < n; ++i)
            {
                p = ptsfc[i] - c;
                Md[5 * i + 0] = (double)p.x * p.x * wd[i];
                Md[5 * i + 1] = (double)p.x * p.y * wd[i];
                Md[5 * i + 2] = (double)p.y * p.y * wd[i];
                Md[5 * i + 3] = (double)p.x * wd[i];
                Md[5 * i + 4] = (double)p.y * wd[i];
                fd[i] = wd[i];
            }
            cv::solve(M, f, x, cv::DecompTypes::DECOMP_SVD);
            A = x.ptr<double>(0)[0];
            B = x.ptr<double>(1)[0];
            C = x.ptr<double>(2)[0];
            D = x.ptr<double>(3)[0];
            E = x.ptr<double>(4)[0];

            x = Mat(2, 1, CV_64F);
            Q.ptr<double>(0)[0] = -2 * A;
            Q.ptr<double>(0)[1] = Q.ptr<double>(1)[0] = -B;
            Q.ptr<double>(1)[1] = -2 * C;
            N.ptr<double>(0)[0] = D;
            N.ptr<double>(1)[0] = E;
            cv::solve(Q, N, x, cv::DecompTypes::DECOMP_SVD);
            xc = x.ptr<double>(0)[0], yc = x.ptr<double>(1)[0];
            c += cv::Point2f(xc, yc);

            M = Mat(n, 3, CV_64F, Md);
            x = Mat(3, 1, CV_64F);
            for (i = 0; i < n; ++i)
            {
                p = ptsfc[i] - c;
                Md[3 * i + 0] = (double)p.x * p.x * wd[i];
                Md[3 * i + 1] = (double)p.x * p.y * wd[i];
                Md[3 * i + 2] = (double)p.y * p.y * wd[i];
            }
            cv::solve(M, f, x, cv::DecompTypes::DECOMP_SVD);
            A = x.ptr<double>(0)[0];
            B = x.ptr<double>(1)[0];
            C = x.ptr<double>(2)[0];
        }
#pragma endregion

#pragma region Caculate Parameters
        double angle = 0.5 * std::atan2(B, A - C);
        double t, a, b;
        if (std::abs(B) <= eps)
        {
            t = A - C;
        }
        else {
            t = B / std::sin(-2 * angle);
        }
        a = std::fabs(A + C - t);
        if (a > eps)
            a = std::sqrt(2. / a);
        b = std::fabs(A + C + t);
        if (b > eps)
            b = std::sqrt(2. / b);

        box.center = c0 + c;
        box.angle = angle * 180 / 3.1415926;
        box.size.width = a * 2;
        box.size.height = 2 * b;
        if (box.size.width > box.size.height)
        {
            cv::swap(box.size.width, box.size.height);
            box.angle += 90;
        }
        if (box.angle < -180)
            box.angle += 360;
        if (box.angle > 360)
            box.angle -= 360;
#pragma endregion
        return box;
    }

#ifdef EIGEN_MAJOR_VERSION
    Eigen::VectorXd ReweightedLeastSquares(Eigen::MatrixXd& A, Eigen::VectorXd& B, Eigen::VectorXd& vectorW)
    {
        //获取矩阵的行数和列数
        int rows = A.rows();
        int col = A.cols();
        //vectorW为空,默认构建对角线矩阵1
        //qDebug()<<"vectorW.isZero():"<<vectorW.isZero();
        if (vectorW.isZero())
        {
            vectorW.resize(rows, 1);
            for (int i = 0; i < rows; ++i)
            {
                vectorW(i, 0) = 1;
            }
        }

        //A的转置矩阵
        Eigen::MatrixXd AT(col, rows);
        // AT.resize(col,rows);
         
        //x矩阵
        Eigen::VectorXd x(col, 1);
        //x.resize(col,1);

        //W的转置矩阵
        Eigen::MatrixXd WT(rows, rows), W(rows, rows);
        // W.resize(rows,rows);
        // WT.resize(rows,rows);

         //生成对角线矩阵
        W = vectorW.asDiagonal();
        //转置
        WT = W.transpose();
        //转置 AT
        AT = A.transpose();

        // x = (A^T * W^T * W * A)^-1 * A^T * W^T * W * B
        x = ((AT * WT * W * A).inverse()) * (AT * WT * W * B);
        return x;
    }
    /* 迭代重加权最小二乘（IRLS）  W为权重,p为范数
    * e = Ax - B
    * W = e^(p−2)/2
    * W²(Ax - B) = 0
    * W²Ax = W²B
    * (A^T * W^T * W * A) * x = A^T * W^T * W * B
    * x = (A^T * W^T * W * A)^-1 * A^T * W^T * W * B
    * 参考论文地址：https://www.semanticscholar.org/paper/Iterative-Reweighted-Least-Squares-%E2%88%97-Burrus/9b9218e7233f4d0b491e1582c893c9a099470a73
    */
    Eigen::VectorXd IterativeReweightedLeastSquares(Eigen::MatrixXd A, Eigen::VectorXd B, double p, int kk)
    {


        /* x(k) = q x1(k) + (1-q)x(k-1)
         * q = 1 / (p-1)
         */
         //获取矩阵的行数和列数
        int rows = A.rows();
        int col = A.cols();

        double pk = 2;//初始同伦值
        double K = 1.5;

        double epsilon = 10e-9; // ε
        double delta = 10e-15; // δ
        Eigen::VectorXd x(col, 1), _x(col, 1), x1(col, 1), e(rows, 1), w = Eigen::MatrixXd::Identity(rows, rows);
        for (int i = 0; i < rows; ++i)
        {
            w[i, i] = 0;
        }

        //初始x  对角矩阵w=1
        x = ReweightedLeastSquares(A, B, w);

        //迭代  最大迭代次数kk
        for (int i = 0; i < kk; ++i)
        {
            //保留前一个x值,用作最后比较确定收敛
            _x = x;

            if (p >= 2)
            {
                pk = std::min(p, K * pk);
            }
            else
            {
                pk = std::max(p, K * pk);
            }
            //偏差
            e = (A * x) - B;
            //偏差的绝对值//  求矩阵绝对值 ：e = e.cwiseAbs(); 或 e.array().abs().matrix()
            e = e.array().abs();
            //对每个偏差值小于delta,用delta赋值给它
            for (int i = 0; i < e.rows(); ++i)
            {
                e(i, 0) = std::max(delta, e(i, 0));
            }
            //对每个偏差值进行幂操作
            w = e.array().pow(p / 2.0 - 1);
            w = w / w.sum();

            x1 = ReweightedLeastSquares(A, B, w);

            double q = 1 / (pk - 1);
            if (p > 2)
            {
                x = x1 * q + x * (1 - q);
            }
            else
            {
                x = x1;
            }
            //达到精度,结束  模1
            if ((x - _x).array().abs().sum() < epsilon)
            {
                return x;
            }
        }
        return x;

    }
#endif

    hwindow::hwindow(const HTuple width, const HTuple height, const char* title)
    {
        w = new HWindow{ 0,0,width,height };
        w->SetWindowParam("window_title", title);
    }
    hwindow::hwindow()
        :hwindow(0, 0)
    {
    }
    //hwindow::hwindow(int width, int height, const char* title)
    //	:hwindow(static_cast< HTuple>(width), static_cast<HTuple>(width), title)
    //{
    //}
    hwindow::hwindow(HalconCpp::HObject& image, const char* title)
    {
        HTuple width, height;
        HalconCpp::GetImageSize(image, &width, &height);
        object = &image;
        w = new HWindow{ 0, 0, width, height };
        w->SetWindowParam("window_title", title);
    }
    void hwindow::setwindowparam(const char* name, const char* param)
    {
        if (w != nullptr)
        {
            w->SetWindowParam(name, param);
        }
    }
    void hwindow::show(const HalconCpp::HObject& obj) const
    {
        w->DispObj(obj);
    }
    void hwindow::show() const
    {
        w->DispObj(*object);
    }
    void hwindow::click() const
    {
        w->Click();
    }
    void hwindow::clearwindow()
    {
        w->ClearWindow();
    }
    hwindow::~hwindow()
    {
        delete w;
    }

    LSEllipse::LSEllipse(void)
    {
    }


    LSEllipse::~LSEllipse(void)
    {
    }
    //列主元高斯消去法  
    //A为系数矩阵，x为解向量，若成功，返回true，否则返回false，并将x清空。  

    bool RGauss(const vector<vector<double> >& A, vector<double>& x)
    {
        x.clear();
        int n = A.size();
        int m = A[0].size();
        x.resize(n);
        //复制系数矩阵，防止修改原矩阵  
        vector<vector<double> > Atemp(n);
        for (int i = 0; i < n; i++)
        {
            vector<double> temp(m);
            for (int j = 0; j < m; j++)
            {
                temp[j] = A[i][j];
            }
            Atemp[i] = temp;
            temp.clear();
        }
        for (int k = 0; k < n; k++)
        {
            //选主元  
            double max = -1;
            int l = -1;
            for (int i = k; i < n; i++)
            {
                if (abs(Atemp[i][k]) > max)
                {
                    max = abs(Atemp[i][k]);
                    l = i;
                }
            }
            if (l != k)
            {
                //交换系数矩阵的l行和k行  
                for (int i = 0; i < m; i++)
                {
                    double temp = Atemp[l][i];
                    Atemp[l][i] = Atemp[k][i];
                    Atemp[k][i] = temp;
                }
            }
            //消元  
            for (int i = k + 1; i < n; i++)
            {
                double l = Atemp[i][k] / Atemp[k][k];
                for (int j = k; j < m; j++)
                {
                    Atemp[i][j] = Atemp[i][j] - l * Atemp[k][j];
                }
            }
        }
        //回代  
        x[n - 1] = Atemp[n - 1][m - 1] / Atemp[n - 1][m - 2];
        for (int k = n - 2; k >= 0; k--)
        {
            double s = 0.0;
            for (int j = k + 1; j < n; j++)
            {
                s += Atemp[k][j] * x[j];
            }
            x[k] = (Atemp[k][m - 1] - s) / Atemp[k][k];
        }
        return true;
    }

    vector<double>  LSEllipse::getEllipseparGauss(vector<cv::Point2f>& vec_point)
    {
        vector<double> vec_result;
        double x3y1 = 0, x1y3 = 0, x2y2 = 0, yyy4 = 0, xxx3 = 0, xxx2 = 0, x2y1 = 0, yyy3 = 0, x1y2 = 0, yyy2 = 0, x1y1 = 0, xxx1 = 0, yyy1 = 0;
        int N = vec_point.size();
        for (int m_i = 0; m_i < N; ++m_i)
        {
            double xi = vec_point[m_i].x;
            double yi = vec_point[m_i].y;
            x3y1 += xi * xi * xi * yi;
            x1y3 += xi * yi * yi * yi;
            x2y2 += xi * xi * yi * yi; ;
            yyy4 += yi * yi * yi * yi;
            xxx3 += xi * xi * xi;
            xxx2 += xi * xi;
            x2y1 += xi * xi * yi;

            x1y2 += xi * yi * yi;
            yyy2 += yi * yi;
            x1y1 += xi * yi;
            xxx1 += xi;
            yyy1 += yi;
            yyy3 += yi * yi * yi;
        }
        double resul[5];
        resul[0] = -(x3y1);
        resul[1] = -(x2y2);
        resul[2] = -(xxx3);
        resul[3] = -(x2y1);
        resul[4] = -(xxx2);
        long double Bb[5], Cc[5], Dd[5], Ee[5], Aa[5];
        Bb[0] = x1y3, Cc[0] = x2y1, Dd[0] = x1y2, Ee[0] = x1y1, Aa[0] = x2y2;
        Bb[1] = yyy4, Cc[1] = x1y2, Dd[1] = yyy3, Ee[1] = yyy2, Aa[1] = x1y3;
        Bb[2] = x1y2, Cc[2] = xxx2, Dd[2] = x1y1, Ee[2] = xxx1, Aa[2] = x2y1;
        Bb[3] = yyy3, Cc[3] = x1y1, Dd[3] = yyy2, Ee[3] = yyy1, Aa[3] = x1y2;
        Bb[4] = yyy2, Cc[4] = xxx1, Dd[4] = yyy1, Ee[4] = N, Aa[4] = x1y1;

        vector<vector<double>>Ma(5);
        vector<double>Md(5);
        for (int i = 0; i < 5; i++)
        {
            Ma[i].push_back(Aa[i]);
            Ma[i].push_back(Bb[i]);
            Ma[i].push_back(Cc[i]);
            Ma[i].push_back(Dd[i]);
            Ma[i].push_back(Ee[i]);
            Ma[i].push_back(resul[i]);
        }

        RGauss(Ma, Md);
        long double A = Md[0];
        long double B = Md[1];
        long double C = Md[2];
        long double D = Md[3];
        long double E = Md[4];
        double XC = (2 * B * C - A * D) / (A * A - 4 * B);
        double YC = (2 * D - A * C) / (A * A - 4 * B);
        long double fenzi = 2 * (A * C * D - B * C * C - D * D + 4 * E * B - A * A * E);
        long double fenmu = (A * A - 4 * B) * (B - sqrt(A * A + (1 - B) * (1 - B)) + 1);
        long double fenmu2 = (A * A - 4 * B) * (B + sqrt(A * A + (1 - B) * (1 - B)) + 1);
        double XA = sqrt(fabs(fenzi / fenmu));
        double XB = sqrt(fabs(fenzi / fenmu2));
        double Xtheta = 0.5 * atan(A / (1 - B)) * 180 / 3.1415926;
        if (B < 1)
            Xtheta += 90;
        vec_result.push_back(XC);
        vec_result.push_back(YC);
        vec_result.push_back(XA);
        vec_result.push_back(XB);
        vec_result.push_back(Xtheta);
        return vec_result;
    }

    void  LSEllipse::cvFitEllipse2f(float* arrayx, float* arrayy, int n, float* box)
    {
        float cx = 0, cy = 0;
        double rp[5], t;
        cv::AutoBuffer<float> _Ad(n * 12 + n);
        float* A1 = _Ad.data();
        float* A3 = A1 + n * 5;
        float* A2 = A3 + n * 5;
        float* B1 = new float[n], * B2 = new float[2], * B3 = new float[n];
        const double min_eps = 1e-6;
        int i;
        for (i = 0; i < n; i++)
        {

            cx += arrayx[i];
            cy += arrayy[i];

        }
        cx /= n;
        cy /= n;
        for (i = 0; i < n; i++)
        {
            int step = i * 5;
            float px, py;
            px = arrayx[i] * 1.0;
            py = arrayy[i] * 1.0;
            px -= cx;
            py -= cy;
            B1[i] = 10000.0;
            A1[step] = -px * px;
            A1[step + 1] = -py * py;
            A1[step + 2] = -px * py;
            A1[step + 3] = px;
            A1[step + 4] = py;
        }
        float* x1 = new float[5];
        //解出Ax^2+By^2+Cxy+Dx+Ey=10000的最小二乘解！
        //cv::SVDecomp()
        SVD(A1, n, 5, B1, x1, min_eps);
        A2[0] = 2 * x1[0], A2[1] = A2[2] = x1[2], A2[3] = 2 * x1[1];
        B2[0] = x1[3], B2[1] = x1[4];
        float* x2 = new float[2];
        //标准化，将一次项消掉，求出center.x和center.y;  
        SVD(A2, 2, 2, B2, x2, min_eps);
        rp[0] = x2[0], rp[1] = x2[1];
        for (i = 0; i < n; i++)
        {
            float px, py;
            px = arrayx[i] - cx;
            py = arrayy[i] - cy;
            B3[i] = 1.0;
            int step = i * 3;
            A3[step] = (px - rp[0]) * (px - rp[0]);
            A3[step + 1] = (py - rp[1]) * (py - rp[1]);
            A3[step + 2] = (px - rp[0]) * (py - rp[1]);

        }
        //求出A(x-center.x)^2+B(y-center.y)^2+C(x-center.x)(y-center.y)的最小二乘解  
        SVD(A3, n, 3, B3, x1, min_eps);

        rp[4] = -0.5 * atan2(x1[2], x1[1] - x1[0]);
        t = sin(-2.0 * rp[4]);
        if (fabs(t) > fabs(x1[2]) * min_eps)
            t = x1[2] / t;
        else
            t = x1[1] - x1[0];
        rp[2] = fabs(x1[0] + x1[1] - t);
        if (rp[2] > min_eps)
            rp[2] = sqrt(2.0 / rp[2]);
        rp[3] = fabs(x1[0] + x1[1] + t);
        if (rp[3] > min_eps)
            rp[3] = sqrt(2.0 / rp[3]);

        box[0] = (float)rp[0] + cx;
        box[1] = (float)rp[1] + cy;
        box[2] = (float)(rp[2] * 2);
        box[3] = (float)(rp[3] * 2);
        if (box[2] > box[3])
        {
            std::swap(box[2], box[3]);
        }
        box[4] = (float)(90 + rp[4] * 180 / 3.1415926);
        if (box[4] < -180)
            box[4] += 360;
        if (box[4] > 360)
            box[4] -= 360;
        delete[]B1;
        delete[]B2;
        delete[]B3;
        delete[]x1;
        delete[]x2;

    }

    int LSEllipse::SVD(float* a, int m, int n, float b[], float x[], float esp)
    {
        float* aa;
        float* u;
        float* v;
        aa = new float[n * m];
        u = new  float[m * m];
        v = new  float[n * n];

        int ka;
        int  flag;
        if (m > n)
        {
            ka = m + 1;
        }
        else
        {
            ka = n + 1;
        }

        flag = gmiv(a, m, n, b, x, aa, esp, u, v, ka);

        delete[]aa;
        delete[]u;
        delete[]v;

        return(flag);
    }

    int LSEllipse::gmiv(float a[], int m, int n, float b[], float x[], float aa[], float eps, float u[], float v[], int ka)
    {
        int i, j;
        i = ginv(a, m, n, aa, eps, u, v, ka);

        if (i < 0) return(-1);
        for (i = 0; i <= n - 1; i++)
        {
            x[i] = 0.0;
            for (j = 0; j <= m - 1; j++)
                x[i] = x[i] + aa[i * m + j] * b[j];
        }
        return(1);
    }

    int LSEllipse::ginv(float a[], int m, int n, float aa[], float eps, float u[], float v[], int ka)
    {

        //  int muav(float a[],int m,int n,float u[],float v[],float eps,int ka);  

        int i, j, k, l, t, p, q, f;
        i = muav(a, m, n, u, v, eps, ka);
        if (i < 0) return(-1);
        j = n;
        if (m < n) j = m;
        j = j - 1;
        k = 0;
        while ((k <= j) && (a[k * n + k] != 0.0)) k = k + 1;
        k = k - 1;
        for (i = 0; i <= n - 1; i++)
            for (j = 0; j <= m - 1; j++)
            {
                t = i * m + j; aa[t] = 0.0;
                for (l = 0; l <= k; l++)
                {
                    f = l * n + i; p = j * m + l; q = l * n + l;
                    aa[t] = aa[t] + v[f] * u[p] / a[q];
                }
            }
        return(1);
    }
}

static void ppp(float a[], float e[], float s[], float v[], int m, int n)
{
    int i, j, p, q;
    float d;
    if (m >= n) i = n;
    else i = m;
    for (j = 1; j <= i - 1; j++)
    {
        a[(j - 1) * n + j - 1] = s[j - 1];
        a[(j - 1) * n + j] = e[j - 1];
    }
    a[(i - 1) * n + i - 1] = s[i - 1];
    if (m < n) a[(i - 1) * n + i] = e[i - 1];
    for (i = 1; i <= n - 1; i++)
        for (j = i + 1; j <= n; j++)
        {
            p = (i - 1) * n + j - 1; q = (j - 1) * n + i - 1;
            d = v[p]; v[p] = v[q]; v[q] = d;
        }
    return;
}

static void sss(float fg[], float cs[])
{
    float r, d;
    if ((fabs(fg[0]) + fabs(fg[1])) == 0.0)
    {
        cs[0] = 1.0; cs[1] = 0.0; d = 0.0;
    }
    else
    {
        d = (float)sqrt(fg[0] * fg[0] + fg[1] * fg[1]);
        if (fabs(fg[0]) > fabs(fg[1]))
        {
            d = (float)fabs(d);
            if (fg[0] < 0.0) d = -d;
        }
        if (fabs(fg[1]) >= fabs(fg[0]))
        {
            d = (float)fabs(d);
            if (fg[1] < 0.0) d = -d;
        }
        cs[0] = fg[0] / d; cs[1] = fg[1] / d;
    }
    r = 1.0;
    if (fabs(fg[0]) > fabs(fg[1])) r = cs[1];
    else
        if (cs[0] != 0.0) r = 1.0f / cs[0];
    fg[0] = d; fg[1] = r;
    return;
}

namespace my{
    int LSEllipse::muav(float a[], int m, int n, float u[], float v[], float eps, int ka)
    {
        int i, j, k, l, it, ll, kk, ix, iy, mm, nn, iz, m1, ks;
        float d, dd, t, sm, sm1, em1, sk, ek, b, c, shh, fg[2], cs[2];
        float* s, * e, * w;

        s = (float*)malloc(ka * sizeof(float));
        e = (float*)malloc(ka * sizeof(float));
        w = (float*)malloc(ka * sizeof(float));
        it = 60; k = n;
        if (m - 1 < n) k = m - 1;
        l = m;
        if (n - 2 < m) l = n - 2;
        if (l < 0) l = 0;
        ll = k;
        if (l > k) ll = l;
        if (ll >= 1)
        {
            for (kk = 1; kk <= ll; kk++)
            {
                if (kk <= k)
                {
                    d = 0.0;
                    for (i = kk; i <= m; i++)
                    {
                        ix = (i - 1) * n + kk - 1; d = d + a[ix] * a[ix];
                    }
                    s[kk - 1] = (float)sqrt(d);
                    if (s[kk - 1] != 0.0)
                    {
                        ix = (kk - 1) * n + kk - 1;
                        if (a[ix] != 0.0)
                        {
                            s[kk - 1] = (float)fabs(s[kk - 1]);
                            if (a[ix] < 0.0) s[kk - 1] = -s[kk - 1];
                        }
                        for (i = kk; i <= m; i++)
                        {
                            iy = (i - 1) * n + kk - 1;
                            a[iy] = a[iy] / s[kk - 1];
                        }
                        a[ix] = 1.0f + a[ix];
                    }
                    s[kk - 1] = -s[kk - 1];
                }
                if (n >= kk + 1)
                {
                    for (j = kk + 1; j <= n; j++)
                    {
                        if ((kk <= k) && (s[kk - 1] != 0.0))
                        {
                            d = 0.0;
                            for (i = kk; i <= m; i++)
                            {
                                ix = (i - 1) * n + kk - 1;
                                iy = (i - 1) * n + j - 1;
                                d = d + a[ix] * a[iy];
                            }
                            d = -d / a[(kk - 1) * n + kk - 1];
                            for (i = kk; i <= m; i++)
                            {
                                ix = (i - 1) * n + j - 1;
                                iy = (i - 1) * n + kk - 1;
                                a[ix] = a[ix] + d * a[iy];
                            }
                        }
                        e[j - 1] = a[(kk - 1) * n + j - 1];
                    }
                }
                if (kk <= k)
                {
                    for (i = kk; i <= m; i++)
                    {
                        ix = (i - 1) * m + kk - 1; iy = (i - 1) * n + kk - 1;
                        u[ix] = a[iy];
                    }
                }
                if (kk <= l)
                {
                    d = 0.0;
                    for (i = kk + 1; i <= n; i++)
                        d = d + e[i - 1] * e[i - 1];
                    e[kk - 1] = (float)sqrt(d);
                    if (e[kk - 1] != 0.0)
                    {
                        if (e[kk] != 0.0)
                        {
                            e[kk - 1] = (float)fabs(e[kk - 1]);
                            if (e[kk] < 0.0) e[kk - 1] = -e[kk - 1];
                        }
                        for (i = kk + 1; i <= n; i++)
                            e[i - 1] = e[i - 1] / e[kk - 1];
                        e[kk] = 1.0f + e[kk];
                    }
                    e[kk - 1] = -e[kk - 1];
                    if ((kk + 1 <= m) && (e[kk - 1] != 0.0))
                    {
                        for (i = kk + 1; i <= m; i++) w[i - 1] = 0.0;
                        for (j = kk + 1; j <= n; j++)
                            for (i = kk + 1; i <= m; i++)
                                w[i - 1] = w[i - 1] + e[j - 1] * a[(i - 1) * n + j - 1];
                        for (j = kk + 1; j <= n; j++)
                            for (i = kk + 1; i <= m; i++)
                            {
                                ix = (i - 1) * n + j - 1;
                                a[ix] = a[ix] - w[i - 1] * e[j - 1] / e[kk];
                            }
                    }
                    for (i = kk + 1; i <= n; i++)
                        v[(i - 1) * n + kk - 1] = e[i - 1];
                }
            }
        }
        mm = n;
        if (m + 1 < n) mm = m + 1;
        if (k < n) s[k] = a[k * n + k];
        if (m < mm) s[mm - 1] = 0.0;
        if (l + 1 < mm) e[l] = a[l * n + mm - 1];
        e[mm - 1] = 0.0;
        nn = m;
        if (m > n) nn = n;
        if (nn >= k + 1)
        {
            for (j = k + 1; j <= nn; j++)
            {
                for (i = 1; i <= m; i++)
                    u[(i - 1) * m + j - 1] = 0.0;
                u[(j - 1) * m + j - 1] = 1.0;
            }
        }
        if (k >= 1)
        {
            for (ll = 1; ll <= k; ll++)
            {
                kk = k - ll + 1; iz = (kk - 1) * m + kk - 1;
                if (s[kk - 1] != 0.0)
                {
                    if (nn >= kk + 1)
                        for (j = kk + 1; j <= nn; j++)
                        {
                            d = 0.0;
                            for (i = kk; i <= m; i++)
                            {
                                ix = (i - 1) * m + kk - 1;
                                iy = (i - 1) * m + j - 1;
                                d = d + u[ix] * u[iy] / u[iz];
                            }
                            d = -d;
                            for (i = kk; i <= m; i++)
                            {
                                ix = (i - 1) * m + j - 1;
                                iy = (i - 1) * m + kk - 1;
                                u[ix] = u[ix] + d * u[iy];
                            }
                        }
                    for (i = kk; i <= m; i++)
                    {
                        ix = (i - 1) * m + kk - 1; u[ix] = -u[ix];
                    }
                    u[iz] = 1.0f + u[iz];
                    if (kk - 1 >= 1)
                        for (i = 1; i <= kk - 1; i++)
                            u[(i - 1) * m + kk - 1] = 0.0;
                }
                else
                {
                    for (i = 1; i <= m; i++)
                        u[(i - 1) * m + kk - 1] = 0.0;
                    u[(kk - 1) * m + kk - 1] = 1.0;
                }
            }
        }
        for (ll = 1; ll <= n; ll++)
        {
            kk = n - ll + 1; iz = kk * n + kk - 1;
            if ((kk <= l) && (e[kk - 1] != 0.0))
            {
                for (j = kk + 1; j <= n; j++)
                {
                    d = 0.0;
                    for (i = kk + 1; i <= n; i++)
                    {
                        ix = (i - 1) * n + kk - 1; iy = (i - 1) * n + j - 1;
                        d = d + v[ix] * v[iy] / v[iz];
                    }
                    d = -d;
                    for (i = kk + 1; i <= n; i++)
                    {
                        ix = (i - 1) * n + j - 1; iy = (i - 1) * n + kk - 1;
                        v[ix] = v[ix] + d * v[iy];
                    }
                }
            }
            for (i = 1; i <= n; i++)
                v[(i - 1) * n + kk - 1] = 0.0;
            v[iz - n] = 1.0;
        }
        for (i = 1; i <= m; i++)
            for (j = 1; j <= n; j++)
                a[(i - 1) * n + j - 1] = 0.0;
        m1 = mm; it = 60;
        while (1 == 1)
        {
            if (mm == 0)
            {
                ppp(a, e, s, v, m, n);
                free(s); free(e); free(w); return(1);
            }
            if (it == 0)
            {
                ppp(a, e, s, v, m, n);
                free(s); free(e); free(w); return(-1);
            }
            kk = mm - 1;
            while ((kk != 0) && (fabs(e[kk - 1]) != 0.0))
            {
                d = (float)(fabs(s[kk - 1]) + fabs(s[kk]));
                dd = (float)fabs(e[kk - 1]);
                if (dd > eps * d) kk = kk - 1;
                else e[kk - 1] = 0.0;
            }
            if (kk == mm - 1)
            {
                kk = kk + 1;
                if (s[kk - 1] < 0.0)
                {
                    s[kk - 1] = -s[kk - 1];
                    for (i = 1; i <= n; i++)
                    {
                        ix = (i - 1) * n + kk - 1; v[ix] = -v[ix];
                    }
                }
                while ((kk != m1) && (s[kk - 1] < s[kk]))
                {
                    d = s[kk - 1]; s[kk - 1] = s[kk]; s[kk] = d;
                    if (kk < n)
                        for (i = 1; i <= n; i++)
                        {
                            ix = (i - 1) * n + kk - 1; iy = (i - 1) * n + kk;
                            d = v[ix]; v[ix] = v[iy]; v[iy] = d;
                        }
                    if (kk < m)
                        for (i = 1; i <= m; i++)
                        {
                            ix = (i - 1) * m + kk - 1; iy = (i - 1) * m + kk;
                            d = u[ix]; u[ix] = u[iy]; u[iy] = d;
                        }
                    kk = kk + 1;
                }
                it = 60;
                mm = mm - 1;
            }
            else
            {
                ks = mm;
                while ((ks > kk) && (fabs(s[ks - 1]) != 0.0))
                {
                    d = 0.0;
                    if (ks != mm) d = d + (float)fabs(e[ks - 1]);
                    if (ks != kk + 1) d = d + (float)fabs(e[ks - 2]);
                    dd = (float)fabs(s[ks - 1]);
                    if (dd > eps * d) ks = ks - 1;
                    else s[ks - 1] = 0.0;
                }
                if (ks == kk)
                {
                    kk = kk + 1;
                    d = (float)fabs(s[mm - 1]);
                    t = (float)fabs(s[mm - 2]);
                    if (t > d) d = t;
                    t = (float)fabs(e[mm - 2]);
                    if (t > d) d = t;
                    t = (float)fabs(s[kk - 1]);
                    if (t > d) d = t;
                    t = (float)fabs(e[kk - 1]);
                    if (t > d) d = t;
                    sm = s[mm - 1] / d; sm1 = s[mm - 2] / d;
                    em1 = e[mm - 2] / d;
                    sk = s[kk - 1] / d; ek = e[kk - 1] / d;
                    b = ((sm1 + sm) * (sm1 - sm) + em1 * em1) / 2.0f;
                    c = sm * em1; c = c * c; shh = 0.0;
                    if ((b != 0.0) || (c != 0.0))
                    {
                        shh = (float)sqrt(b * b + c);
                        if (b < 0.0) shh = -shh;
                        shh = c / (b + shh);
                    }
                    fg[0] = (sk + sm) * (sk - sm) - shh;
                    fg[1] = sk * ek;
                    for (i = kk; i <= mm - 1; i++)
                    {
                        sss(fg, cs);
                        if (i != kk) e[i - 2] = fg[0];
                        fg[0] = cs[0] * s[i - 1] + cs[1] * e[i - 1];
                        e[i - 1] = cs[0] * e[i - 1] - cs[1] * s[i - 1];
                        fg[1] = cs[1] * s[i];
                        s[i] = cs[0] * s[i];
                        if ((cs[0] != 1.0) || (cs[1] != 0.0))
                            for (j = 1; j <= n; j++)
                            {
                                ix = (j - 1) * n + i - 1;
                                iy = (j - 1) * n + i;
                                d = cs[0] * v[ix] + cs[1] * v[iy];
                                v[iy] = -cs[1] * v[ix] + cs[0] * v[iy];
                                v[ix] = d;
                            }
                        sss(fg, cs);
                        s[i - 1] = fg[0];
                        fg[0] = cs[0] * e[i - 1] + cs[1] * s[i];
                        s[i] = -cs[1] * e[i - 1] + cs[0] * s[i];
                        fg[1] = cs[1] * e[i];
                        e[i] = cs[0] * e[i];
                        if (i < m)
                            if ((cs[0] != 1.0) || (cs[1] != 0.0))
                                for (j = 1; j <= m; j++)
                                {
                                    ix = (j - 1) * m + i - 1;
                                    iy = (j - 1) * m + i;
                                    d = cs[0] * u[ix] + cs[1] * u[iy];
                                    u[iy] = -cs[1] * u[ix] + cs[0] * u[iy];
                                    u[ix] = d;
                                }
                    }
                    e[mm - 2] = fg[0];
                    it = it - 1;
                }
                else
                {
                    if (ks == mm)
                    {
                        kk = kk + 1;
                        fg[1] = e[mm - 2]; e[mm - 2] = 0.0;
                        for (ll = kk; ll <= mm - 1; ll++)
                        {
                            i = mm + kk - ll - 1;
                            fg[0] = s[i - 1];
                            sss(fg, cs);
                            s[i - 1] = fg[0];
                            if (i != kk)
                            {
                                fg[1] = -cs[1] * e[i - 2];
                                e[i - 2] = cs[0] * e[i - 2];
                            }
                            if ((cs[0] != 1.0) || (cs[1] != 0.0))
                                for (j = 1; j <= n; j++)
                                {
                                    ix = (j - 1) * n + i - 1;
                                    iy = (j - 1) * n + mm - 1;
                                    d = cs[0] * v[ix] + cs[1] * v[iy];
                                    v[iy] = -cs[1] * v[ix] + cs[0] * v[iy];
                                    v[ix] = d;
                                }
                        }
                    }
                    else
                    {
                        kk = ks + 1;
                        fg[1] = e[kk - 2];
                        e[kk - 2] = 0.0;
                        for (i = kk; i <= mm; i++)
                        {
                            fg[0] = s[i - 1];
                            sss(fg, cs);
                            s[i - 1] = fg[0];
                            fg[1] = -cs[1] * e[i - 1];
                            e[i - 1] = cs[0] * e[i - 1];
                            if ((cs[0] != 1.0) || (cs[1] != 0.0))
                                for (j = 1; j <= m; j++)
                                {
                                    ix = (j - 1) * m + i - 1;
                                    iy = (j - 1) * m + kk - 2;
                                    d = cs[0] * u[ix] + cs[1] * u[iy];
                                    u[iy] = -cs[1] * u[ix] + cs[0] * u[iy];
                                    u[ix] = d;
                                }
                        }
                    }
                }
            }
        }

        free(s); free(e); free(w);
        return(1);
    }

    auto EliipsePara::getEliipsePara(cv::RotatedRect& ellipse) -> EliipsePara
    {
        EliipsePara params;
        params.theta = ellipse.angle * CV_PI / 180.0;
        float a = ellipse.size.width / 2.0;
        float b = ellipse.size.height / 2.0;
        params.c.x = ellipse.center.x;
        params.c.y = ellipse.center.y;
        params.A = a * a * sin(params.theta) + b * b * cos(params.theta) * cos(params.theta);
        params.B = b * b * sin(params.theta) + a * a * cos(params.theta) * cos(params.theta);
        params.C = -2.0 * (a * a - b * b) * sin(params.theta) * cos(params.theta);
        params.D = -2.0 * params.A * ellipse.center.x - params.C * ellipse.center.y;
        params.E= -params.C * ellipse.center.x-2* params.B * ellipse.center.y;
        return params;
    }
}

#ifdef EIGEN_MAJOR_VERSION
//计算初始A,B,C,D,E,F
void Ellispefitting_error_adjustment::Cal_Y0(vector<P2d> input)
{
    double cx = 0, cy = 0, a = 0, b = 0, minx = input[0].x, maxx = input[0].x, miny = input[0].y, maxy = input[0].y;
    for (int i = 0; i < input.size(); i++)
    {
        cx += input[i].x / input.size();
        cy += input[i].y / input.size();
        if (input[i].x < minx)minx = input[i].x;
        if (input[i].x > maxx)maxx = input[i].x;
        if (input[i].y < miny)miny = input[i].y;
        if (input[i].y > maxy)maxy = input[i].y;
    }
    if ((maxx - minx) <= (maxy - miny))
    {
        a = (maxx - minx) / 2;
        b = (maxx - minx) / 2;
    }
    else if ((maxx - minx) > (maxy - miny))
    {
        a = (maxy - miny) / 2;
        b = (maxy - miny) / 2;
    }
    Y0(0) = 0;
    Y0(1) = a * a / (b * b);
    Y0(2) = -2 * cx;
    Y0(3) = (-2 * a * a * cy) / (b * b);
    Y0(4) = cx * cx + (a * a * cy * cy) * (b * b) - a * a;

}
//椭圆拟合总函数，包含椭圆拟合流程
bool Ellispefitting_error_adjustment::fitting(vector<P2d> input, Parameter_ell& Final_ellispe_par)
{
    double A, B, C, D, E, F;
    pointsize = input.size();
    //Cal_Y0(input);//计算初始A,B,C,D,E,F
    SetX(input);
    SetY0(input);
    Cal_LSadj(input, Final_ellispe_par);//最小二乘法平差
    vector<double>adf;
    for (int i = 0; i < 5; i++)adf.push_back(Dt_Y(i));
    A = Y0(0);
    B = Y0(1);
    C = Y0(2);
    D = Y0(3);
    E = Y0(4);


    Parameter_ell tra1;
    tra1.xc = (2 * B * C - A * D) / (A * A - 4 * B);
    tra1.yc = (2 * D - A * C) / (A * A - 4 * B);
    double qwd = 2 * (A * C * D - B * C * C - D * D + 4 * B * E - A * A * E);
    double asd = (A * A - 4 * B) * (B + 1 - sqrt(A * A + (1 - B) * (1 - B)));
    double wifb = (A * A - 4 * B) * (B + 1 + sqrt(A * A + (1 - B) * (1 - B)));
    tra1.ae = sqrt(abs(2 * (A * C * D - B * C * C - D * D + 4 * B * E - A * A * E)
        / ((A * A - 4 * B) * (B + 1 - sqrt(A * A + (1 - B) * (1 - B))))));
    tra1.be = sqrt(abs(2 * (A * C * D - B * C * C - D * D + 4 * B * E - A * A * E)
        / ((A * A - 4 * B) * (B + 1 + sqrt(A * A + (1 - B) * (1 - B))))));
    tra1.angle = -0.5 * atan(A / (B - 1));

    Final_ellispe_par = tra1;
    return true;
}
//最小二乘法平差，迭代的求解方式
void Ellispefitting_error_adjustment::Cal_LSadj(vector<P2d> input, Parameter_ell& Final_ellispe_par)
{

    MatrixXd L0 = MatrixXd::Zero(pointsize, 1);//初始L阵

    int kk = 0;
    //for (int i = 0; i < 6; i++)Y0(i) = 0;
    while (true)
    {	//MatrixXd L = MatrixXd::Zero(pointsize, 1);//L阵
        L0 = X * (Y0);
        for (int i = 0; i < input.size(); i++)
        {
            L0(i) = -input[i].x * input[i].x - L0(i);
        }
        Dt_Y = (X.transpose() * X).inverse() * X.transpose() * L0;//平差求解公式
        Y0 = Y0 + Dt_Y;

        MatrixXd df = X * Y0;
       // vector<double> vd;
        for (int i = 0; i < input.size(); i++)
        {
           // vd.push_back(input[i].x * input[i].x + df(i));
        }

        double A, B, C, D, E;
        A = Y0(0);
        B = Y0(1);
        C = Y0(2);
        D = Y0(3);
        E = Y0(4);

        vector<double>adc, wds;
        for (int i = 0; i < 5; i++)
        {
            adc.push_back(Y0(i));
            wds.push_back(Dt_Y(i));
        }
        kk++;
        if (abs(Dt_Y(0)) < 1e-6 && abs(Dt_Y(1)) < 1e-6 && abs(Dt_Y(2)) < 1e-6 && abs(Dt_Y(3)) < 1e-6 || kk > 100)break;
    }
}

void Ellispefitting_error_adjustment::SetX(vector<P2d> input)
{
    double xi = 0, yi = 0;
    X.resize(pointsize, 5);
    for (int i = 0; i < input.size(); i++)
    {
        xi = input[i].x;
        yi = input[i].y;
        X(i, 0) = xi * yi;
        X(i, 1) = yi * yi;
        X(i, 2) = xi;
        X(i, 3) = yi;
        X(i, 4) = 1;
    }
}

void Ellispefitting_error_adjustment::SetY0(vector<P2d> input)
{
    double cx = 0, cy = 0, a = 0, b = 0, minx = input[0].x, maxx = input[0].x, miny = input[0].y, maxy = input[0].y;
    for (int i = 0; i < input.size(); i++)
    {
        cx += input[i].x / input.size();
        cy += input[i].y / input.size();
        if (input[i].x < minx)minx = input[i].x;
        if (input[i].x > maxx)maxx = input[i].x;
        if (input[i].y < miny)miny = input[i].y;
        if (input[i].y > maxy)maxy = input[i].y;
    }
    if ((maxx - minx) <= (maxy - miny))
    {
        a = (maxx - minx) / 2;
        b = (maxx - minx) / 2;
    }
    else if ((maxx - minx) > (maxy - miny))
    {
        a = (maxy - miny) / 2;
        b = (maxy - miny) / 2;
    }
    Y0(0) = 0;
    Y0(1) = a * a / (b * b);
    Y0(2) = -2 * cx;
    Y0(3) = (-2 * a * a * cy) / (b * b);
    Y0(4) = cx * cx + (a * a * cy * cy) * (b * b) - a * a;
}
#endif
