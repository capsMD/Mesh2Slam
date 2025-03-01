//
// Created by caps80 on 14.10.23.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <vector>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>


#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include "g2o/core/robust_kernel_impl.h"
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/sim3.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

#include "mvg.h"
#include "frame.h"
#include "mapPoint.h"


static inline cv::Mat toCVMat(const g2o::SE3Quat &SE3);
static inline cv::Mat toCVMat(const Eigen::Matrix<double,4,4> &m);
static inline cv::Mat toCVMat(const Eigen::Matrix3d &m);
static inline cv::Mat toCVMat(const Eigen::Matrix<double,3,1> &m);
static inline cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);
static inline g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
static inline g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

static inline Eigen::Matrix<double,3,1> toEigenVector3D(const cv::Mat &pt)
{
    Eigen::Matrix<double,3,1> v(pt.at<float>(0),pt.at<float>(1),pt.at<float>(2));
    return v;
}

static inline cv::Mat toCVPos(const Eigen::Matrix<double,3,1>& pos)
{
    cv::Mat returnMat = cv::Mat(3,1,CV_32F);
    for(size_t i = 0; i < 3;i++)
        for(size_t j = 0; j < 1;j++)
            returnMat.at<float>(i,j)=pos(i,j);
    std::vector<std::vector<float> > tMat;
    MVGUtils::cvTovector(returnMat,tMat);
    return returnMat;
}

static inline Eigen::Matrix<double,4,4> toEigenPose(const cv::Mat& pose)
{

    Eigen::Matrix<double,4,4> M;
    M(0,0) = pose.at<float>(0,0);
    M(1,0) = pose.at<float>(1,0);
    M(2,0) = pose.at<float>(2,0);
    M(3,0) = pose.at<float>(3,0);

    M(0,1) = pose.at<float>(0,1);
    M(1,1) = pose.at<float>(1,1);
    M(2,1) = pose.at<float>(2,1);
    M(3,1) = pose.at<float>(3,1);

    M(0,2) = pose.at<float>(0,2);
    M(1,2) = pose.at<float>(1,2);
    M(2,2) = pose.at<float>(2,2);
    M(3,2) = pose.at<float>(3,2);

    M(0,3) = pose.at<float>(0,3);
    M(1,3) = pose.at<float>(1,3);
    M(2,3) = pose.at<float>(2,3);
    M(3,3) = pose.at<float>(3,3);

    return M;
}

class Optimizer
{
public:
    static void globalBundleAdjustment(std::shared_ptr< Map> pMap, const int iterations);
    static void bundleAdjustN(const std::vector<Frame*>& vpFrames, const std::vector<MapPoint*>& vpMapPoints, const bool isSetRobust, const int iterations);
    static void bundleAdjust(std::shared_ptr<Map> pMap, Frame* newFrame, bool* stop, SlamParams* slamParams);
    static int pnpOnFrame(Frame& frame, const int iterations);
private:
};
#endif //OPTIMIZER_H
