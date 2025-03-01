#pragma once
#ifndef MVG_H
#define MVG_H


#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <set>
#include <numeric>
#include <algorithm>
#include <thread>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/imgproc.hpp>

#include "frame.h"
#include "slamUtils.h"
#include "slamParams.h"

class MVG
{
public:
	MVG()
	{
        m_K = cv::Mat::eye(3, 3, CV_32F);
		cv::Mat mRInitial= cv::Mat::eye(3, 3, CV_32F);
		cv::Mat mtInitial= cv::Mat::eye(3, 1, CV_32F);
		cv::Mat mH = cv::Mat::eye(3, 1, CV_32F);
		cv::Mat mF = cv::Mat::eye(3, 1, CV_32F);
	}
	MVG(SlamParams* slamParams);

	bool initializePoseAndMap(
		Frame& currentFrame,
		Frame& previousFrame,
		std::vector<cv::Point3f>& p3D,
		std::vector<PtPair>& prunedMatches);

public:
	void updateParams(SlamParams* slamParams);

private:

	void estimateHomography(
		cv::Mat& H,
		const std::vector<cv::Point2f>& pts1,
		const std::vector<cv::Point2f>& pts2,
		const std::vector<PtPair>& mvMatches12,
		const std::vector<std::vector<size_t> >& indexSets,
		float& hScore,
		std::vector<bool>& inliersH);

	void estimateFundamental(
		cv::Mat& M,
		const std::vector<cv::Point2f>& pts1,
		const std::vector<cv::Point2f>& pts2,
		const std::vector<PtPair>& mvMatches12,
		const std::vector<std::vector<size_t> >& indexSets,
		float& fScore,
		std::vector<bool>& inliersF);
private:
	static void normalizePts(const std::vector<cv::Point2f>& inPts, std::vector<cv::Point2f>& outPts, cv::Mat& T);	
	static inline void computeHomography (cv::Mat& H, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2);
	static inline void computeFundamental(cv::Mat& F, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2);
	static void decomposeHomography(const cv::Mat& H, const cv::Mat&  K, std::vector<cv::Mat>& Rs, std::vector<cv::Mat>& Ts);
	static void decomposeFundamental(const cv::Mat& F, const cv::Mat& K, cv::Mat& R1, cv::Mat& R2,cv::Mat& t);
	bool reconstructFromHomography(
		const cv::Mat& H, 
		const std::vector<cv::Point2f>& pts1,
		const std::vector<cv::Point2f>& pts2,
		cv::Mat& R,
		cv::Mat& t,
		std::vector<cv::Point3f>& p3D,
		std::vector<PtPair>& prunedMatches);

	bool reconstructFromFundamental(
		const cv::Mat& F, 
		const std::vector<cv::Point2f>& pts1,
		const std::vector<cv::Point2f>& pts2,
		cv::Mat& R,
		cv::Mat& t,
		std::vector<cv::Point3f>& p3D,
		std::vector<PtPair>& prunedMatches);

	bool chooseRTSolution(
		const std::vector<cv::Mat>& Rs,
		const std::vector<cv::Mat>& Ts,
		const std::vector<cv::Point2f>& pts1,
		const std::vector<cv::Point2f>& pts2,
		cv::Mat& R,
		cv::Mat& t,
		std::vector<cv::Point3f>& p3D,
		std::vector<PtPair>& prunedMatches);
		
		inline static void triangulate(const cv::Mat& P1, const cv::Mat& P2, const cv::Point2f& pts1, const cv::Point2f& pts2, cv::Mat& p3D);
		inline static void triangulateNpts(const cv::Mat& K, const std::vector<cv::Mat>& vPs, const std::vector<cv::Point2f>& vPts, cv::Mat& p3D);


private:
	SlamParams* m_slamParams{ NULL };

	std::vector<bool> m_inliersModel;
	
	cv::Mat m_K;

	std::vector<PtPair> m_kptsMatches;
	
	//TODO: change from cv::Point2f to KeyPoint
	std::vector<cv::Point2f> m_kpts1;
	std::vector<cv::Point2f> m_kpts2;

private:
	//SlamParams
	float	m_minParallax{1 };
	float	m_minPassRatio{0.5f };
	float	m_reprojectionThreshold{4.0f };
	int		m_minTriangulations{50 };
	float	m_maxSecondPassRatio{0.8f };
	int     m_samples{8 };
	int		m_maxTrials{200 };

};

class MVGError
{
public:
	inline static float reprojectionError(const cv::Mat& T, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const std::vector<PtPair>& mvMatches12, std::vector<bool>& inliers);
	inline static float epipolarLineError(const cv::Mat& T, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const std::vector<PtPair>& mvMatches12, std::vector<bool>& inliers);
	inline static float sampsonError(const cv::Mat& T, const std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2, std::vector<bool>& inliers);
	inline static size_t updateTrials(const size_t inlierCount, const size_t nOfPoints, const size_t numberOfSamples, const float confidence);
};

class MVGUtils
{
public:

	MVGUtils() {}

	static bool generateRandomSets(size_t nElements, size_t indexRange, std::vector<std::vector<size_t>>& sets, unsigned long int nSets);
	static void generateRandomIndexes(size_t start, size_t end, std::set<size_t>& indexes, size_t numberOfIndexes);
    static void cvTovector(const cv::Mat& mat, std::vector<std::vector<float> >& vector);
private:
	
};
#endif
