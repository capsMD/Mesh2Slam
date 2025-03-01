#include "mvg.h"

MVG::MVG(SlamParams* slamParams) : m_slamParams(slamParams)
{
}

bool MVG::initializePoseAndMap
(
	Frame& currentFrame,
	Frame& previousFrame,
	std::vector<cv::Point3f>& p3D,
	std::vector<PtPair>& prunedMatches)
{

	std::vector<bool> inliersH;
	std::vector<bool> inliersF;

	cv::Mat H = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat F = cv::Mat::eye(3, 3, CV_32F);

	float hScore = 0.0f;
	float fScore = 0.0f;

    auto& pKPts1 = previousFrame.getKeyPoints();
    auto& pKPts2 = currentFrame.getKeyPoints();
    m_kptsMatches = currentFrame.getFeatureMatches();

    auto n1 = pKPts1.size();
    auto n2 = pKPts2.size();
    auto n21= m_kptsMatches.size();

    m_kpts1.clear();
    m_kpts2.clear();
    m_kpts1 = std::vector<cv::Point2f >(n1, cv::Point2f (0, 0));
    m_kpts2 = std::vector<cv::Point2f >(n2, cv::Point2f (0, 0));


    //convert from keypts to points2f from both frames and use for initialization
    for(size_t i = 0; i<n1;i++)
        m_kpts1[i] = pKPts1[i].pt;
    for(size_t i = 0; i<n2;i++)
        m_kpts2[i] = pKPts2[i].pt;


    std::vector<std::vector<size_t> > indexSets;
    if (!MVGUtils::generateRandomSets(8, n21, indexSets, m_maxTrials))
        return false;

    std::thread tH(&MVG::estimateHomography, this, std::ref(H), std::cref(m_kpts1), std::cref(m_kpts2), std::cref(m_kptsMatches), std::cref(indexSets), std::ref(hScore), std::ref(inliersH));
    std::thread tF(&MVG::estimateFundamental, this, std::ref(F), std::cref(m_kpts1), std::cref(m_kpts2), std::cref(m_kptsMatches), std::cref(indexSets), std::ref(fScore), std::ref(inliersF));

    tH.join();
    tF.join();

    cv::Mat R, t;
    // Compute ratio of scores
    float RH = hScore / (hScore + fScore);

    bool reconstructionOk = false;
    if (RH > 0.40)
    {
        m_inliersModel = inliersH;
        reconstructionOk = MVG::reconstructFromHomography(H, m_kpts1, m_kpts2, R, t, p3D, prunedMatches);
    }
    else //if(pF_HF>0.6)
    {
        m_inliersModel = inliersF;
        reconstructionOk =MVG::reconstructFromFundamental(F, m_kpts1, m_kpts2, R, t, p3D, prunedMatches);
    }


    //first frame is identity (origin)
    previousFrame.setPose(cv::Mat::eye(4,4,CV_32F));
    //current frame is new pose
    currentFrame.setPose(cv::Mat::eye(4,4,CV_32F));


    if(reconstructionOk)
    {
        currentFrame.setR(R);
        currentFrame.sett(t);
        currentFrame.updateTransform();

        return true;
    }

    return false;
}

void MVG::updateParams(SlamParams* slamParams)
{
	m_slamParams = slamParams;
    m_K.at<float>(0, 0) = m_slamParams->camParams.fx;
    m_K.at<float>(1, 1) = m_slamParams->camParams.fy;
    m_K.at<float>(0, 2) = m_slamParams->camParams.cx;
    m_K.at<float>(1, 2) = m_slamParams->camParams.cy;

    m_minParallax			= m_slamParams->errorParams.minParallax;
    m_minPassRatio			= m_slamParams->errorParams.minInlierRatio;
    m_reprojectionThreshold	= m_slamParams->errorParams.reprojectThreshold;
    m_minTriangulations		= m_slamParams->errorParams.minTriangulations;
    m_maxSecondPassRatio		= m_slamParams->errorParams.maxSecondPassRatio;

    m_samples				= m_slamParams->ransacParams.samples;
    m_maxTrials				= m_slamParams->ransacParams.maxTrials;

}

void MVG::estimateHomography(
	cv::Mat& H,
	const std::vector<cv::Point2f>& pts1,
	const std::vector<cv::Point2f>& pts2,
	const std::vector<PtPair>& mvMatches12,
	const std::vector<std::vector<size_t> >& indexSets,
	float& hScore,
	std::vector<bool>& inliersH)
{
	//to be normalized
	std::vector<cv::Point2f> pts1N;
	std::vector<cv::Point2f> pts2N;

	//Normalization Matrices
	cv::Mat T1;
	cv::Mat T2;

	//normalize
	MVG::normalizePts(pts1, pts1N, T1);
	MVG::normalizePts(pts2, pts2N, T2);
	cv::Mat T2inv = T2.inv();

	//points to be sampled for each set
	std::vector<cv::Point2f> samplePts1(m_samples, cv::Point2f(0.0f, 0.0f));
	std::vector<cv::Point2f> samplePts2(m_samples, cv::Point2f(0.0f, 0.0f));

	std::vector<bool> newInliers(mvMatches12.size(), false);
	hScore = -1.0f;

	float newScore = 0.0f;

	cv::Mat Htemp;
	for (size_t i = 0; i < m_maxTrials; i++)
	{
		//fetch pre-randomized set of points
		for (size_t j = 0; j < m_samples; j++)
		{
			const size_t idx = indexSets[i][j];
			samplePts1[j] = pts1N[mvMatches12[idx].first];
			samplePts2[j] = pts2N[mvMatches12[idx].second];
		}

		cv::Mat newH;
		MVG::computeHomography(newH, samplePts1, samplePts2);
		//denormalize	
		Htemp = T2inv * newH * T1;
		//Get score/inlier count based on reprojection
		newScore = MVGError::reprojectionError(Htemp, pts1, pts2, mvMatches12, inliersH);
		//Update Ransac
		if (newScore > hScore)
		{
			H = Htemp.clone();
			hScore = newScore;
			inliersH = newInliers;
		}
	}
}


void MVG::estimateFundamental(
	cv::Mat& F,
	const std::vector<cv::Point2f>& pts1,
	const std::vector<cv::Point2f>& pts2,
	const std::vector<PtPair>& mvMatches12,
	const std::vector<std::vector<size_t> >& indexSets,
	float& fScore,
	std::vector<bool>& inliersF)
{
		//to be normalized
	std::vector<cv::Point2f> pts1N;
	std::vector<cv::Point2f> pts2N;

	//Normalization Matrices
	cv::Mat T1;
	cv::Mat T2;

	//normalize
	MVG::normalizePts(pts1, pts1N, T1);
	MVG::normalizePts(pts2, pts2N, T2);
	cv::Mat T2t = T2.t();

	//points to be sampled for each set
	std::vector<cv::Point2f> samplePts1(m_samples, cv::Point2f(0.0f, 0.0f));
	std::vector<cv::Point2f> samplePts2(m_samples, cv::Point2f(0.0f, 0.0f));

	std::vector<bool> newInliers(mvMatches12.size(), false);
	fScore = -1.0f;

	float newScore = 0.0f;

	cv::Mat Ftemp;
	for (size_t i = 0; i < m_maxTrials; i++)
	{

		//fetch pre-randomized set of points
		for (size_t j = 0; j < m_samples; j++)
		{
			const size_t idx = indexSets[i][j];
			samplePts1[j] = pts1N[mvMatches12[idx].first];
			samplePts2[j] = pts2N[mvMatches12[idx].second];
		}


		//Estimate Homography
		cv::Mat newF;
		MVG::computeFundamental(newF, samplePts1, samplePts2);
		//denormalize	
		Ftemp = T2t * newF * T1;

		//Get score/inlier count based on reprojection
		newScore = MVGError::epipolarLineError(Ftemp, pts1, pts2, mvMatches12, newInliers);

		//Update Ransac
		if (newScore > fScore)
		{
			F = Ftemp.clone();
			fScore = newScore;
            inliersF = newInliers;
		}
	}
}


void MVG::computeHomography(cv::Mat& H, const std::vector<cv::Point2f>& samplePts1, const std::vector<cv::Point2f>& samplePts2)
{
	if (samplePts1.size() != samplePts2.size())
		return;

	const int nPoints = samplePts1.size();

	cv::Mat A(2 * nPoints, 9, CV_32F);

	for (int i = 0; i < nPoints; i++)
	{
		const float p1X = samplePts1[i].x;
		const float p1Y = samplePts1[i].y;

		const float p2X = samplePts2[i].x;
		const float p2Y = samplePts2[i].y;

		A.at<float>(2 * i, 0) = 0.0;
		A.at<float>(2 * i, 1) = 0.0;
		A.at<float>(2 * i, 2) = 0.0;
		A.at<float>(2 * i, 3) = -p1X;
		A.at<float>(2 * i, 4) = -p1Y;
		A.at<float>(2 * i, 5) = -1;
		A.at<float>(2 * i, 6) = p2Y * p1X;
		A.at<float>(2 * i, 7) = p2Y * p1Y;
		A.at<float>(2 * i, 8) = p2Y;


		A.at<float>(2 * i + 1, 0) = -p1X;
		A.at<float>(2 * i + 1, 1) = -p1Y;
		A.at<float>(2 * i + 1, 2) = -1;
		A.at<float>(2 * i + 1, 3) = 0.0;
		A.at<float>(2 * i + 1, 4) = 0.0;
		A.at<float>(2 * i + 1, 5) = 0.0;
		A.at<float>(2 * i + 1, 6) = p2X * p1X;
		A.at<float>(2 * i + 1, 7) = p2X * p1Y;
		A.at<float>(2 * i + 1, 8) = p2X;
	}

	cv::Mat u, w, vt;

	cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	H = vt.row(8).reshape(0, 3);

	H /= H.at<float>(2, 2);

}

void MVG::computeFundamental(cv::Mat& F, const std::vector<cv::Point2f>& samplePts1, const std::vector<cv::Point2f>& samplePts2)
{
	//Does not work for:
	//planar points
	//pure rotation

	if (samplePts1.size() != samplePts2.size())
		return;

	const int N = samplePts1.size();

	cv::Mat A(N, 9, CV_32F);

	for (int i = 0; i < N; i++)
	{
		const float p1x = samplePts1[i].x;
		const float p1y = samplePts1[i].y;
		const float p2x = samplePts2[i].x;
		const float p2y = samplePts2[i].y;

		A.at<float>(i, 0) = p2x * p1x;
		A.at<float>(i, 1) = p2x * p1y;
		A.at<float>(i, 2) = p2x;
		A.at<float>(i, 3) = p2y * p1x;
		A.at<float>(i, 4) = p2y * p1y;
		A.at<float>(i, 5) = p2y;
		A.at<float>(i, 6) = p1x;
		A.at<float>(i, 7) = p1y;
		A.at<float>(i, 8) = 1;
	}

	cv::Mat u, w, vt;

	cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	cv::Mat Fpre = vt.row(8).reshape(0, 3);

	cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	w.at<float>(2) = 0;

	F = u * cv::Mat::diag(w) * vt;
}

void MVG::decomposeHomography(const cv::Mat& H, const cv::Mat& K, std::vector<cv::Mat>& Rs, std::vector<cv::Mat>& Ts)
{
	
	/*
	* //Following Matlab code:
	*	% Compute the 4 possible pairs of R and t from homography matrix

		% Copyright 2018 The MathWorks, Inc.

		% References:
		% -----------
		% [1] O. Faugeras and F. Lustman, �Motion and structure from motion in a
		% piecewise planar environment�, in International Journal of Pattern
		% Recognition and Artificial Intelligence, 2(3):485�508, 1988.
	//there are 4 solutions, need to store each rotation, translation and plane normal

	*/

	std::vector<cv::Mat> vRotations, vTranslations, vNormals;

	cv::Mat invK = K.inv();
	cv::Mat A = invK * H * K;

	cv::Mat U, S, Vt;
	cv::SVD::compute(A, S, U, Vt, cv::SVD::FULL_UV);

	const float s = static_cast<float>(cv::determinant(U) * cv::determinant(Vt));
	cv::Mat V = Vt.t();

	const float d1 = S.at<float>(0);
	const float d2 = S.at<float>(1);
	const float d3 = S.at<float>(2);

	const float epsilon = 0.00001f;

	bool condition1 = (fabs(d1 - d2) < epsilon);
	bool condition2 = (fabs(d2 - d3) < epsilon);

	cv::Mat N;

	const float d1Sqrd = d1 * d1;
	const float d2Sqrd = d2 * d2;
	const float d3Sqrd = d3 * d3;

	const float aux1 = sqrtf((d1Sqrd - d2Sqrd) / (d1Sqrd - d3Sqrd));
	const float aux3 = sqrtf((d2Sqrd - d3Sqrd) / (d1Sqrd - d3Sqrd));

	std::vector<float> x1 = { aux1, aux1, -aux1, -aux1 };
	std::vector<float> x3 = { aux3, -aux3, aux3, -aux3 };

	const float auxSinTheta = sqrtf((d1Sqrd - d2Sqrd) * (d2Sqrd - d3Sqrd)) / ((d1 + d3) * d2);
	const float cosTheta = sqrtf((d2Sqrd + (d1 * d3)) / ((d1 + d3) * d2));
	std::vector<float> sinTheta = { auxSinTheta, -auxSinTheta, -auxSinTheta, auxSinTheta };


    if ((!condition1) && (!condition2))
    {
		for (char i = 0; i < 4; i++)
		{
			//Rotation matrix
			cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
			Rp.at<float>(0, 0) = cosTheta;
			Rp.at<float>(0, 2) = -sinTheta[i];
			Rp.at<float>(2, 0) = sinTheta[i];
			Rp.at<float>(2, 2) = cosTheta;

			cv::Mat R = s * U * Rp * Vt;
			vRotations.push_back(R);

			//Translation vector
			cv::Mat Tp(3, 1, CV_32F);
			Tp.at<float>(0) = x1[i] * (d1 - d3);
			Tp.at<float>(1) = 0;
			Tp.at<float>(2) = -x3[i] * (d1 - d3);
			cv::Mat t = U * Tp;
			vTranslations.push_back(t / cv::norm(t));

			//Plane normal
			cv::Mat Np(3, 1, CV_32F);
			Np.at<float>(0) = x1[i];
			Np.at<float>(1) = 0;
			Np.at<float>(2) = x3[i];

			cv::Mat N = V * Np;
			if (N.at<float>(2) < 0) N = -N;
			vNormals.push_back(N / cv::norm(N));
		}
	}

	else if ((condition1) && (!condition2))
	{
		//Rotation matrix
		cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
		cv::Mat R = s * U * Rp * Vt;

		std::vector<float> x{ 1, -1 };

		for (char i = 0; i < 2; i++)
		{
			vRotations.push_back(R);

			//Plane normal
			cv::Mat Np(3, 1, CV_32F);
			Np.at<float>(0) = 0;
			Np.at<float>(1) = 0;
			Np.at<float>(2) = x[i];

			//Translation vector
			cv::Mat Tp(3, 1, CV_32F);
			Tp.at<float>(0) = 0;
			Tp.at<float>(1) = 0;
			Tp.at<float>(2) = (d3 - d1) * x[i];
			cv::Mat t = U * Tp;
			vTranslations.push_back(t / norm(t));

			cv::Mat n(3, 1, CV_32F);
			n = V * Np;
			vNormals.push_back(n / norm(n));
		}

	}

	else if ((!condition1) && (condition2))
	{
		//Rotation matrix
		cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
		cv::Mat R = s * U * Rp * Vt;

		std::vector<float> x{ 1, -1 };

		for (char i = 0; i < 2; i++)
		{
			vRotations.push_back(R);

			//Plane normal
			cv::Mat Np(3, 1, CV_32F);
			Np.at<float>(0) = x[i];
			Np.at<float>(1) = 0;
			Np.at<float>(2) = 0;

			//Translation vector
			cv::Mat Tp(3, 1, CV_32F);
			Tp.at<float>(0) = (d3 - d1) * x[i];
			Tp.at<float>(1) = 0;
			Tp.at<float>(2) = 0;
			cv::Mat t = U * Tp;
			vTranslations.push_back(t / norm(t));

			cv::Mat n(3, 1, CV_32F);
			n = V * Np;
			vNormals.push_back(n / norm(n));
		}
	}

	//degenerate case
	else if ((condition1) && (condition2))
	{
		//Rotation matrix
		//Rotation matrix
		cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
		cv::Mat R = s * U * Rp * Vt;
		vRotations.push_back(R);
		cv::Mat t = cv::Mat::zeros(3, 1, CV_32F);
		cv::Mat n = cv::Mat::zeros(3, 1, CV_32F);

		vTranslations.push_back(t);
		vRotations.push_back(n);

	}

	Rs = vRotations;
	Ts = vTranslations;
}

void MVG::decomposeFundamental(const cv::Mat& F, const cv::Mat& K, cv::Mat& R1, cv::Mat& R2, cv::Mat& t)
{
	//Must assume cameras are pre-calibrated
	//recover E given F = K2^-T * E * K1^-1 ===> K2^T * F = E * K1^-1 ===> K2^T * F * K1
	cv::Mat E = K.t() * F * K;

    std::vector<std::vector<float> > Kd;
    std::vector<std::vector<float> > Fd;

    MVGUtils::cvTovector(K,Kd);
    MVGUtils::cvTovector(F,Fd);

    //when decomposing E, there are 2 possible tanslations and two possible rotations
	cv::Mat U, S, Vt;
	cv::SVD::compute(E, S, U, Vt);

	//t is third column of U
	U.col(2).copyTo(t);
	t = t / cv::norm(t);


	cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
	W.at<float>(0, 1) = -1;
	W.at<float>(1, 0) = 1;
	W.at<float>(2, 2) = 1;

	R1 = U * W * Vt;
	if (cv::determinant(R1) < 0) R1 = -R1;
	R2 = U * W.t() * Vt;
	if (cv::determinant(R2) < 0) R2 = -R2;




}

bool MVG::reconstructFromHomography(
	const cv::Mat& H,
	const std::vector<cv::Point2f>& pts1,
	const std::vector<cv::Point2f>& pts2,
	cv::Mat& R,
	cv::Mat& t,
	std::vector<cv::Point3f>& p3D,
	std::vector<PtPair>& prunedMatches)
{
	std::vector<cv::Mat> Rs;
	std::vector<cv::Mat> Ts;
	std::vector<cv::Mat> Ns;

	decomposeHomography(H, m_K, Rs, Ts);
	return chooseRTSolution(Rs, Ts, pts1, pts2, R, t, p3D, prunedMatches);
}

bool MVG::reconstructFromFundamental(
	const cv::Mat& F,
	const std::vector<cv::Point2f>& pts1,
	const std::vector<cv::Point2f>& pts2,
	cv::Mat& R,
	cv::Mat& t,
	std::vector<cv::Point3f>& p3D,
	std::vector<PtPair>& prunedMatches)
{
	cv::Mat R1, R2;
	cv::Mat t1, t2;

	decomposeFundamental(F, m_K, R1, R2, t1);
	t2 = -t1;

	std::vector<cv::Mat> Rs{ R1,R2,R1,R2 };
	std::vector<cv::Mat> ts{ t1,t1,t2,t2 };
	bool solutionFound = false;

    solutionFound = chooseRTSolution(Rs, ts, pts1, pts2, R, t, p3D, prunedMatches);


    return solutionFound;
}


bool MVG::chooseRTSolution(
        const std::vector<cv::Mat>& Rs,
        const std::vector<cv::Mat>& Ts,
        const std::vector<cv::Point2f>& pts1,
        const std::vector<cv::Point2f>& pts2,
        cv::Mat& R,
        cv::Mat& t,
        std::vector<cv::Point3f>& p3Ds,
        std::vector<PtPair>& prunedMatches)
{
    size_t modelInliersCount =0;
    for (size_t i = 0; i < m_kptsMatches.size(); i++)
        if (m_inliersModel[i])
            modelInliersCount++;

    const size_t nPoints = m_kptsMatches.size();
    const size_t nSolutions = Rs.size();
	const float minParallax = m_slamParams->errorParams.minParallax;

    //store for every solution
    std::vector<std::vector<float> > vCosParallaxes(nSolutions, std::vector<float>(nPoints, 0.0f));
    std::vector<std::vector<cv::Point3f> > vP3Ds(nSolutions, std::vector<cv::Point3f>(nPoints, cv::Point3f(0.0f, 0.0f, 0.0f)));
    std::vector<std::vector<bool> > valid3DPts(nSolutions, std::vector<bool>(nPoints, false));

    //used for solution evaluation: pPass: passed reprojection and negative depth, pFit enough parallax

    std::vector<float> pParallax(nPoints, 0);
    std::vector<int>score3Dpts(nSolutions, 0);
    std::vector<int>valid3DptsCount(nSolutions, 0);

    const float fx = m_K.at<float>(0, 0);
    const float fy = m_K.at<float>(1, 1);
    const float cx = m_K.at<float>(0, 2);
    const float cy = m_K.at<float>(1, 2);

    //cam1 matrix
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
    //cam2 matrix
    cv::Mat P2(3, 4, CV_32F, cv::Scalar(0));

    bool solutionFound = false;
    int bestScorePts = 0;
    for (size_t j = 0; j < nSolutions; j++)
    {
        int score3DPtsi = 0;
        int valid3DPtsi = 0;

        cv::Mat Rj = Rs[j];
        cv::Mat Tj = Ts[j];

        // Camera 1 Projection Matrix K[I|0]
        cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
        m_K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

        //Camera 1 center in world coord. (in this case origin)
        cv::Mat C1 = cv::Mat::zeros(3, 1, CV_32F);

        //Camera 2 Projection Matrix K[R|t]
        cv::Mat P2(3, 4, CV_32F);
        Rj.copyTo(P2.rowRange(0, 3).colRange(0, 3));
        Tj.copyTo(P2.rowRange(0, 3).col(3));
        P2 = m_K * P2;

        //Camera2 center in world coord.
        cv::Mat C2 = -Rj.t() * Tj;

        if (cv::norm(C1-C2) <= 0.1)
            return false;

        for (size_t i = 0; i < m_kptsMatches.size(); i++)
        {
            if (!m_inliersModel[i])
                continue;

            cv::Point2f pt1(pts1[m_kptsMatches[i].first].x, pts1[m_kptsMatches[i].first].y);
            cv::Point2f pt2(pts2[m_kptsMatches[i].second].x, pts2[m_kptsMatches[i].second].y);


            cv::Mat p3dC1;

            MVG::triangulate(P1, P2, pt1, pt2, p3dC1);
            std::vector<float> p3D{p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2)};

            if (std::isinf(p3dC1.at<float>(0)) || std::isinf(p3dC1.at<float>(1)) || std::isinf(p3dC1.at<float>(2)))
            {
                continue;
            }

            // Check parallax
            cv::Mat p3DCam1 = p3dC1 - C1;
            float p3DCam1Norm = cv::norm(p3DCam1);

            cv::Mat p3DCam2 = p3dC1 - C2;
            float p3DCam2Norm = cv::norm(p3DCam2);

            float cosParallax = p3DCam1.dot(p3DCam2) / (p3DCam1Norm * p3DCam2Norm);

            // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            float p3DDepth = p3dC1.at<float>(2);
            if (p3DDepth <= 0 && cosParallax < minParallax)
                continue;

            // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            cv::Mat p3dC2 = Rj * p3dC1 + Tj;

            if (p3dC2.at<float>(2) <= 0 && cosParallax < minParallax)
                continue;

            // Check reprojection error in first image
            float im1x, im1y;
            float invZ1 = 1.0f / p3dC1.at<float>(2);
            im1x = fx * p3dC1.at<float>(0) * invZ1 + cx;
            im1y = fy * p3dC1.at<float>(1) * invZ1 + cy;

            float squareError1 = (im1x - pt1.x) * (im1x - pt1.x) + (im1y - pt1.y) * (im1y - pt1.y);

            if (squareError1 > 4)
                continue;

            // Check reprojection error in second image
            float im2x, im2y;
            float invZ2 = 1.0f / p3dC2.at<float>(2);
            im2x = fx * p3dC2.at<float>(0) * invZ2 + cx;
            im2y = fy * p3dC2.at<float>(1) * invZ2 + cy;

            float squareError2 = (im2x - pt2.x) * (im2x - pt2.x) + (im2y - pt2.y) * (im2y - pt2.y);

            if (squareError2 > 4)
                continue;

            pParallax[i] = cosParallax;
            vP3Ds[j][i] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
            score3DPtsi++;

            //after all pruning, if point has enough cosParallax it is valid
            //1 is a value for parallel rays
            if (cosParallax < m_minParallax)
            {
                valid3DPtsi++;
                valid3DPts[j][i] = true;
            }
        }
        if (score3DPtsi > bestScorePts)	bestScorePts = score3DPtsi;

        //store score for each solution
        score3Dpts[j] = score3DPtsi;
        valid3DptsCount[j] = valid3DPtsi;
        vCosParallaxes[j] = pParallax;
    }
    //now evaluate each solution
    size_t highScoreSolution = 0;
    for (size_t j = 0; j < nSolutions; j++)
    {
        //ignore these solutions
        float score = static_cast<float>(score3Dpts[j]) / static_cast<float>(modelInliersCount);
        Logger<std::string>::LogInfoI("initialization score: " + std::to_string(score));
        if (score3Dpts[j] <= 0 || score3Dpts[j] < (m_minPassRatio * modelInliersCount))
            continue;
        else
        {
            //reject if more than 1 solution has high score
            if (score3Dpts[j] > m_maxSecondPassRatio * bestScorePts)
                highScoreSolution++;

            if (highScoreSolution > 1)
            {
                Logger<std::string>::LogError("More than one solution found.");
                return false;
            }

            if (valid3DptsCount[j] > m_minTriangulations)
            {
                //sort parallax for good solution
                sort(vCosParallaxes[j].begin(), vCosParallaxes[j].end());
                size_t idx = std::min(50, int(vCosParallaxes[j].size() - 1));
                const float parallax = acosf(vCosParallaxes[j][idx]) * 180.0f / CV_PI;
                if (parallax > m_minParallax)
                {
                    Rs[j].copyTo(R);
                    Ts[j].copyTo(t);
                    p3Ds = vP3Ds[j];
                    prunedMatches = m_kptsMatches;
                    solutionFound = true;


                    for (size_t i = 0; i < valid3DPts[j].size(); i++)
                    {
                        if (!valid3DPts[j][i])
                            prunedMatches[i] = std::make_pair(-1,-1);
                    }
                }
            }
        }
    }
    return solutionFound;
}


void MVG::triangulate(const cv::Mat& P1, const cv::Mat& P2, const cv::Point2f& pts1, const cv::Point2f& pts2, cv::Mat& p3D)
{
	/*
	Each 2d projection point x defined by P * X, where P is projection matrix, X 3d point; (xProjected = P*X)
	cross-product between x and PX must equal zero (same ray-direction, difference by magnitude), xProjected x (P*X) = 0

	|x| |P11,P12,P13,P14| |X|  =>         |x|   |P11,P12,P13,P14| |X| = 0   =>  |x|   |Xx*P11 + Xy*P12 + Xz*P13 + P14| = 0
	|y|=|P21,P22,P23,P24| |Y|			  |y| X |P21,P22,P23,P24| |Y|			|y| X |Xx*P21 + Xy*P22 + Xz*P23 + P24|
	|1| |P31,P32,P33,P34| |Z|			  |1|   |P31,P32,P33,P34| |Z|			|1|   |Xx*P31 + Xy*P32 + Xz*P33 + P34|
						  |1|	                                  |1|


or, having the j-th row of P denoted Pj => |x|   |X*P1| = 0, since cross-product between vectors, take skew-symmetric and multiply:
										   |y| X |X*P2|
										   |1|   |X*P3|


| 0   -1   y| * |X*P1| = 0   =>  -XP2 + yXP3 = 0, 3rd line is linear combination  =>	 XP1 - xXP3 = 0 => |P1 - xP3| * X = 0, or
| 1   0   -x|   |X*P2|			  XP1 - xXP3 = 0  combination of 1st and 2nd			-XP2 + yXP3 = 0    |P2 + yP3|
|-y   x    0|   |X*P3|			 -yXP1 +xXP2 = 0  x*1st +y*2nd


|xP3 - P1| * X = 0,  => x * P3 - P1 = 0  , Ax= B, SVD
|yP3 - P2|				y * P3 - P2 = 0

*/

//initialize constraints matrix
	cv::Mat A(4, 4, CV_32F);

	A.row(0) = pts1.x * P1.row(2) - P1.row(0);
	A.row(1) = pts1.y * P1.row(2) - P1.row(1);
	A.row(2) = pts2.x * P2.row(2) - P2.row(0);
	A.row(3) = pts2.y * P2.row(2) - P2.row(1);

	cv::Mat U, S, Vt;
	cv::SVD::compute(A, S, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	p3D = Vt.row(3).t();
	p3D = p3D.rowRange(0, 3) / p3D.at<float>(3);
}

void MVG::triangulateNpts(const cv::Mat& K, const std::vector<cv::Mat> &vPs, const std::vector<cv::Point2f> &vPts, cv::Mat &p3D)
{
    size_t nRows = vPs.size();

    cv::Mat A(2*nRows,4,CV_32F);

    for(size_t i = 0; i < nRows; i++)
    {
        cv::Mat P = vPs[i].clone();
        P = K * P;
        A.row(i*2) = vPts[i].x * P.row(2) - P.row(0);
        A.row(i*2+1) = vPts[i].y * P.row(2) - P.row(1);
    }

    cv::Mat U, S, Vt;
    cv::SVD::compute(A, S, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    p3D = Vt.row(3).t();
    p3D = p3D.rowRange(0, 3) / p3D.at<float>(3);

}

void MVG::normalizePts(const std::vector<cv::Point2f>& inPts, std::vector<cv::Point2f>& outPts, cv::Mat& T)

{
	/*
	*Using Isotropic scaling
	"This normalizing transformation will nullify the
	effect of the arbitrary selection of origin and scale in the coordinate frame of the image."
	p.104, R. Hartley, A. Zisserman, "Multiple View Geometry in Computer Vision,"

	For each set of points:
	1- bring centroid of points to origin
	2- scale points so that average distance to center is sqrt(2)

	//T = scaleM * translationM

	i.e.
		scaleM		=	[Scale, 0,	0]  translationM =	[1,		0,	-TX]
						[0	,Scale,	0] 					[0,		1,	-TY]
						[0	,0,		1] 					[0,		0,	  1]

	*/

	const size_t nPoints = inPts.size();
	outPts = std::vector<cv::Point2f>(nPoints, cv::Point2f(0.0f, 0.0f));

	//find centroid translation
	float centroidX = 0.0f;
	float centroidY = 0.0f;
	for (size_t i = 0; i < nPoints; i++)
	{
		centroidX += inPts[i].x;
		centroidY += inPts[i].y;
	}
	centroidX /= nPoints;
	centroidY /= nPoints;

	//find average dist to centroid
	float distCentroidX = 0.0f;
	float distCentroidY = 0.0f;
	for (size_t i = 0; i < nPoints; i++)
	{
		distCentroidX += std::fabs(inPts[i].x - centroidX);
		distCentroidY += std::fabs(inPts[i].y - centroidY);

	}
	distCentroidX /= nPoints;
	distCentroidY /= nPoints;

    if(distCentroidX < 1e-8) distCentroidX =1e-8;
    if(distCentroidY < 1e-8) distCentroidY =1e-8;

    float scaleX = 1.0f / distCentroidX;
	float scaleY = 1.0f / distCentroidY;

	//Write normalization matrix T
	T = cv::Mat::eye(3, 3, CV_32F);
	T.at<float>(0, 0) = scaleX;
	T.at<float>(0, 2) = -scaleX * centroidX;
	T.at<float>(1, 1) = scaleY;
	T.at<float>(1, 2) = -scaleY * centroidY;
	T.at<float>(2, 2) = 1;

	//multiply all points
	for (size_t i = 0; i < nPoints; i++)
		outPts[i] = cv::Point2f((scaleX * inPts[i].x - scaleX * centroidX), (scaleY * inPts[i].y + -scaleY * centroidY));

}

float MVGError::reprojectionError(const cv::Mat& T, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const std::vector<PtPair>& mvMatches12, std::vector<bool>& inliers)
{
	float threshold = 5.991f;
	float error = 0.0f;

	size_t nPts = mvMatches12.size();
	inliers = std::vector<bool>(nPts, false);

	cv::Mat Tinv = T.inv();

	//read all elements from Transformation matrices once and store as consts
	const float T11 = T.at<float>(0, 0);
	const float T12 = T.at<float>(0, 1);
	const float T13 = T.at<float>(0, 2);
	const float T21 = T.at<float>(1, 0);
	const float T22 = T.at<float>(1, 1);
	const float T23 = T.at<float>(1, 2);
	const float T31 = T.at<float>(2, 0);
	const float T32 = T.at<float>(2, 1);
	const float T33 = T.at<float>(2, 2);

	const float Tinv11 = Tinv.at<float>(0, 0);
	const float Tinv12 = Tinv.at<float>(0, 1);
	const float Tinv13 = Tinv.at<float>(0, 2);
	const float Tinv21 = Tinv.at<float>(1, 0);
	const float Tinv22 = Tinv.at<float>(1, 1);
	const float Tinv23 = Tinv.at<float>(1, 2);
	const float Tinv31 = Tinv.at<float>(2, 0);
	const float Tinv32 = Tinv.at<float>(2, 1);
	const float Tinv33 = Tinv.at<float>(2, 2);

	//reprojection errors from pt1 to image2 and vice-versa
	float error12 = 0.0f;
	float error21 = 0.0f;

	//score( w.r.t threshold parameter)
	float score12 = 0.0f;
	float score21 = 0.0f;

	for (size_t i = 0; i < nPts; i++)
	{
		//inlier or not
		bool isInlier = true;

		const float p1X = pts1[mvMatches12[i].first].x;
		const float p1Y = pts1[mvMatches12[i].first].y;

		const float p2X = pts2[mvMatches12[i].second].x;
		const float p2Y = pts2[mvMatches12[i].second].y;

		//project pt1 in image 2
		const float denP1I2 = 1.0f / (T31 * p1X + T32 * p1Y + T33);
		const float p1XinImage2 = (T11 * p1X + T12 * p1Y + T13) * denP1I2;
		const float p1YinImage2 = (T21 * p1X + T22 * p1Y + T23) * denP1I2;

		//error from pt1 on image 2
		const float dist12 = ((p2X - p1XinImage2) * (p2X - p1XinImage2) + (p2Y - p1YinImage2) * (p2Y - p1YinImage2));
		error12 += dist12;
		const float passScore12 = threshold - dist12;
		if (passScore12 < 0)
			isInlier = false;
		else
			score12 += passScore12;

		//maybe stop here and avoid checking pt2 in image 1?


		//project pt2 in image 1
		const float denP2I1 = 1.0f / (Tinv31 * p2X + Tinv32 * p2Y + Tinv33);
		const float p2XinImage2 = (Tinv11 * p2X + Tinv12 * p2Y + Tinv13) * denP2I1;
		const float p2Y1inImage2 = (Tinv21 * p2X + Tinv22 * p2Y + Tinv23) * denP2I1;

		//error from pt2 on image 1
		const float dist21 = ((p1X - p2XinImage2) * (p1X - p2XinImage2) + (p1Y - p2Y1inImage2) * (p1Y - p2Y1inImage2));
		error21 += dist21;
		const float passScore21 = threshold - dist21;

		if (passScore21 < 0)
			isInlier = false;
		else
			score21 += passScore21;

		inliers[i] = isInlier;
	}

	return score12 + score21;
}

float MVGError::epipolarLineError(const cv::Mat& T, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const std::vector<PtPair>& mvMatches12, std::vector<bool>& inliers)
{
	const float th = 3.841f;
	const float thScore = 5.991f;
	float error = 0.0f;


	size_t nPts = mvMatches12.size();
	inliers = std::vector<bool>(nPts, false);

	//reprojection errors from pt1 to image2 and vice-versa
	float error1 = 0.0f;
	float error2 = 0.0f;

	//score( w.r.t threshold parameter)
	float score1 = 0.0f;
	float score2 = 0.0f;

	const float f11 = T.at<float>(0, 0);
	const float f12 = T.at<float>(0, 1);
	const float f13 = T.at<float>(0, 2);
	const float f21 = T.at<float>(1, 0);
	const float f22 = T.at<float>(1, 1);
	const float f23 = T.at<float>(1, 2);
	const float f31 = T.at<float>(2, 0);
	const float f32 = T.at<float>(2, 1);
	const float f33 = T.at<float>(2, 2);



	for (size_t i = 0; i < nPts; i++)
	{
		//inlier or not
		bool isInlier = true;

		const float p1X = pts1[mvMatches12[i].first].x;
		const float p1Y = pts1[mvMatches12[i].first].y;

		const float p2X = pts2[mvMatches12[i].second].x;
		const float p2Y = pts2[mvMatches12[i].second].y;

		//epipolar line from point 1 in image 2
		//F*p1
		const float eplImage2A = f11 * p1X + f12 * p1Y + f13;
		const float eplImage2B = f21 * p1X + f22 * p1Y + f23;
		const float eplImage2C = f31 * p1X + f32 * p1Y + f33;

		//epipolar line from point 2 in image 1
		//FT*p2
		const float eplImage1A = f11 * p2X + f21 * p2Y + f31;
		const float eplImage1B = f12 * p2X + f22 * p2Y + f32;
		const float eplImage1C = f13 * p2X + f23 * p2Y + f33;

		//distance l2 to p2
		const float distl2p2 = eplImage2A * p2X + eplImage2B * p2Y + eplImage2C;
		const float distl2p2Norm = eplImage2A * eplImage2A + eplImage2B * eplImage2B;
		const float dist2 = distl2p2 * distl2p2 / distl2p2Norm;

		//distance l1 to p1
		const float distl1p1 = eplImage1A * p1X + eplImage1B * p1Y + eplImage1C;
		const float distl1p1Norm = eplImage1A * eplImage1A + eplImage1B * eplImage1B;
		const float dist1 = distl1p1 * distl1p1 / distl1p1Norm;

		error2 += dist2;
		error1 += dist1;
		if (dist1 > th)
			isInlier = false;
		else
			score1 += thScore - dist1;

		if (dist2 > th)
			isInlier = false;
		else
			score2 += thScore - dist2;

		inliers[i] = isInlier;
	}

	const float rms1 = sqrt(error1 / nPts);
	const float rms2 = sqrt(error1 / nPts);

	return score1 + score2;
}

float MVGError::sampsonError(const cv::Mat& T, const std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2, std::vector<bool>& inliers)
{
	return 0.0f;
}

size_t MVGError::updateTrials(const size_t inlierCount, const size_t nOfPoints, const size_t numberOfSamples, const float confidence)
{
	//Update N of Trials given better model
	float inlierRatio = static_cast<float>(inlierCount) * (1.0f / static_cast<float>(nOfPoints));
	float inlierRatio8 = powf((inlierRatio), numberOfSamples);
	float logOneMinusInlierRatio8 = std::log(1 - inlierRatio8);
	float logOneMinusP = std::log(1 - confidence);
	return  static_cast<size_t>(std::ceil(logOneMinusP / logOneMinusInlierRatio8));
}

bool MVGUtils::generateRandomSets(size_t nElements, size_t indexRange, std::vector<std::vector<size_t>>& sets, unsigned long int nSets)
{
	if (nElements > indexRange)
		return false;

	size_t possibleSets = 0;

	for (size_t i = 0; i < nSets; i++)
	{
		std::set<size_t> indexSet;
		generateRandomIndexes(0, indexRange, indexSet, nElements);
		std::vector<size_t> setToVector(indexSet.begin(), indexSet.end());
		sets.push_back(setToVector);
	}

	return (nSets > 0);
}

void MVGUtils::generateRandomIndexes(size_t start, size_t end, std::set<size_t>& indexes, size_t numberOfIndexes)
{
	indexes.clear();
	std::random_device rd; // obtain a random number from hardware
	std::mt19937 gen(rd()); // seed the generator
	std::uniform_int_distribution<> distr(start, end - 1); // define the range

	while (indexes.size() < numberOfIndexes)
		indexes.insert(distr(gen));
}

void MVGUtils::cvTovector(const cv::Mat &mat, std::vector<std::vector<float>> &v)
{
    v = std::vector<std::vector<float> > (mat.rows,std::vector<float>(mat.cols));
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            v[i][j] = mat.at<float>(i, j);
        }
    }
}



