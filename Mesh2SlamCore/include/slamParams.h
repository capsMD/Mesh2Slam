#pragma once
#ifndef SLAM_PARAMS_H
#define SLAM_PARAMS_H

#include <vector>
#include <string>
#include <glm/glm.hpp>

struct SlamParams
{
    SlamParams& operator=(const SlamParams& other)
	{
		if (this != &other)
		{
			inputParams     = other.inputParams;
			camParams       = other.camParams;
			featureParams   = other.featureParams;
            ransacParams    = other.ransacParams;
            errorParams     = other.errorParams;
            viewerParams    = other.viewerParams;
            mapPointParams  = other.mapPointParams;
		}
		return *this;
	}


	struct InputParams
	{
        int type { 0 };
        int runMode { 0 };
		int useCam{ 1 };
		int  camDevice{ 0 };
		std::string fileDir;
		std::string sourceFileName;
		std::string imageFilesSubdirectory;
        std::string camCaptureString;
	}inputParams;

	struct CamParams
	{
		float fx{ 0.0f };
		float fy{ 0.0f };
		float cx{ 0.0f };
		float cy{ 0.0f };

		float k1{ 0.0f };
		float k2{ 0.0f };
		float p1{ 0.0f };
		float p2{ 0.0f };
		float k3{ 0.0f };

        int w{1024};
        int h{1024};


	}camParams;

    struct OptimizationParams
    {
        int BAIterations1{50};
        int BAIterations2{10};
        int GBAIterations{50};
        int PNPIterations{10};
        int GBAJumpFrames{500};
        int SearchN{10};
        int minJumpFrames{0};
        int maxJumpFrames{10};
        float minDisplacement{1};
    }optimizationParams;

    struct MapPointParams
    {
        int maxPointDistance{60};
        int maxNewPoints{50};
    }mapPointParams;


	struct FeatureParams
	{
        int maxFeatures{ 200 };
		int minFeatures{ 100 };
		int minMatches{ 100 };
		int maxMatches{ 200 };
		int pruneDist{ 50 };
        int minRefFrameMatches{10};
        int searchRadius{50};
        int minTracked{30};
        float featuresMaxDepth{100.0f};

	}featureParams;

	struct RansacParams
	{	
		int maxTrials{ 1000 };
		float threshold{ 2.0f };
		float confidence{ 0.99f };
		int samples{ 8 };
	}ransacParams;

	struct ErrorParams
	{
		int		maxBadTrackCnt{3};
		float	minInlierRatio{ 0.9f };
		float	minParallax{ 0.99998f };
		float	reprojectThreshold{ 4 };
		int		minTriangulations{ 10 };
		float     maxSecondPassRatio{ 0.7f };
        int     maxBBRepDist{100};
	}errorParams;


    struct ViewerParams
    {
	    int forceOriginStart{1};
	    int runViewer{1};
    	int width{640};
    	int height{480};
    	float scaleFactor{1.0f};
    	float fov{90.0f};
    	float far{1000.0f};
    	float near{1.0f};
    	float camMoveFactor{1.0f};
    	std::string title {"CAPS-SLAM"};
    	glm::vec3 frameColor       { 0.0f,0.0f,0.0f };
    	glm::vec3 tframeColor      { 1.0f,1.0f,1.0f };
    	glm::vec3 framePathColor   { 0.0f,0.0f,0.0f };
    	glm::vec3 pointCloudColor  { 0.0f,0.0f,0.0f };
    	glm::vec3 tFrameColor      { 0.0f,0.0f,0.0f };
    	glm::vec3 meshColor		{0.0f,0.0f,0.0f};

        float camFov{ 0.0f };
        glm::vec3 camPos   { 0.0f,0.0f,0.0f };
        glm::vec3 camRight { 1.0f,0.0f,0.0f };
        glm::vec3 camUp    { 0.0f,1.0f,0.0f };
        glm::vec3 camTarget{ 0.0f,0.0f,1.0f };
    }viewerParams;


};

#endif // !SLAM_PARAMS_H
