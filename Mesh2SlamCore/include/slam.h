#pragma once
#ifndef SLAM_H
#define SLAM_H
//system headers
#include <thread>
#include <chrono>
#include <condition_variable>		 

//external libraries headers
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/imgproc.hpp>

//needed to read files in Android
#include <HelperFunctions.h>

//current application headers
#include "slamUtils.h"
#include "viewer.h"
#include "tracking.h"
#include "mapping.h"



class Map;



enum SlamState
{
	STARTUP = -1,
	IDLE = 0,
	RUNNING = 1,
	STOPPED = 3

};

class Slam
{
public:
	Slam(SlamParams* params, AAssetManager *assetManager);
    void processFrame(const VertexFeatures& vertexFeatures);
    bool updateViewer();
    void renderViewer();
	void run();
    void shutdown();
    void stopSLAM();
    Viewer* getViewer() {return m_viewer;}
    const SlamState getSlamState() const {return m_slamState;}
    bool cleanup();
private:

    SlamState m_slamState{SlamState::STARTUP};
    //this is counting directly from the source (plays independent of performance of slam)
    unsigned long int m_sourceFrameCnt{0};
	double m_timestamp;

    std::thread* m_mappingThread{nullptr};
    std::thread* m_viewerThread{nullptr};

    Mapping* m_mapper{nullptr};
    Viewer* m_viewer{nullptr};
    Map* m_map{nullptr};
	Tracking* m_tracker{nullptr};

	SlamParams* m_slamParams{nullptr};

	SlamData m_slamData;

    VertexFeatures m_frameData;
    std::atomic<bool> m_stop{false};
    std::condition_variable m_frameCondVariable;
    std::mutex m_frameMutex;
    std::mutex m_slamDataMutex;
    bool m_frameAvailable{false};

   AAssetManager *m_assetManager;

};

class SlamManager
{
public:
    SlamManager(SlamParams* slamParams, AAssetManager *assetManager) : m_slamParams(slamParams), m_assetManager(assetManager){}
    bool initializeSLAM();
    void updateFrame();
    void renderSLAM();
    void startSLAM() ;
    void stopSLAM();
    Slam* getSlam() {return m_slam;}
    Viewer* getViewer() { if(m_slam!= nullptr) return m_slam->getViewer(); return nullptr;}
    bool isInitialized(void) const {return m_slamInitialized;}
    bool cleanup();
private:
    Slam* m_slam{nullptr};
    SlamParams* m_slamParams{nullptr};
    std::thread* m_slamThread{nullptr};
    std::shared_ptr<std::map<std::string, std::shared_ptr<Shader> > > m_shaders;
    std::atomic<bool> m_slamInitialized{false};
    std::atomic<bool> m_stopped{false};
    AAssetManager *m_assetManager;
    cv::Mat m_camImage;

};
#endif
