#pragma once
#ifndef TRACKING_H
#define TRACKING_H

#include <iostream>

//external libraries headers
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/imgproc.hpp>
#include <glm/glm.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "slamParams.h"
#include "frame.h"
#include "mvg.h"
#include "map.h"
#include "mapPoint.h"
#include "optimizer.h"
#include "mapping.h"
#include "mapping.h"

class Mapping;
class Map;
class Viewer;

enum TrackingStates
{
    NO_FRAME,
    NOT_INITIALIZED,
    TRACKING,
    LOST,
    STOP_TRACKING
};


class Tracking
{
public:
	Tracking() {};
    ~Tracking();
	Tracking(SlamParams* slamParams);
	Tracking(Map* map, SlamParams* slamParams);
	bool run(const VertexFeatures& vertexFeatures, const double timestamp, unsigned long int mFrameCounter);

    void removeFrame(Frame* frame);
    void setMapping(Mapping* mapping);
    void setViewer(Viewer* viewer);
    void setUpdateFlag();
    void clearUpdateFlag();
    bool checkUpdateFlag(void) const {return m_frameUpdateFlag;}
    void convertGLM2CV(const glm::mat4& glmMat, cv::Mat& cvMat);
    void stopTracking(){ m_trackingState = TrackingStates::STOP_TRACKING;    Logger<std::string>::LogInfoI("Stopping Tracking.");}
	std::vector<Frame*> getFrames() { return m_frameSequence; }
	std::vector<TFrame*> getTFrames() { return m_tFrameSequence; }
private:
    bool initialize();
    void initializeKMatrix();
	bool createInitialMap();
    bool createNewFrame();
    bool trackNextFrame();
    bool trackLostFrame(Frame* frame);
    void updateLocalMap();
    size_t trackLocalMap();
    bool isInFrustum(Frame* frame, MapPoint* mp);
    size_t updateFrameOutliers();
    bool updateTrackedFrame();

private:
	size_t mframesGap{ 5 };
	size_t mNextFrameNumber{ 0 };
	SlamParams* m_slamParams{ NULL };
	TrackingStates m_trackingState;

    unsigned long int m_lastKFrameID{0};
    unsigned long int m_trackFrameCnt{0};
    size_t m_badTrackingCount{0};

    size_t m_minJumpFrames{0};
    size_t m_maxJumpFrames{0};

    //initial point matches
	std::vector<PtPair> m_prunedMatchesIdx;

    Mapping* m_mapper{nullptr};
	Map* m_map{nullptr};
    Viewer* m_viewer{nullptr};

    Frame m_newFrame;
    Frame m_oldFrame;
    Frame* m_referenceFrame{nullptr};

    std::vector<Frame*> m_frameSequence;
    std::vector<TFrame*> m_tFrameSequence;

    std::vector<Frame*> m_localFrames;
    std::vector<MapPoint*> m_localMapPoints;
	MVG m_MVG;
	std::vector<cv::Point3f> m_initP3Dw;

    cv::Mat m_motion;
    bool m_trackWithMotion{false};
	size_t m_stillCount{0};
    bool m_frameUpdateFlag{true};
    std::mutex m_mutexUpdate;

    int m_trackInliers{0};
    int m_matchedtoLocalMap{0};


};
#endif // !TRACKING_H

