#pragma once
#ifndef FRAME_H
#define FRAME_H

#include <iostream>
#include <list>
#include <set>
#include <unordered_map>

#include <Eigen/Dense>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

#include "slamUtils.h"
#include "slamParams.h"
//#include "FeatureDetector.h"
#include "mapPoint.h"
//#include "camera.h"

#define GRID_ROWS 48
#define GRID_COLS 64

class MapPoint;


struct CamParams {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    int width = 0;
    int height = 0;

    CamParams() = default;

    CamParams(SlamParams* slamParams) :
            fx(slamParams->camParams.fx),
            fy(slamParams->camParams.fy),
            cx(slamParams->camParams.cx),
            cy(slamParams->camParams.cy),
            width(slamParams->camParams.w),
            height(slamParams->camParams.h){}

    CamParams(const CamParams& other) = default;
    CamParams& operator=(const CamParams& other) = default;
};

typedef std::pair<int, int> PtPair;

enum FrameType
{
    PTS = 0,
    FEATURES = 1,
    SIMPTS,
};

class Frame
{
public:
	Frame(){std::cout << "Frame default constructor being called"<< std::endl;}
	Frame(const Frame& other);
	Frame(const Frame* other);
	Frame(const VertexFeatures& vertexFeatures, double timestamp, unsigned long int frameCount, SlamParams* params);
	Frame& operator=(const Frame& other);
	Frame& operator=(const Frame* other);


	const double getTimeStamp() const {return m_timeStamp;}
	unsigned long int getID(void) const { return m_ID; }
	unsigned long int getActiveBAID(void) const { return m_BA_ID; }
	unsigned long int getActiveFuseID(void) const { return m_Fuse_ID; }
	void setActiveBAID(unsigned long int id) { m_BA_ID = id; }
	void setActiveFuseID(unsigned long int id) { m_Fuse_ID = id; }
    void setDescriptorMapPoint(const size_t descriptor, MapPoint* mp);
    void setDescriptorIdx(const size_t descriptor, const size_t idx);
    MapPoint* getDescriptorMapPoint(const size_t descriptor);
    MapPoint* getMapPoint(const size_t idx);
    bool getDescriptorIdx(const size_t descriptor, size_t& idx);
    const CamParams& getCamPrjParams(void) const {return m_camParams;}

    void setR(const cv::Mat& orientation);
    void sett(const cv::Mat& position);
    void setPose(const cv::Mat& pose);
    void updateTransform(void);
    void setError(float e) { m_error=e;}
    void addMapPoint(MapPoint*, const size_t idx);
    void replaceMapPoint(const size_t idx, MapPoint* mapPoint);
    bool removeMapPoint(MapPoint* mapPoint);
    bool removeMapPoint(const size_t idx);
    void removeFrame();
    void addConnection(Frame*, const int &weight);
    void removeConnection(Frame* frame);
    void addChild(Frame*);
    void removeChild(Frame*);
    void setParent(Frame*);
	void setConnections();
    void updateConnections();

	const std::vector<cv::KeyPoint>& getKeyPoints(void) const { return m_keyPoints; }
	const cv::Mat& getDescriptors(void) const { return m_descriptors; }
	const std::vector<PtPair>& getFeatureMatches(void) const { return m_initMatches; }
    const std::vector<size_t>& getSimDescriptors(void) const {return m_vtxDescriptors;}

    std::vector<MapPoint*> getMapPoints(void);
    std::vector<bool>       getOutliers(void) {return m_outliers;}
    void setMapPoints(std::vector<MapPoint*>& mps) { m_mapPoints = mps;}
    const unsigned int getNTrackedMapPoints(const unsigned int) ;
    const unsigned int checkRepeatedFeatures(Frame* frame);

    //can be written
	cv::Mat getRc(void) { std::unique_lock<std::mutex> lock(m_mutexPose);    return m_Rc.clone(); }
	cv::Mat getTc(void) {std::unique_lock<std::mutex> lock(m_mutexPose);     return m_Tc.clone(); }
	cv::Mat getPosec(void) {std::unique_lock<std::mutex> lock(m_mutexPose);  return m_posec.clone(); }
	cv::Mat getTw(void) {std::unique_lock<std::mutex> lock(m_mutexPose);     return m_Tw.clone();}
    cv::Mat getRw(void) {std::unique_lock<std::mutex> lock(m_mutexPose);     return m_Rw.clone();}
    cv::Mat getPosew(void) {std::unique_lock<std::mutex> lock(m_mutexPose);  return m_posew.clone();}

	bool MatchWithFrame(Frame& otherFrame);

    std::vector<size_t> getFeaturesIdxFromGrid(const cv::Point2i &imagePt);
    std::vector<Frame*> getConnectedFrames();
    std::vector<Frame*> getNConnectedFrames(const int N);
    std::vector<int> getConnectedWeights() const {return m_orderedConnectedWeights;}
    int getConnectedWeight(Frame* frame);

    size_t getN(void) const {return N;}
    float computeMedianDepth();
    bool checkReprojectionError();

    unsigned long int getImageFrameNumber(void) const {return m_ID;}
    std::vector<bool> getOutliers(void) const {return m_outliers;}
    std::map<size_t, MapPoint*> getDescriptorToMapPoints(void) const {return m_descriptor_mapPoint;}
    std::unordered_map<size_t, size_t> getDescriptorToIdx(void) const {return m_descriptor_idx;}
    bool getFirstConnection() const {return m_firstConnection;}
    std::map<Frame*,int> getConnectionToWeights(){return m_connectionWeights;}
    std::set<Frame*> getChildren(){return m_children;}
    const glm::mat4 getVirtualCamPose(void) const {return m_virtualCamPose;}
    const std::vector<size_t>(&getGrid())[GRID_COLS][GRID_ROWS]{return m_grid;}
	const bool isInitialized() const {return m_initialized;}
private:
    void setMatricesD();
    void initialize();
    void setFeaturesIdx2Grid();
    void undistortKeypoints();

private:
    static unsigned long int nextID;
    unsigned long int m_ID{0};
    unsigned long int m_BA_ID{0};
    unsigned long int m_Fuse_ID{0};

	bool m_initialized{false};
    unsigned int m_trackedCnt{0};

	Frame* m_previousFrame{nullptr};

    CamParams m_camParams;

    size_t N{0};
    std::vector<bool> m_outliers;
    std::vector<MapPoint*> m_mapPoints;

    std::vector<cv::KeyPoint> m_keyPoints;
	cv::Mat m_descriptors;
	std::vector<size_t> m_vtxDescriptors; //descriptor
    std::map<size_t, MapPoint*> m_descriptor_mapPoint; // descriptor - map point
    std::unordered_map<size_t, size_t> m_descriptor_idx; //descriptor - index

	std::vector<cv::DMatch> m_matches;
	std::vector<PtPair> m_initMatches;

    //Frame connections and weights
    std::vector<Frame*> m_orderedConnectedFrames;
    std::vector<int> m_orderedConnectedWeights;
    bool m_firstConnection{true};
    Frame* m_parent{nullptr};
    std::map<Frame*,int> m_connectionWeights;
    std::set<Frame*> m_children;

    //std::vector<MapPoint*> m_mapPoints;

	cv::Mat m_Rc;
	cv::Mat m_Tc;
	cv::Mat m_posec;

	cv::Mat m_Rw;
	cv::Mat m_Tw;
	cv::Mat m_posew;

    std::vector<std::vector<float> > m_vR;
    std::vector<std::vector<float> > m_vT;
    std::vector<std::vector<float> > m_vPose;

    int m_searchRadius{0};
    float m_error{0.0f};
	double m_timeStamp{0};

    SlamParams* m_slamParams{ nullptr };

    std::mutex m_mutexPose;
    std::mutex m_mutexConnections;
    std::mutex m_mutexMapPoints;
    std::mutex m_mutexFeatures;

    static float m_invWidthScale;
    static float m_invHeightScale;
    std::vector<size_t> m_grid[GRID_COLS][GRID_ROWS];
    glm::mat4 m_virtualCamPose;
};

class TFrame
{
public:
    TFrame(){ initialize(); }
	const double getTimeStamp() const {return m_timeStamp;}
    void setPose(const cv::Mat &pose);
    void setFrameID(const unsigned long int id) {m_ID = id;}
	void setTimeStamp(const double timestamp) {m_timeStamp = timestamp;}
    cv::Mat getPosew() {std::unique_lock<std::mutex> lock(m_mutexPose);  return m_posew.clone();}
    unsigned long int getFrameID() const {return m_ID;}
private:
    void updateTransform();
    void setMatricesD();
    void initialize();
private:

    unsigned long int m_ID{0};
	double m_timeStamp{0};

    cv::Mat m_Rc;
    cv::Mat m_Tc;
    cv::Mat m_posec;

    cv::Mat m_Rw;
    cv::Mat m_Tw;
    cv::Mat m_posew;

    std::vector<std::vector<float> > m_vR;
    std::vector<std::vector<float> > m_vT;
    std::vector<std::vector<float> > m_vPose;

    std::mutex m_mutexPose;
};

class Matcher
{
public:
    Matcher(){}
    static int matchAndAddAll(Frame &newFrame, Frame *referenceFrame);
    static int fuseByProjection(Frame* newFrame, const  std::vector<MapPoint*> & mapPoints,std::vector<PtPair>& matches);
    static int matchAndAddByProjection(Frame* newFrame, Frame* oldFrame);
    static int matchAndAddByProjection(Frame* newFrame, const  std::vector<MapPoint*> & mapPoints);
    static int matchByEpipolarDist(Frame* f1, Frame* f2, cv::Mat F32, std::vector<PtPair>& matches);
    static bool checkEpipolarDistance(cv::Point2i p1, cv::Point2i p2, const cv::Mat& F, float distance);
    static int matchOldFrames(Frame* newFrame, std::vector<Frame*>& oldFrames, size_t maxFrames, Frame* bestFrame);
private:
};

#endif // !FRAME_H

