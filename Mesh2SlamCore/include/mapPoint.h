//
// Created by caps on 12.10.23.
//

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <map>
#include <mutex>

#include<opencv2/core/core.hpp>

#include "frame.h"
#include "map.h"

class Frame;
class Map;


class MapPoint
{
public:
    MapPoint(const cv::Mat& position, Frame* frame, std::shared_ptr<Map>& map);
    MapPoint(const cv::Mat& position, Frame* frame, std::shared_ptr<Map>& map, const size_t descriptor);
    MapPoint(const cv::Mat& position, Frame* frame, std::shared_ptr<Map>& map, const size_t descriptor, float e);
    MapPoint(const cv::Mat& position);
    MapPoint(const MapPoint& other) = default;
    MapPoint& operator=(MapPoint&& other) noexcept = default;

    std::map<Frame*,size_t> getFrameViews(void);
    const cv::Mat& getPosition() const {return mPosition;}
    const int getDescriptor() const {return m_vtxDescriptor;}
    long unsigned int getFrameUpdateID(void) {return m_frameUpdate_ID;}
    size_t getVisibleCount(void) const {return m_visibleCount;}
    const long unsigned int getLastFrameViewID() const {return m_lastFrame_ID;}
    const cv::Point2i& getLastImageCoord(void) const {return m_lastCoord;}
    const unsigned long int getID() const {return m_ID;}
    unsigned long int getActiveBAID(void) const { return m_BA_ID; }
    unsigned long int getActiveFuseID(void) const { return m_Fuse_ID; }
    unsigned long int getNFrameViews(void) const {return m_ViewCnt;}
    bool isInFrameView(Frame* frame);
    char getPointType(void) const {return mPointType;}
    size_t getBadCount(void) const {return m_badCount;}
    bool isBad(void) const {return m_isBad;}

    void setBad() {m_isBad = true;}
    void addFrameView(Frame* frame, size_t idx);
    bool removeFrameView(Frame*);
    void replace(MapPoint* mapPoint);
    void setActiveBAID(unsigned long int id) { m_BA_ID = id; }
    void setActiveFuseID(unsigned long int id) { m_Fuse_ID = id; }
    void setPos(const cv::Mat& pos);
    void setFrameUpdateID(long unsigned int id) { m_frameUpdate_ID = id;}
    void increaseVisible(void) {++m_visibleCount;}
    void increaseBad(void) {++m_badCount;}
    void setLastFrameViewID(const long unsigned int id) { m_lastFrame_ID = id;}
    void setLastImageCoord(const cv::Point2i& pt) { m_lastCoord = pt;}
    void setDescriptor(const size_t descriptor) { m_vtxDescriptor = descriptor;}
    void setPointType(char type){mPointType = type;}
    void setError(const float error){m_error = error;}
    const float getError() const {return m_error;}
public:
    static std::mutex mMapPointsGlobalMutex;

private:
    static unsigned long int nextID;
    unsigned long int m_ID{0};
    unsigned long int m_BA_ID{0};
    unsigned long int m_Fuse_ID{0};
    long unsigned int m_lastFrame_ID{0};
    long unsigned int m_frameUpdate_ID{0};

    float m_error{0.0f};
    std::shared_ptr<Map> m_map{nullptr};
    MapPoint* m_mapPointReplaced{nullptr};

    bool m_fused{false};

    std::map<Frame*,size_t> m_frameViews;
    unsigned long int m_ViewCnt{0};

    cv::Mat mPosition;
    std::vector<float> m_vposition{0.0f, 0.0f, 0.0f};
    cv::Mat m_normal;
    cv::Mat m_initialNormal;
    cv::Mat m_descriptor;
    size_t m_vtxDescriptor{0};

    size_t m_visibleCount{1};
    size_t m_badCount{0};
    bool m_isBad{false};

    cv::Point2i m_lastCoord{0, 0};
    float m_initialDistance{0.0f};

    std::mutex m_mutexFeatures;
    std::mutex m_mutexPosition;

    char mPointType{0}; //new=0,old=1,fused=2
};
#endif //MAPPOINT_H
