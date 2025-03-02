//
// Created by caps80 on 03.12.23.
//

#ifndef MAPPING_H
#define MAPPING_H

#include <mutex>
#include <chrono>
#include <thread>

#include "tracking.h"
#include "optimizer.h"
#include "frame.h"
#include "map.h"
#include "mapPoint.h"
#include "viewer.h"

class Tracking;
class Map;

enum MappingStates
{
    NO_MAP = 0,
    MAP_INITIALIZED = 1,
    STOP_MAPPING = 2
};

class Mapping
{
public:

    Mapping(Map* map, SlamParams* slamParams) : m_map(map), m_slamParams(slamParams){ m_mappingState=MappingStates::MAP_INITIALIZED;}
    void run();
    void insertFrame(Frame* frame);
    void setTracker(Tracking* tracker);
    void setViewer(Viewer* viewer);
    void setState(const MappingStates& mapState){m_mappingState = mapState;}
    void stopMapping(void) {m_mappingState = MappingStates::STOP_MAPPING; m_stop = true;    Logger<std::string>::LogInfoI("Stopping Mapper.");}

private:
    bool processNewFrames();
    bool deleteMapPoints();
    int  createMapPoints();
    bool searchNeighbors(int n);
    void removeFrames();
    bool newFrameAvailable();

private:
    SlamParams* m_slamParams;
    bool m_updated{true};
    Frame* m_newFrame{nullptr};
    std::list<Frame*> m_availableFrames;
    Map* m_map{nullptr};
    Tracking* m_tracker{nullptr};
    Viewer* m_viewer{nullptr};
    std::vector<MapPoint*> m_newMapPoints;

    MappingStates m_mappingState{MappingStates::NO_MAP};
    std::atomic<bool> m_abortBA;
    std::atomic<bool> m_stop{false};

    std::mutex m_frameMutex;
};
#endif //MAPPING_H
