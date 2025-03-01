

#ifndef MAP_H
#define MAP_H

#include <iostream>
#include <set>
#include <vector>
#include <mutex>

//#include "keyFrame.h"
#include "frame.h"
#include "slamParams.h"

class Frame;
class TFrame;
class MapPoint;

struct SlamData
{
	const bool isEmpty() const {if(mapPoints.empty() && tFrames.empty() && kFrames.empty()){return true;}return false;}
	std::vector<glm::vec3> mapPoints;
	std::vector<glm::mat4> tFrames;
	std::vector<glm::mat4> kFrames;
	std::vector<double> tFrameTimestamps;
	std::vector<double> kFrameTimestamps;
	std::string name;
};

class Map
{
public:
	Map() {};
	Map(SlamParams* slamParams);
    ~Map() {}

	void insertFrame(Frame* frame);
	void insertTFrame(TFrame* frame);
	void removeFrame(Frame* frame);
    void addMapPoint(MapPoint* map);
    void removeMapPoint(MapPoint* mapPoint);
    void addNewMapPoints(std::vector<MapPoint*> newMapPoints);
    std::vector<Frame*> getFrames();
    std::vector<TFrame*> getTFrames();
    std::vector<MapPoint*> getMapPoints();
    std::map<unsigned long int, MapPoint*> getMapPointsByIdx();
    std::mutex& getMutexMap() {return m_mutexMap;}

private:

	long unsigned int m_maxKF_ID{0};

	std::set<Frame*> m_KFrames;
	std::set<TFrame*> m_TFrames;
	std::set<MapPoint*> m_mapPoints;
	std::set<MapPoint*> m_newMapPoints;
    std::map<unsigned long int, MapPoint*> m_ID_mapPoints;
	std::mutex m_mutexMap;
};

#endif