#include "map.h"

void Map::addMapPoint(MapPoint* point)
{
	std::unique_lock<std::mutex> lock(m_mutexMap);
	m_mapPoints.insert(point);
    if(!m_ID_mapPoints.count(point->getID()))
        m_ID_mapPoints[point->getID()] = point;
}

void Map::insertFrame(Frame* frame)
{
    std::unique_lock<std::mutex> lock(m_mutexMap);
    m_KFrames.insert(frame);

}

void Map::insertTFrame(TFrame* frame)
{
    std::unique_lock<std::mutex> lock(m_mutexMap);
    m_TFrames.insert(frame);
}

std::vector<Frame *> Map::getFrames()
{
    std::unique_lock<std::mutex> lock(m_mutexMap);
    return std::vector<Frame *>(m_KFrames.begin(), m_KFrames.end());
}

std::vector<TFrame *> Map::getTFrames()
{
    std::unique_lock<std::mutex> lock(m_mutexMap);
    return std::vector<TFrame *>(m_TFrames.begin(), m_TFrames.end());
}

std::vector<MapPoint *> Map::getMapPoints()
{
    std::unique_lock<std::mutex> lock(m_mutexMap);
    return std::vector<MapPoint *>(m_mapPoints.begin(), m_mapPoints.end());
}

std::map<unsigned long int, MapPoint *> Map::getMapPointsByIdx()
{
    std::unique_lock<std::mutex> lock(m_mutexMap);
    return m_ID_mapPoints;
}

void Map::removeMapPoint(MapPoint *mapPoint)
{
    std::unique_lock<std::mutex> lock(m_mutexMap);
    m_mapPoints.erase(mapPoint);

    //TODO: to be tested
    //delete mapPoint;
}

void Map::addNewMapPoints(std::vector<MapPoint *> newMapPoints)
{
    std::unique_lock<std::mutex> lock(m_mutexMap);
    m_newMapPoints.clear();
    m_newMapPoints.insert(newMapPoints.begin(), newMapPoints.end());
}

void Map::removeFrame(Frame *frame)
{
    std::unique_lock<std::mutex> lock(m_mutexMap);
    m_KFrames.erase(frame);

    //TODO: to be tested
    //delete frame;
}
