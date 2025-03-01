//
// Created by caps on 12.10.23.
//
#include "mapPoint.h"

long unsigned int MapPoint::nextID = 0;
std::mutex MapPoint::mMapPointsGlobalMutex;

MapPoint::MapPoint(const cv::Mat &position, Frame *frame, std::shared_ptr<Map> &map, const size_t descriptor, float e) : mPosition(position.clone()), m_map(map), m_vtxDescriptor(descriptor), m_error(e)
{
    {
        std::unique_lock<std::mutex> lock(mMapPointsGlobalMutex);
        m_ID = nextID++;
    }

//TODO: remove this; this is more for debugging purposes (visible)
    m_vposition[0] = mPosition.at<float>(0);
    m_vposition[1] = mPosition.at<float>(1);
    m_vposition[2] = mPosition.at<float>(2);

    cv::Mat cameraOrigin = frame->getTw();
    cv::Mat pt2Cam = mPosition-cameraOrigin;
    m_initialNormal = pt2Cam/cv::norm(pt2Cam);
    m_initialDistance = cv::norm(pt2Cam);
}

MapPoint::MapPoint(const cv::Mat &position, Frame *frame, std::shared_ptr<Map> &map, const size_t descriptor) : mPosition(position.clone()), m_map(map), m_vtxDescriptor(descriptor)
{
    {
        std::unique_lock<std::mutex> lock(mMapPointsGlobalMutex);
        m_ID = nextID++;
    }

    //TODO: remove this; this is more for debugging purposes (visible)
    m_vposition[0] = mPosition.at<float>(0);
    m_vposition[1] = mPosition.at<float>(1);
    m_vposition[2] = mPosition.at<float>(2);

    cv::Mat cameraOrigin = frame->getTw();
    cv::Mat pt2Cam = mPosition-cameraOrigin;
    m_initialNormal = pt2Cam/cv::norm(pt2Cam);
    m_initialDistance = cv::norm(pt2Cam);
}


MapPoint::MapPoint(const cv::Mat &position) : mPosition(position.clone())
{
    {
        std::unique_lock<std::mutex> lock(mMapPointsGlobalMutex);
        m_ID = nextID++;
    }
    //TODO: remove this; this is more for debugging purposes (visible)
    m_vposition[0] = mPosition.at<float>(0);
    m_vposition[1] = mPosition.at<float>(1);
    m_vposition[2] = mPosition.at<float>(2);
}



void MapPoint::addFrameView(Frame *frame, size_t idx)
{
    std::unique_lock<std::mutex> lock(m_mutexPosition);
    //check if frame already a viewer
    if(m_frameViews.count(frame))
        return;
    m_frameViews[frame] = idx;

    //increment n of viewers
    m_ViewCnt++;
}

const std::map<Frame *, size_t>& MapPoint::getFrameViews(void)
{
    std::unique_lock<std::mutex> lock(m_mutexFeatures);
    return  m_frameViews;
}

void MapPoint::setPos(const cv::Mat &pos)
{
    //std::unique_lock<std::mutex> lock(mMapPointsGlobalMutex);
    std::unique_lock<std::mutex> lock2(m_mutexPosition);
    pos.copyTo(mPosition);
    m_vposition[0] = mPosition.at<float>(0);
    m_vposition[1] = mPosition.at<float>(1);
    m_vposition[2] = mPosition.at<float>(2);
}

bool MapPoint::isInFrameView(Frame* frame)
{
    std::unique_lock<std::mutex> lock(m_mutexFeatures);
    return m_frameViews.count(frame);
}

//TODO: Further test removal, for now simplified version
bool MapPoint::removeFrameView(Frame *frame)
{
    std::unique_lock<std::mutex> lock(m_mutexFeatures);
    if(m_frameViews.count(frame))
    {
        m_frameViews.erase(frame);
        m_ViewCnt--;
        return true;
    }
    return false;
}

void MapPoint::replace(MapPoint *other)
{
    //here 'other' map point replaces by this map point
    if(other->m_ID == m_ID)
        return;

    //make deep copy of this MP's data structures frame views
    int nVisibleCopy, nFoundCopy;
    std::map<Frame*, size_t> frameViewsCopy;
    {
        std::unique_lock<std::mutex> lock1(m_mutexFeatures);
        std::unique_lock<std::mutex> lock2(m_mutexPosition);
        frameViewsCopy = m_frameViews;
        //nVisibleCopy = mNVisible;
        //nFoundCopy = mNFound;
        m_frameViews.clear();
        m_mapPointReplaced = other;
    }

    //replace this map point with 'other' map point on frames that view it
    for(std::map<Frame*, size_t>::iterator it=frameViewsCopy.begin(),itEnd=frameViewsCopy.end(); it!=itEnd; it++)
    {
        Frame* frame = it->first;
        size_t idx = it->second;

        //if other map point is not in frame view replace/add it using current map points index
        //also add map point to the frame
        if(!other->isInFrameView(frame))
        {
            frame->replaceMapPoint(idx,other);
            other->addFrameView(frame,idx);
        }
        else
        {
            //otherwise if map point is present in this frame's view, set it to null
            frame->removeMapPoint(this);
        }
    }

    //other->incrementFound
    //other->incrementVisible

    //finally remove point from map
    m_map->removeMapPoint(this);

}


