#include "frame.h"

unsigned long int Frame::nextID = 0;
float Frame::m_invWidthScale;
float Frame::m_invHeightScale;

Frame::Frame(const VertexFeatures &vertexFeatures, double timestamp, unsigned long int frameCount, SlamParams* slamParams) :
        m_virtualCamPose(vertexFeatures.m_pose), m_timeStamp(timestamp), m_ID(frameCount), m_slamParams(slamParams), N(vertexFeatures.m_pts.size()), m_camParams(slamParams)
{
    m_keyPoints.reserve(N);
    m_vtxDescriptors.reserve(N);

    for(const auto& pt : vertexFeatures.m_pts)
    {
        m_keyPoints.emplace_back(cv::Point2f(pt.y, pt.z), 1);
        m_vtxDescriptors.emplace_back(pt.x);
    }

    //assume every feature could generate a map point
    m_mapPoints.assign(N, NULL);
    m_outliers.assign(N, false);

    initialize();
}

void Frame::initialize()
{

    m_posec = cv::Mat::eye (4, 4, CV_32F);
    m_Rc = cv::Mat::eye (3, 3, CV_32F);
    m_Tc = cv::Mat::zeros(3, 1, CV_32F);
    m_posew = cv::Mat::eye (4, 4, CV_32F);
    m_Rw = cv::Mat::eye (3, 3, CV_32F);
    m_Tw = cv::Mat::zeros(3, 1, CV_32F);
    m_searchRadius = m_slamParams->featureParams.searchRadius;
    setMatricesD();


    m_invWidthScale = static_cast<float>(GRID_COLS) / static_cast<float>(m_camParams.width);
    m_invHeightScale = static_cast<float>(GRID_ROWS) / static_cast<float>(m_camParams.height);

    setFeaturesIdx2Grid();
    m_initialized = true;
}


bool Frame::MatchWithFrame(Frame& otherFrame)
{
    size_t nMatches = 0;

        static const size_t minMatches = m_slamParams->featureParams.minMatches;

        auto& otherFrameDescriptors = otherFrame.getSimDescriptors();
        for (size_t i = 0; i < N; i++)
        {
            for(size_t j = 0; j < otherFrame.N; j++)
            {
                if (m_vtxDescriptors[i] == otherFrameDescriptors[j])
                {
                    m_initMatches.push_back(std::make_pair(j, i));
                    nMatches++;
                }

            }
        }

        //check number of matches
        if(nMatches >= minMatches)
            return true;
        return false;

}

void Frame::undistortKeypoints()
{

}

void Frame::addMapPoint(MapPoint* mapPoint, const size_t idx)
{
    std::unique_lock<std::mutex> lock(m_mutexMapPoints);
    if (idx >= N) return; //this should never happen
    m_mapPoints[idx] = mapPoint;
    setDescriptorIdx(m_vtxDescriptors[idx], idx);
    setDescriptorMapPoint(m_vtxDescriptors[idx], mapPoint);
}

const unsigned int Frame::getNTrackedMapPoints(const unsigned int minFrames)
{
    m_trackedCnt = 0;

    //make local copy of map points from this frame
    std::vector<MapPoint*> vpMapPoints;
    {
        std::unique_lock<std::mutex> lock(m_mutexMapPoints);
        vpMapPoints = m_mapPoints;
    }
    for(size_t i = 0; i < N; i++)
    {
        MapPoint* pMP = vpMapPoints[i];
        if(pMP != nullptr)
        {
            std::map<Frame *, size_t> frameViews = pMP->getFrameViews();
            if (frameViews.size() >= minFrames)
                m_trackedCnt++;
        }
    }
    return m_trackedCnt;
}

void Frame::setConnections()
{
    //build a map of keyframes and a counter associated with it
    std::map<Frame*, int> frameConnectionWeights;

    //make local copy of map points from this frame
    std::vector<MapPoint*> vpMapPoints;
    {
        std::unique_lock<std::mutex> lock(m_mutexMapPoints);
        vpMapPoints = m_mapPoints;
    }

    //for every mp, iterate through each frame that views it and increment counter
    for(std::vector<MapPoint*>::iterator vMPIt = vpMapPoints.begin(), vMPEnd = vpMapPoints.end(); vMPIt != vMPEnd; vMPIt++)
    {
        MapPoint* pMP = *vMPIt;

        if(!pMP)
            continue;
        //check if map point is bad (when does this happen?)
        //if(vMPIt->bad)
        //  continue;

        std::map<Frame*,size_t> frameViews = pMP->getFrameViews();

        for(std::map<Frame*,size_t>::iterator frameViewIt = frameViews.begin(), frameViewITEnd = frameViews.end(); frameViewIt!=frameViewITEnd; frameViewIt++)
        {
            //dont count itself
            if(frameViewIt->first->m_ID==m_ID)
                continue;
            //otherwise
            frameConnectionWeights[frameViewIt->first]++;
        }
    }
    if(frameConnectionWeights.empty())
        return;

    int maxMapPointViews=0;
    Frame* mostMapPointViews=NULL;
    int th = 15;

    std::vector<std::pair<int,Frame*> > vPairs;
    vPairs.reserve(frameConnectionWeights.size());

    //iterate through frame counter map in order to sort most connected (co-viewed map points) frames
    //to add a connection between frames, n of co-viewed map points must be above threshold
    for(std::map<Frame*,int>::iterator fCounterIT= frameConnectionWeights.begin(), fCounterITEnd=frameConnectionWeights.end(); fCounterIT != fCounterITEnd; fCounterIT++)
    {
        //number of points seen by each frame
        if(fCounterIT->second>maxMapPointViews)
        {
            maxMapPointViews=fCounterIT->second;
            mostMapPointViews=fCounterIT->first;
        }
        if(fCounterIT->second>=th)
        {
            vPairs.push_back(std::make_pair(fCounterIT->second,fCounterIT->first));
            (fCounterIT->first)->addConnection(this,fCounterIT->second);
        }
    }
    if(vPairs.empty() && mostMapPointViews != nullptr)
    {
        vPairs.push_back(std::make_pair(maxMapPointViews,mostMapPointViews));
        mostMapPointViews->addConnection(this,maxMapPointViews);
    }

    //sort by descending order, most connections (n connections corresponding frame)
    std::sort(vPairs.begin(),vPairs.end());
    std::list<Frame*> connectedFrames;
    std::list<int> connectedWeights;
    for(size_t i=0; i<vPairs.size();i++)
    {
        connectedFrames.push_front(vPairs[i].second);
        connectedWeights.push_front(vPairs[i].first);
    }

    {
        std::unique_lock<std::mutex> lockCon(m_mutexConnections);

        //Add information about connection, connected keyframes their corresponding weights in descending order
        // mspConnectedKeyFrames = spConnectedKeyFrames;
        m_connectionWeights = frameConnectionWeights;
        m_orderedConnectedFrames = std::vector<Frame*>(connectedFrames.begin(), connectedFrames.end());
        m_orderedConnectedWeights = std::vector<int>(connectedWeights.begin(), connectedWeights.end());

        if(m_firstConnection && m_ID != 0)
        {
            m_parent = m_orderedConnectedFrames.front();
            m_parent->addChild(this);
            m_firstConnection = false;
        }

    }



}

void Frame::addConnection(Frame *pFrame, const int &weight)
{
    {
        std::unique_lock<std::mutex> lock(m_mutexConnections);
        if(!m_connectionWeights.count(pFrame))
            m_connectionWeights[pFrame]=weight;

        else if(m_connectionWeights[pFrame] != weight)
            m_connectionWeights[pFrame]=weight;
        else
            return;
    }

        //update best connections
        updateConnections();
}

void Frame::addChild(Frame *pFrame)
{
    std::unique_lock<std::mutex> lock(m_mutexConnections);
    m_children.insert(pFrame);
}

void Frame::setR(const cv::Mat &orientation)
{
    orientation.copyTo(m_Rc);
    updateTransform();
}

void Frame::sett(const cv::Mat &position)
{
    position.copyTo(m_Tc);
    updateTransform();
}

void Frame::setPose(const cv::Mat &pose)
{
    pose.copyTo(m_posec);
    m_posec(cv::Range(0, 3), cv::Range(0, 3)).copyTo(m_Rc);
    m_posec(cv::Range(0, 3), cv::Range(3, 4)).copyTo(m_Tc);
    updateTransform();
}

void Frame::updateTransform(void)
{
    m_Rc.copyTo(m_posec.rowRange(0, 3).colRange(0, 3));
    m_Tc.copyTo(m_posec.rowRange(0, 3).col(3));

    for (int i = 0; i < m_posec.rows; i++)
        for (int j = 0; j < m_posec.cols; j++)
            m_vPose[i][j] = m_posec.at<float>(i, j);

    m_Rw = m_Rc.t();
    m_Tw = -m_Rw * m_Tc;
    m_Rw.copyTo(m_posew.rowRange(0, 3).colRange(0, 3));
    m_Tw.copyTo(m_posew.rowRange(0, 3).col(3));

    setMatricesD();
}

void Frame::setMatricesD()
{
    //TODO: REMOVE THIS BEFORE RELEASE
    //set matrices as std vector for debugging purpose
    m_vR = std::vector<std::vector<float> > (m_Rc.rows, std::vector<float>(m_Rc.cols));
    for (int i = 0; i < m_Rc.rows; i++)
        for (int j = 0; j < m_Rc.cols; j++)
            m_vR[i][j] = m_Rc.at<float>(i, j);

    m_vT = std::vector<std::vector<float> > (m_Tc.rows, std::vector<float>(m_Tc.cols));
    for (int i = 0; i < m_Tc.rows; i++)
        for (int j = 0; j < m_Tc.cols; j++)
            m_vT[i][j] = m_Tc.at<float>(i, j);

    m_vPose = std::vector<std::vector<float> > (m_posec.rows, std::vector<float>(m_posec.cols));
    for (int i = 0; i < m_posec.rows; i++)
        for (int j = 0; j < m_posec.cols; j++)
            m_vPose[i][j] = m_posec.at<float>(i, j);

}


void Frame::setFeaturesIdx2Grid()
{
    //the feature index is set to grid location
    float ptx,pty;
    for(size_t i = 0; i < m_keyPoints.size(); i++)
    {
        ptx = m_keyPoints[i].pt.x;
        pty = m_keyPoints[i].pt.y;

        size_t gridX = round(ptx * m_invWidthScale);
        size_t gridY = round(pty * m_invHeightScale);

        if(gridX < GRID_COLS && gridY < GRID_ROWS)
            m_grid[gridX][gridY].push_back(i);
    }
}

std::vector<size_t> Frame::getFeaturesIdxFromGrid(const cv::Point2i &imagePt)
{
    const int radius = m_searchRadius;
    std::vector<size_t> featureIdx;

    const int ptx = imagePt.x;
    const int pty = imagePt.y;

    const int xCellStart = std::max(0,static_cast<int>(std::floor((ptx - radius) * m_invWidthScale)));
    const int xCellStop  = std::min(GRID_COLS-1,std::max(static_cast<int>(std::ceil((ptx + radius) * m_invWidthScale)), 0));

    const int yCellStart = std::max(0,static_cast<int>(floor((pty - radius) * m_invHeightScale)));
    const int yCellStop  = std::min(GRID_ROWS-1,std::max(static_cast<int>(std::ceil((pty + radius) * m_invHeightScale)), 0));

    if((xCellStop>= GRID_COLS) || (yCellStop >= GRID_ROWS))
        std::cout << "out of bounds" << std::endl;

    for(size_t i = xCellStart; i <= xCellStop; i++)
    {
        for(size_t j = yCellStart; j <= yCellStop; j++)
        {
            if(m_grid[i][j].empty())
                continue;
            else
            {
                for(auto& idx : m_grid[i][j])
                    featureIdx.push_back(idx);
            }

        }
    }
    return featureIdx;
}

std::vector<Frame *> Frame::getConnectedFrames()
{
    std::unique_lock<std::mutex> lock(m_mutexConnections);
    return m_orderedConnectedFrames;
}

std::vector<Frame *> Frame::getNConnectedFrames(const int N)
{
    std::unique_lock<std::mutex> lock(m_mutexConnections);
    if(m_orderedConnectedFrames.size() < N)
        return m_orderedConnectedFrames;
    else
        return std::vector<Frame*>(m_orderedConnectedFrames.begin(), m_orderedConnectedFrames.begin() + N);
}

float Frame::computeMedianDepth()
{
    std::vector<MapPoint*> mapPoints;
    cv::Mat pose;
    {
        std::unique_lock<std::mutex> lockMPs(m_mutexMapPoints);
        std::unique_lock<std::mutex> lockPose(m_mutexPose);
        pose = m_posec.clone();
        mapPoints = m_mapPoints;
    }

    //depth with respect to camera pose (only last row of R is necessary)
    cv::Mat camRrow = m_posec.row(2).colRange(0, 3);
    camRrow = camRrow.t();
    float t = m_posec.at<float>(2, 3);

    std::vector<float> depths;
    depths.reserve(N);
    for(size_t i = 0; i < N; i++)
    {
        if(mapPoints[i] == nullptr)
            continue;
        cv::Mat p3D = mapPoints[i]->getPosition();
        float z = camRrow.dot(p3D) + t;
        depths.push_back(z);
    }

    //sort and take median
    if (depths.empty()) return 0.0f;
    std::sort(depths.begin(),depths.end());
    return depths[(depths.size()-1)/2];

}

bool Frame::removeMapPoint(MapPoint *mapPoint)
{
    std::unique_lock<std::mutex> lock(m_mutexFeatures);
    if(mapPoint->getFrameViews().count(this))
    {
        size_t idx = mapPoint->getFrameViews().at(this);
        m_mapPoints[idx] = nullptr;
        return true;
    }
    return false;
}

bool Frame::removeMapPoint(const size_t idx)
{
    std::unique_lock<std::mutex> lock(m_mutexFeatures);
    if(idx < m_mapPoints.size())
    {
        m_mapPoints[idx]=nullptr;
        return true;
    }
    return false;
}

bool Frame::checkReprojectionError()
{
    int nBad = 0;
    int nCnt = 0;
    const float fx = m_camParams.fx;
    const float fy = m_camParams.fy;
    const float cx = m_camParams.cx;
    const float cy = m_camParams.cy;

    //run through map points, project them into screen space and check error with current descriptors
    for(size_t i = 0; i < N; i++)
    {
        if(m_mapPoints[i] == nullptr)
            continue;

        cv::KeyPoint p2D = m_keyPoints[i];
        const cv::Mat p3Dw = m_mapPoints[i]->getPosition();
        const cv::Mat p3Dc = m_Rc * p3Dw + m_Tc;

        const float xc = p3Dc.at<float>(0);
        const float yc = p3Dc.at<float>(1);
        const float invzc = 1.0f/p3Dc.at<float>(2);


        const float u = fx*xc*invzc + cx;
        const float v = fy*yc*invzc + cy;

        const float error = (u - p2D.pt.x)*(u - p2D.pt.x)+(v - p2D.pt.y)*(v - p2D.pt.y);
        if(error > 5.991f)
            nBad++;
        nCnt++;
    }

    if (nCnt > 0 && static_cast<float>(nBad) / nCnt > 0.1f)
        return false;
    return true;
}

std::vector<MapPoint *> Frame::getMapPoints(void)
{
    std::unique_lock<std::mutex> lock(m_mutexMapPoints);
    return m_mapPoints;
}

void Frame::setDescriptorMapPoint(const size_t descriptor, MapPoint* mp)
{
    m_descriptor_mapPoint[descriptor] = mp;
}

MapPoint *Frame::getDescriptorMapPoint(size_t descriptor)
{
    std::unique_lock<std::mutex> lock(m_mutexMapPoints);
    if(m_descriptor_mapPoint.count(descriptor) > 0)
        return m_descriptor_mapPoint[descriptor];
    else
        return nullptr;
}

void Frame::replaceMapPoint(const size_t idx, MapPoint *mapPoint)
{
    m_mapPoints[idx] = mapPoint;
}

void Frame::setDescriptorIdx(const size_t descriptor, const size_t idx)
{
    m_descriptor_idx[descriptor] = idx;
}

bool Frame::getDescriptorIdx(const size_t descriptor, size_t &idx)
{
    if(m_descriptor_idx.count(descriptor) > 0)
    {
        idx = m_descriptor_idx[descriptor];
        return true;
    }
    else
        return false;
}

MapPoint *Frame::getMapPoint(const size_t idx)
{
    std::unique_lock<std::mutex> lock(m_mutexMapPoints);
    if(idx < m_mapPoints.size())
    {
        MapPoint* mapPoint = m_mapPoints[idx];
        if(mapPoint)
            return mapPoint;
        return nullptr;
    }
    return nullptr;
}

const unsigned int Frame::checkRepeatedFeatures(Frame *frame)
{

    return 0;
}

void Frame::removeFrame()
{
    //check if first frame
    if(m_ID == 0)
        return;


    //iterate through connected frame weights and remove self from that
    for(std::map<Frame*,int>::iterator it = m_connectionWeights.begin(), itEnd = m_connectionWeights.end(); it != itEnd; it++)
        it->first->removeConnection(this);

    //for every map point seen from this frame, remove this frame view
    for(size_t i = 0; i < N; i++)
    {
        if(m_mapPoints[i] != nullptr)
            m_mapPoints[i]->removeFrameView(this);
    }

    //if there is no parent, attempt to get one
    if(m_parent == nullptr)
        setConnections();

    if(m_parent == nullptr)
        return;

    //Readjust graph of parent child connections
    {
        std::unique_lock<std::mutex> lockConnections(m_mutexConnections);
        std::unique_lock<std::mutex> lockFeatures(m_mutexFeatures);

        //clear stored connected frames and respective weights
        m_orderedConnectedFrames.clear();
        m_orderedConnectedWeights.clear();

        std::set<Frame*> parentCandidates;
        parentCandidates.insert(m_parent);

        //iterate through children
        while(!m_children.empty())
        {
            int max = -1;
            Frame* child;
            Frame* parent;
            bool needToReconnect = false;

            //for each child
            for(std::set<Frame*>::iterator it = m_children.begin(), itEnd = m_children.end(); it != itEnd; it++)
            {
                Frame* childFrame = *it;
                if(childFrame == nullptr)
                    continue;

                //iterate through connected frames
                std::vector<Frame*> connectedFrames = childFrame->getConnectedFrames();
                for(size_t i = 0; i < connectedFrames.size(); i++)
                {
                    //compare if any connected frame is also a parent candidate (parent candidates change in this loop)
                    for(std::set<Frame*>::iterator itP = parentCandidates.begin(), itPEnd = parentCandidates.end(); itP != itPEnd; itP++)
                    {
                        //check that parent candidate id is one of connected frames id
                        if(connectedFrames[i]->getID() == (*itP)->getID())
                        {
                            //because there could be more than a parent candidate, take one with highest weight
                            int w = childFrame->getConnectedWeight(connectedFrames[i]);
                            if(w>max)
                            {
                                child = childFrame;
                                parent = connectedFrames[i];
                                max = w;
                                needToReconnect = true;
                            }
                        }
                    }
                }
            }

            //need to reconnect:
            //child's parent is this frame's parent
            //child is new candidate parent for other children
            //child is no longer child of this frame (erase)
            if(needToReconnect)
            {
                child->setParent(parent);
                parentCandidates.insert(child);
                m_children.erase(child);

            }
            else
                break;


        }
        //In case no reconnection to any parent candidate
        //reconnect to parent
        if(!m_children.empty())
        {
            for(std::set<Frame*>::iterator it = m_children.begin(), itEnd = m_children.end(); it != itEnd; it++)
            {
                (*it)->setParent(m_parent);
            }
        }
        //now remove this frame from parent's child
        if(m_parent != nullptr)
            m_parent->removeChild(this);
    }

}

int Frame::getConnectedWeight(Frame *frame)
{
    std::unique_lock<std::mutex> lock(m_mutexConnections);
    if(m_connectionWeights.count(frame))
        return m_connectionWeights[frame];
    else
        return 0;
}

void Frame::setParent(Frame* frame)
{
    std::unique_lock<std::mutex> lock(m_mutexConnections);
    m_parent = frame;
    frame->addChild(this);
}

void Frame::removeConnection(Frame* frame)
{
    bool needUpdate = false;
    {
        std::unique_lock<std::mutex> lock(m_mutexConnections);
        if (m_connectionWeights.count(frame))
        {
            m_connectionWeights.erase(frame);
            needUpdate = true;
        }
    }
    if(needUpdate)
        updateConnections();
}

void Frame::updateConnections()
{
    //update best connections
    {
        std::unique_lock<std::mutex> lock(m_mutexConnections);
        std::vector<std::pair<int, Frame *> > frameWeights;
        frameWeights.reserve(m_connectionWeights.size());
        for (std::map<Frame *, int>::iterator it = m_connectionWeights.begin(), itEnd = m_connectionWeights.end();
             it != itEnd; it++)
            frameWeights.push_back(std::make_pair(it->second, it->first));

        sort(frameWeights.begin(), frameWeights.end());
        std::list<Frame *> connectedFrames;
        std::list<size_t> connectedWeights;
        for (size_t i = 0; i < frameWeights.size(); i++)
        {
            connectedFrames.push_front(frameWeights[i].second);
            connectedWeights.push_front(frameWeights[i].first);
        }

        m_orderedConnectedFrames = std::vector<Frame *>(connectedFrames.begin(), connectedFrames.end());
        m_orderedConnectedWeights = std::vector<int>(connectedWeights.begin(), connectedWeights.end());
    }

}

void Frame::removeChild(Frame *frame)
{
    std::unique_lock<std::mutex> lock(m_mutexConnections);
    m_children.erase(frame);

}


//copy constructor
Frame::Frame(const Frame& other)
        : m_ID(other.m_ID),
          m_BA_ID(other.m_BA_ID),
          m_Fuse_ID(other.m_Fuse_ID),
          m_initialized(other.m_initialized),
          m_trackedCnt(other.m_trackedCnt),
          m_previousFrame(other.m_previousFrame),
          m_camParams(other.m_camParams),
          N(other.N),
          m_outliers(other.m_outliers),
          m_mapPoints(other.m_mapPoints),
          m_keyPoints(other.m_keyPoints),
          m_descriptors(other.m_descriptors.clone()),
          m_vtxDescriptors(other.m_vtxDescriptors),
          m_descriptor_mapPoint(other.m_descriptor_mapPoint),
          m_descriptor_idx(other.m_descriptor_idx),
          m_matches(other.m_matches),
          m_initMatches(other.m_initMatches),
          m_orderedConnectedFrames(other.m_orderedConnectedFrames),
          m_orderedConnectedWeights(other.m_orderedConnectedWeights),
          m_firstConnection(other.m_firstConnection),
          m_parent(other.m_parent),
          m_connectionWeights(other.m_connectionWeights),
          m_children(other.m_children),
          m_Rc(other.m_Rc.clone()),
          m_Tc(other.m_Tc.clone()),
          m_posec(other.m_posec.clone()),
          m_Rw(other.m_Rw.clone()),
          m_Tw(other.m_Tw.clone()),
          m_posew(other.m_posew.clone()),
          m_vR(other.m_vR),
          m_vT(other.m_vT),
          m_vPose(other.m_vPose),
          m_searchRadius(other.m_searchRadius),
          m_error(other.m_error),
          m_timeStamp(other.m_timeStamp),
          m_slamParams(other.m_slamParams),
          m_virtualCamPose(other.m_virtualCamPose)
{
    for (size_t i = 0; i < GRID_COLS; i++)
        for (size_t j = 0; j < GRID_ROWS; j++)
            m_grid[i][j] = other.m_grid[i][j];
    // Mutexes (m_mutexPose, etc.) are default-constructed, not copied
}
Frame::Frame(const Frame* other) {
    if (!other) return; // Handle null safely

    m_ID = other->m_ID;
    m_BA_ID = other->m_BA_ID;
    m_Fuse_ID = other->m_Fuse_ID;
    m_initialized = other->m_initialized;
    m_trackedCnt = other->m_trackedCnt;
    m_previousFrame = other->m_previousFrame;
    m_camParams = other->m_camParams;
    N = other->N;
    m_outliers = other->m_outliers;
    m_mapPoints = other->m_mapPoints;
    m_keyPoints = other->m_keyPoints;
    m_descriptors = other->m_descriptors.clone();
    m_vtxDescriptors = other->m_vtxDescriptors;
    m_descriptor_mapPoint = other->m_descriptor_mapPoint;
    m_descriptor_idx = other->m_descriptor_idx;
    m_matches = other->m_matches;
    m_initMatches = other->m_initMatches;
    m_orderedConnectedFrames = other->m_orderedConnectedFrames;
    m_orderedConnectedWeights = other->m_orderedConnectedWeights;
    m_firstConnection = other->m_firstConnection;
    m_parent = other->m_parent;
    m_connectionWeights = other->m_connectionWeights;
    m_children = other->m_children;
    m_Rc = other->m_Rc.clone();
    m_Tc = other->m_Tc.clone();
    m_posec = other->m_posec.clone();
    m_Rw = other->m_Rw.clone();
    m_Tw = other->m_Tw.clone();
    m_posew = other->m_posew.clone();
    m_vR = other->m_vR;
    m_vT = other->m_vT;
    m_vPose = other->m_vPose;
    m_searchRadius = other->m_searchRadius;
    m_error = other->m_error;
    m_timeStamp = other->m_timeStamp;
    m_slamParams = other->m_slamParams;
    m_virtualCamPose = other->m_virtualCamPose;

    for (size_t i = 0; i < GRID_COLS; i++)
        for (size_t j = 0; j < GRID_ROWS; j++)
            m_grid[i][j] = other->m_grid[i][j];
}

//copy assignment
Frame& Frame::operator=(const Frame& other) {
    if (this != &other) {
        m_ID = other.m_ID;
        m_BA_ID = other.m_BA_ID;
        m_Fuse_ID = other.m_Fuse_ID;
        m_initialized = other.m_initialized;
        m_trackedCnt = other.m_trackedCnt;
        m_previousFrame = other.m_previousFrame;
        m_camParams = other.m_camParams;
        N = other.N;
        m_outliers = other.m_outliers;
        m_mapPoints = other.m_mapPoints;
        m_keyPoints = other.m_keyPoints;
        m_descriptors = other.m_descriptors.clone();
        m_vtxDescriptors = other.m_vtxDescriptors;
        m_descriptor_mapPoint = other.m_descriptor_mapPoint;
        m_descriptor_idx = other.m_descriptor_idx;
        m_matches = other.m_matches;
        m_initMatches = other.m_initMatches;
        m_orderedConnectedFrames = other.m_orderedConnectedFrames;
        m_orderedConnectedWeights = other.m_orderedConnectedWeights;
        m_firstConnection = other.m_firstConnection;
        m_parent = other.m_parent;
        m_connectionWeights = other.m_connectionWeights;
        m_children = other.m_children;
        m_Rc = other.m_Rc.clone();
        m_Tc = other.m_Tc.clone();
        m_posec = other.m_posec.clone();
        m_Rw = other.m_Rw.clone();
        m_Tw = other.m_Tw.clone();
        m_posew = other.m_posew.clone();
        m_vR = other.m_vR;
        m_vT = other.m_vT;
        m_vPose = other.m_vPose;
        m_searchRadius = other.m_searchRadius;
        m_error = other.m_error;
        m_timeStamp = other.m_timeStamp;
        m_slamParams = other.m_slamParams;
        m_virtualCamPose = other.m_virtualCamPose;

        for (size_t i = 0; i < GRID_COLS; i++)
            for (size_t j = 0; j < GRID_ROWS; j++)
                m_grid[i][j] = other.m_grid[i][j];
    }
    return *this;
}
Frame& Frame::operator=(const Frame* other) {
    if (!other || this == other) return *this; // Handle null and self-assignment

    m_ID = other->m_ID;
    m_BA_ID = other->m_BA_ID;
    m_Fuse_ID = other->m_Fuse_ID;
    m_initialized = other->m_initialized;
    m_trackedCnt = other->m_trackedCnt;
    m_previousFrame = other->m_previousFrame;
    m_camParams = other->m_camParams;
    N = other->N;
    m_outliers = other->m_outliers;
    m_mapPoints = other->m_mapPoints;
    m_keyPoints = other->m_keyPoints;
    m_descriptors = other->m_descriptors.clone();
    m_vtxDescriptors = other->m_vtxDescriptors;
    m_descriptor_mapPoint = other->m_descriptor_mapPoint;
    m_descriptor_idx = other->m_descriptor_idx;
    m_matches = other->m_matches;
    m_initMatches = other->m_initMatches;
    m_orderedConnectedFrames = other->m_orderedConnectedFrames;
    m_orderedConnectedWeights = other->m_orderedConnectedWeights;
    m_firstConnection = other->m_firstConnection;
    m_parent = other->m_parent;
    m_connectionWeights = other->m_connectionWeights;
    m_children = other->m_children;
    m_Rc = other->m_Rc.clone();
    m_Tc = other->m_Tc.clone();
    m_posec = other->m_posec.clone();
    m_Rw = other->m_Rw.clone();
    m_Tw = other->m_Tw.clone();
    m_posew = other->m_posew.clone();
    m_vR = other->m_vR;
    m_vT = other->m_vT;
    m_vPose = other->m_vPose;
    m_searchRadius = other->m_searchRadius;
    m_error = other->m_error;
    m_timeStamp = other->m_timeStamp;
    m_slamParams = other->m_slamParams;
    m_virtualCamPose = other->m_virtualCamPose;

    for (size_t i = 0; i < GRID_COLS; i++)
        for (size_t j = 0; j < GRID_ROWS; j++)
            m_grid[i][j] = other->m_grid[i][j];

    return *this;
}

void TFrame::setPose(const cv::Mat &pose)
{
    pose.copyTo(m_posec);
    m_posec(cv::Range(0, 3), cv::Range(0, 3)).copyTo(m_Rc);
    m_posec(cv::Range(0, 3), cv::Range(3, 4)).copyTo(m_Tc);
    updateTransform();
}

void TFrame::updateTransform()
{
    m_Rc.copyTo(m_posec.rowRange(0, 3).colRange(0, 3));
    m_Tc.copyTo(m_posec.rowRange(0, 3).col(3));

    for (int i = 0; i < m_posec.rows; i++)
        for (int j = 0; j < m_posec.cols; j++)
            m_vPose[i][j] = m_posec.at<float>(i, j);

    m_Rw = m_Rc.t();
    m_Tw = -m_Rw * m_Tc;
    m_Rw.copyTo(m_posew.rowRange(0, 3).colRange(0, 3));
    m_Tw.copyTo(m_posew.rowRange(0, 3).col(3));

    setMatricesD();
}

void TFrame::setMatricesD()
{
    //TODO: REMOVE THIS BEFORE RELEASE
    //set matrices as std vector for debugging purpose
    m_vR = std::vector<std::vector<float> > (m_Rc.rows, std::vector<float>(m_Rc.cols));
    for (int i = 0; i < m_Rc.rows; i++)
        for (int j = 0; j < m_Rc.cols; j++)
            m_vR[i][j] = m_Rc.at<float>(i, j);

    m_vT = std::vector<std::vector<float> > (m_Tc.rows, std::vector<float>(m_Tc.cols));
    for (int i = 0; i < m_Tc.rows; i++)
        for (int j = 0; j < m_Tc.cols; j++)
            m_vT[i][j] = m_Tc.at<float>(i, j);

    m_vPose = std::vector<std::vector<float> > (m_posec.rows, std::vector<float>(m_posec.cols));
    for (int i = 0; i < m_posec.rows; i++)
        for (int j = 0; j < m_posec.cols; j++)
            m_vPose[i][j] = m_posec.at<float>(i, j);

}

void TFrame::initialize()
{
    m_Rc = cv::Mat::eye(3, 3, CV_32F);
    m_Tc = cv::Mat::zeros(3, 1, CV_32F);
    m_posec = cv::Mat::eye(4, 4, CV_32F);

    m_Rw = cv::Mat::eye(3, 3, CV_32F);
    m_Tw = cv::Mat::zeros(3, 1, CV_32F);
    m_posew = cv::Mat::eye(4, 4, CV_32F);

    m_vR = std::vector<std::vector<float> > (m_Rc.rows, std::vector<float>(m_Rc.cols));
    m_vT = std::vector<std::vector<float> > (m_Tc.rows, std::vector<float>(m_Tc.cols));
    m_vPose = std::vector<std::vector<float> > (m_posec.rows, std::vector<float>(m_posec.cols));
}

int Matcher::matchAndAddAll(Frame &newFrame, Frame *oldFrame)
{
    //TODO: time this, see how slow it is
    //TODO: time this, see how slow it is
    int matches = 0;
    {
        std::unique_lock<std::mutex> lock(MapPoint::mMapPointsGlobalMutex);

        std::vector<MapPoint*> oldMapPoints = oldFrame->getMapPoints();
        std::vector<MapPoint*> newMapPoints = newFrame.getMapPoints();
        std::vector<size_t> newFrameDescriptors = newFrame.getSimDescriptors();

        for (size_t i = 0; i < oldMapPoints.size(); i++)
        {
            if (oldMapPoints[i] != nullptr)
            {
                //iterate through reference frame map points and retrieve frame views
                std::map<Frame *, size_t> frameViews = oldMapPoints[i]->getFrameViews();
                size_t oldFrameIdx;
                //get the corresponding stored index for this mappoint for this frame
                auto it = frameViews.find(oldFrame);
                if(it != frameViews.end())
                    oldFrameIdx = it->second;
                else
                    continue;

                auto oldFrameDescriptor = oldFrame->getSimDescriptors()[oldFrameIdx];

                for (size_t j = 0; j < newFrameDescriptors.size(); j++)
                {
                    if (oldFrameDescriptor == newFrameDescriptors[j])
                    {
                        newFrame.addMapPoint(oldMapPoints[i],j);
                        matches++;
                    }
                }
            }
        }
    }
    return matches;
}

int Matcher::matchAndAddByProjection(Frame *newFrame, Frame *oldFrame)
{
    int nMatches = 0;
    //from cam2 (new) to cam1(old)
    const cv::Mat R2c = newFrame->getRc();
    const cv::Mat t2c = newFrame->getTc();
    const cv::Mat t2w = newFrame->getTw();

    const cv::Mat R1c = oldFrame->getRc();
    const cv::Mat t1c = oldFrame->getTc();

    //cam2 to cam1 position
    const cv::Mat t12 = R1c * t2w + t1c;

    const float fx = newFrame->getCamPrjParams().fx;
    const float fy = newFrame->getCamPrjParams().fy;
    const float cx = newFrame->getCamPrjParams().cx;
    const float cy = newFrame->getCamPrjParams().cy;

    std::vector<MapPoint*> mapPoints = oldFrame->getMapPoints();
    for(size_t i = 0; i < oldFrame->getN(); i++)
    {
        MapPoint* mapPoint = mapPoints[i];
        if(mapPoint != nullptr)
        {
            //project old frame's map point (in world coords.) to new frame coords.
            const cv::Mat p3Dw = mapPoint->getPosition();
            const cv::Mat p3D2c = R2c * p3Dw + t2c;

            const float xc = p3D2c.at<float>(0);
            const float yc = p3D2c.at<float>(1);
            const float invzc = 1.0f/p3D2c.at<float>(2);

            const float u = fx*xc*invzc + cx;
            const float v = fy*yc*invzc + cy;

            if(invzc < 0)
                continue;

            //get vector of indexes of features/descriptors in grid area
            std::vector<size_t> idxInArea = newFrame->getFeaturesIdxFromGrid(cv::Point2i(u, v));
            if(idxInArea.empty())
                continue;

            size_t mapPointDescriptor = mapPoint->getDescriptor();

            //search for best fit descriptors from found indexes in area
            for(size_t i = 0; i < idxInArea.size(); i++)
            {
                size_t newFramePtDescriptor = newFrame->getSimDescriptors()[idxInArea[i]];

                if(mapPointDescriptor == newFramePtDescriptor)
                {
                    newFrame->addMapPoint(mapPoint,idxInArea[i]);
                    nMatches++;
                    break;
                }
            }
        }
    }
    return nMatches;
}

int Matcher::matchAndAddByProjection(Frame *newFrame, const  std::vector<MapPoint*> & mapPoints)
{
    //compare 3D map points projected into newFrame image plane, check if any descriptors in same grid match
    int N = mapPoints.size();
    if(N <= 0)
        return 0;

    int nMatches = 0;

    const float w  = newFrame->getCamPrjParams().width;
    const float h  = newFrame->getCamPrjParams().height;
    const float fx = newFrame->getCamPrjParams().fx;
    const float fy = newFrame->getCamPrjParams().fy;
    const float cx = newFrame->getCamPrjParams().cx;
    const float cy = newFrame->getCamPrjParams().cy;

    const cv::Mat R2c = newFrame->getRc();
    const cv::Mat t2c = newFrame->getTc();
    const cv::Mat t2w = newFrame->getTw();

    for(size_t i = 0; i < N; i++)
    {
        MapPoint* mapPoint = mapPoints[i];
        if((mapPoint == nullptr) || (mapPoint->isBad()))
            continue;

        //convert map point world pos to newFrames frame
        cv::Mat p3Dw = mapPoint->getPosition();
        cv::Mat p3Df = R2c * p3Dw + t2c;

        //check that depth is positive
        if(p3Df.at<float>(2) < 0.0f)
            continue;

        //get coord. in frame space
        const float invz = 1.0f/p3Df.at<float>(2);
        const float x = p3Df.at<float>(0);
        const float y = p3Df.at<float>(1);

        //get screen coord.
        const float u = fx*x*invz + cx;
        const float v = fy*y*invz + cy;

        //check that point is within image
        if((w < u) || (h < v))
            continue;

        cv::Mat p3Dv = t2w - p3Dw;
        float p3DDist = (float)cv::norm(p3Dv);

        //TODO: check if within distance
        //if((p3DDist < minDist) || (p3DDist > maxDist))
        //  continue;


        const std::vector<size_t> idxInArea = newFrame->getFeaturesIdxFromGrid(cv::Point2i(u, v));
        if(idxInArea.empty())
            continue;

        size_t mapPointDescriptor = mapPoint->getDescriptor();
        //search for best fit descriptors from found indexes in area
        for(size_t i = 0; i < idxInArea.size(); i++)
        {
            size_t newFramePtDescriptor = newFrame->getSimDescriptors()[idxInArea[i]];

            if(mapPointDescriptor == newFramePtDescriptor)
            {
                newFrame->addMapPoint(mapPoint,idxInArea[i]);
                nMatches++;
            }
        }
    }
    return nMatches;
}

int Matcher::matchByEpipolarDist(Frame *f1, Frame *f2, cv::Mat F, std::vector<PtPair> &matches)
{
    int nChecked = 0;
    int nMatches = 0;

    std::vector<size_t> f1Descriptors = f1->getSimDescriptors();
    std::vector<size_t> f2Descriptors = f2->getSimDescriptors();
    size_t N1 = f1->getN();
    size_t N2 = f2->getN();
    std::vector<MapPoint*> f1MPs = f1->getMapPoints();
    std::vector<MapPoint*> f2MPs = f2->getMapPoints();


    for(size_t i = 0; i < N1; i++)
    {
        if(f1MPs[i] != nullptr)
            continue;
        for(size_t j = 0; j < N2; j++)
        {
            if(f2MPs[j] != nullptr)
                continue;

            if( f1Descriptors[i] == f2Descriptors[j])
            {
                nChecked++;
                if (Matcher::checkEpipolarDistance(f1->getKeyPoints()[i].pt, f2->getKeyPoints()[j].pt, F, 2.0f))
                {
                    matches.push_back(PtPair(i,j));
                    nMatches++;
                }

            }
        }
    }

    Logger<std::string>::LogInfoI("Total checked pts : " + std::to_string(nChecked));
    return nMatches;
}

bool Matcher::checkEpipolarDistance(cv::Point2i p1, cv::Point2i p2, const cv::Mat& F, float distance)
{
    cv::Mat FT = F.t();
    const float f11 = FT.at<float>(0,0);
    const float f12 = FT.at<float>(0,1);
    const float f13 = FT.at<float>(0,2);
    const float f21 = FT.at<float>(1,0);
    const float f22 = FT.at<float>(1,1);
    const float f23 = FT.at<float>(1,2);
    const float f31 = FT.at<float>(2,0);
    const float f32 = FT.at<float>(2,1);
    const float f33 = FT.at<float>(2,2);

    const float u1 = p1.x;
    const float v1 = p1.y;
    const float u2 = p2.x;
    const float v2 = p2.y;
    // Reprojection error in second image
    // l2=F12Tx1=(a2,b2,c2)
    const float a2 = f11*u1+f12*v1+f13;
    const float b2 = f21*u1+f22*v1+f23;
    const float c2 = f31*u1+f32*v1+f33;
    const float num2 = a2*u2+b2*v2+c2;
    const float squareDist1 = num2*num2/(a2*a2+b2*b2);
    //std::cout << "d1: " + std::to_string(squareDist1) << std::endl;

    // Reprojection error in second image
    // l1 =x2tF21=(a1,b1,c1)

    const float a1 = f11*u2+f21*v2+f31;
    const float b1 = f12*u2+f22*v2+f32;
    const float c1 = f13*u2+f23*v2+f33;
    const float num1 = a1*u1+b1*v1+c1;
    const float squareDist2 = num1*num1/(a1*a1+b1*b1);
    //std::cout << "d2: " + std::to_string(squareDist2) << std::endl;


    return (squareDist1<3.84 && squareDist2<3.84);

}

int Matcher::fuseByProjection(Frame *newFrame, const std::vector<MapPoint *> &mapPoints,std::vector<PtPair>& matches)
{
    //Here duplicated map points are to be replaced:
    //check this frame's map points wrt other frames map points
    //this frame's map points projected into other frame's image plane, check if any descriptors in same grid area match
    //if descriptors match, check if feature descriptor has associated map point
    //decide which map point to keep: this frame's or other frame's, the one with most frame views
    //if no associated map point, add the map point to the feature in the grid in this frame

    int N = mapPoints.size();
    if(N <= 0)
        return 0;

    int nMatches = 0;

    const float w  = newFrame->getCamPrjParams().width;
    const float h  = newFrame->getCamPrjParams().height;
    const float fx = newFrame->getCamPrjParams().fx;
    const float fy = newFrame->getCamPrjParams().fy;
    const float cx = newFrame->getCamPrjParams().cx;
    const float cy = newFrame->getCamPrjParams().cy;

    const cv::Mat R2c = newFrame->getRc();
    const cv::Mat t2c = newFrame->getTc();
    const cv::Mat t2w = newFrame->getTw();

    for(size_t i = 0; i < N; i++)
    {
        MapPoint* mapPoint = mapPoints[i];

        if((mapPoint == nullptr) || (mapPoint->isBad()))
            continue;

        //convert map point world pos to newFrames frame
        cv::Mat p3Dw = mapPoint->getPosition();
        cv::Mat p3Df = R2c * p3Dw + t2c;

        //check that depth is positive
        if(p3Df.at<float>(2) < 0.0f)
            continue;

        //get coord. in frame space
        const float invz = 1.0f/p3Df.at<float>(2);
        const float x = p3Df.at<float>(0);
        const float y = p3Df.at<float>(1);

        //get screen coord.
        const float u = fx*x*invz + cx;
        const float v = fy*y*invz + cy;

        //check that point is within image
        if((w < u) || (h < v))
            continue;

        cv::Mat p3Dv = t2w - p3Dw;
        float p3DDist = (float)cv::norm(p3Dv);

        //TODO: check if within distance
        //if((p3DDist < minDist) || (p3DDist > maxDist))
        //  continue;


        const std::vector<size_t> idxInArea = newFrame->getFeaturesIdxFromGrid(cv::Point2i(u, v));
        if(idxInArea.empty())
            continue;

        size_t mapPointDescriptor = mapPoint->getDescriptor();
        //search for best fit descriptors from found indexes in area
        for(size_t i = 0; i < idxInArea.size(); i++)
        {
            size_t newFramePtDescriptor = newFrame->getSimDescriptors()[idxInArea[i]];
            //matched by projection AND feature ID
            if(mapPointDescriptor == newFramePtDescriptor)
            {
                //fetch corresponding map point of the matched feature (feature's idx is map point's idx)
                size_t mpIdx;
                newFrame->getDescriptorIdx(newFramePtDescriptor, mpIdx);
                MapPoint* targetMapPoint = newFrame->getDescriptorMapPoint(newFramePtDescriptor);
                if(targetMapPoint != nullptr)
                {
                    if(mapPoint == targetMapPoint)
                        continue;
                    //Fuse points (check map point that has most views)
                    if(mapPoint->getNFrameViews() > targetMapPoint->getNFrameViews())
                    {
                        mapPoint->replace(targetMapPoint);
                    }
                    else
                    {
                        targetMapPoint->replace(mapPoint);
                    }
                    mapPoint->setPointType(2);
                    targetMapPoint->setPointType(2);

                }
                else
                {
                    size_t idx = 0;
                    newFrame->getDescriptorIdx(newFramePtDescriptor,idx);
                    newFrame->addMapPoint(mapPoint,idx);

                }
                //continue;


                nMatches++;
            }
        }

    }
    return nMatches;
}

int Matcher::matchOldFrames(Frame *newFrame, std::vector<Frame *> &oldFrames, size_t maxFrames, Frame* bestFrame)
{
    //search all previous keyframes, start from most recent
    int maxMatchCount = -1;
    int bestMatchIndex = -1;
    bool bestMatchFound = false;

    auto newDescriptors = newFrame->getSimDescriptors();
    std::set<int> newSet (newDescriptors.begin(),newDescriptors.end());

    //search max number of frames, or if available frames is less, then use that
    size_t idSub = 0;
    if (oldFrames.size() > maxFrames)
        idSub = oldFrames.size() - maxFrames;

    for(size_t i = oldFrames.size(); i>idSub; i--)
    {
        Logger<std::string>::LogInfoII("For re-localization, try past frame: " + std::to_string(oldFrames[i-1]->getID()));
        auto oldDescriptors = oldFrames[i-1]->getSimDescriptors();
        std::set<int> oldSet (oldDescriptors.begin(),oldDescriptors.end());
        int matchCount = 0;
        for(const int& element : newSet)
        {
            if(oldSet.find(element) != oldSet.end())
            {
                ++matchCount;
            }
        }

        if(matchCount>maxMatchCount)
        {
            maxMatchCount = matchCount;
            bestMatchIndex = i-1;
            bestMatchFound = true;
            Logger<std::string>::LogInfoII("Best frame for relocalization: " + std::to_string(oldFrames[bestMatchIndex]->getID()) + ", matches: " + std::to_string(matchCount));
        }
    }

    if(bestMatchFound)
    {
        bestFrame = oldFrames[bestMatchIndex];
        return matchAndAddAll(*newFrame, bestFrame);
    }

    return 0;
}
