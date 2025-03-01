
#include <set>

#include "mapping.h"
#include "optimizer.h"

static inline cv::Mat computeF12(Frame* f1, Frame* f2)
{
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = f1->getCamPrjParams().fx;
    K.at<float>(1,1) = f1->getCamPrjParams().fy;
    K.at<float>(0,2) = f1->getCamPrjParams().cx;
    K.at<float>(1,2) = f1->getCamPrjParams().cy;

    const cv::Mat R1 = f1->getRc();
    const cv::Mat t1 = f1->getTc();
    const cv::Mat R2 = f2->getRc();
    const cv::Mat t2 = f2->getTc();
    cv::Mat R1t = -R1.t()*t1;
    cv::Mat R2t = -R2.t()*t2;

    //compute F from 2 to 1
    //R2 is in cam space, convert to world space
    //R1 converts R2 (world) to its frame (R1)

    cv::Mat R12 = R1*R2.t();
    cv::Mat t12 = -R12 * t2 + t1;

    cv::Mat t12x = cv::Mat::zeros(3,3,CV_32F);
    t12x.at<float>(0,1) = -t12.at<float>(2);
    t12x.at<float>(0,2) =  t12.at<float>(1);
    t12x.at<float>(1,2) = -t12.at<float>(0);
    t12x.at<float>(1,0) =  t12.at<float>(2);
    t12x.at<float>(2,0) = -t12.at<float>(1);
    t12x.at<float>(2,1) =  t12.at<float>(0);
    return K.t().inv()*t12x*R12*K.inv();

}

void Mapping::run()
{
    while(!m_stop)
    {
        while (m_mappingState == MappingStates::MAP_INITIALIZED)
        {

            if (newFrameAvailable())
            {
                auto mapperTimerStart = std::chrono::high_resolution_clock::now();
                m_tracker->clearUpdateFlag();
                m_updated = false;


                auto timerStartProcessNewFrames = std::chrono::high_resolution_clock::now();
                processNewFrames();
                deleteMapPoints();
                auto timerEndProcessNewFrames = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> ProcessNewFramesDuration =
                        timerEndProcessNewFrames - timerStartProcessNewFrames;
                Logger<std::string>::LogWarning(
                        "Mapper: Process frame exec. time: " + std::to_string(ProcessNewFramesDuration.count()));


                auto timerStartCreateNewFrames = std::chrono::high_resolution_clock::now();
                m_newMapPoints.clear();
                int nNewMapPoints = createMapPoints();
                auto timerEndCreateNewFrames = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> CreateNewFramesDuration =
                        timerEndCreateNewFrames - timerStartCreateNewFrames;
                Logger<std::string>::LogWarning(
                        "Mapper: Create new frame exec. time: " + std::to_string(CreateNewFramesDuration.count()));


                auto timerStartSearchN = std::chrono::high_resolution_clock::now();
                searchNeighbors(m_slamParams->optimizationParams.SearchN);
                auto timerEndSearchN = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> SearchNDuration = timerEndSearchN - timerStartSearchN;
                Logger<std::string>::LogWarning("Mapper: Search Ns exec. time: " + std::to_string(SearchNDuration.count()));


                bool stop;

                auto timerStartBA = std::chrono::high_resolution_clock::now();
                //perform local BA (around the new frame)

                if (m_map->getFrames().size() > 2)
                    Optimizer::localBA(m_map, m_newFrame, &stop, m_slamParams);

                auto timerEndBA = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> BADuration = timerEndBA - timerStartBA;
                Logger<std::string>::LogWarning("Mapper: BA exec. time: " + std::to_string(BADuration.count()));


                //frame culling
                //if (m_tracker->getFrameSequence().size() > 2)
                    //removeFrames();


                auto mapperTimerEnd = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> mapperTotalduration = mapperTimerEnd - mapperTimerStart;

                std::chrono::duration<double, std::milli> mapperFuncsTotalduration =
                        ProcessNewFramesDuration + CreateNewFramesDuration + SearchNDuration + BADuration;
                Logger<std::string>::LogWarning(
                        "Mapper total execution time: " + std::to_string(mapperTotalduration.count()) +
                        "All functions added: " + std::to_string(mapperFuncsTotalduration.count()));
            } else
            {
                if (!m_updated)
                {
                    if (m_slamParams->viewerParams.runViewer)
                        m_viewer->setMapUpdateFlag();
                    m_tracker->setUpdateFlag();
                    Logger<std::string>::LogInfoIV("Mapper is AVAILABLE!");
                    m_updated = true;
                }

            }

        }

        if (m_mappingState == MappingStates::STOP_MAPPING)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds (1000));
            break;
        }

    }
}

    bool Mapping::newFrameAvailable()
    {
        std::unique_lock<std::mutex> lock(m_frameMutex);
        return(!m_availableFrames.empty());
    }

    bool Mapping::processNewFrames()
    {
        {
            std::unique_lock<std::mutex> lock(m_frameMutex);
            Logger<std::string>::LogInfoIV("Mapper processing frames:");

            for(auto& f : m_availableFrames)
                Logger<std::string>::LogInfoIV("Frame: " + std::to_string(f->getID()));
            m_newFrame = m_availableFrames.front();
            m_availableFrames.pop_front();
        }

        //add frame view (current frame) to map points matched
        const std::vector<MapPoint*> mapPoints = m_newFrame->getMapPoints();
        for(size_t i = 0; i<mapPoints.size();i++)
        {
            MapPoint* mapPoint = mapPoints[i];
            if(mapPoint != nullptr)
            {
                if(!mapPoint->isInFrameView(m_newFrame))
                {
                    mapPoint->addFrameView(m_newFrame, i);
                    //update normal and depth
                    //compute ditinctiveDescriptors
                }
                else
                {

                }
            }
        }

        m_newFrame->setConnections();
        //m_map->addKeyFrame()
        return false;
    }

    int Mapping::createMapPoints()
    {
        //1- Computes F from both camera poses (new frame and its connected frames)
        //2- Checks Keypoints (all keypoints, O(N^2)) from both frames, if small epipolar distance (using F)
        //3- Triangulate close points and create new map points
        static const float maxDepth         = m_slamParams->mapPointParams.maxPointDistance;
        static const int   maxNewMapPoints  = m_slamParams->mapPointParams.maxNewPoints;

        struct MapPointData
        {
            MapPointData(const cv::Mat& pos, const size_t des, float e,size_t i1,size_t i2, Frame* connFrame) : position(pos), descriptor(des), error(e), idx1(i1),idx2(i2),connectedFrame(connFrame){}
            cv::Mat position;
            size_t descriptor;
            float error;
            size_t idx1;
            size_t idx2;
            Frame* connectedFrame;
        };
        std::vector<MapPointData> preMapPoints;

        bool mapPointsCreated = false;
        int nMapPoints = 0;
        int nn = 10;
        const float fx = m_newFrame->getCamPrjParams().fx;
        const float fy = m_newFrame->getCamPrjParams().fy;
        const float invFx = 1.0f/fx;
        const float invFy = 1.0f/fy;
        const float cx = m_newFrame->getCamPrjParams().cx;
        const float cy = m_newFrame->getCamPrjParams().cy;
        cv::Mat K = cv::Mat::zeros(3,3,CV_32F);
        K.at<float>(0,0) = fx;
        K.at<float>(1,1) = fy;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;
        K.at<float>(2,2) = 1.0f;

        const std::vector<Frame*> connectedFrames = m_newFrame->getNConnectedFrames(nn);

        cv::Mat R1c = m_newFrame->getRc();
        cv::Mat R1w = m_newFrame->getRw();

        cv::Mat t1c = m_newFrame->getTc();
        cv::Mat t1w  = m_newFrame->getTw();

        std::vector<cv::KeyPoint> kpts1 = m_newFrame->getKeyPoints();
        std::vector<MapPoint*> mps1 = m_newFrame->getMapPoints();

        for(size_t i = 0; i < connectedFrames.size(); i++)
        {

            if(i>0 && newFrameAvailable())
                return false;

            Frame* connectedFrame = connectedFrames[i];
            std::vector<MapPoint*> mps2 = connectedFrame->getMapPoints();

            cv::Mat R2c = connectedFrame->getRc();
            cv::Mat R2w = connectedFrame->getRw();

            cv::Mat t2c = connectedFrame->getTc();
            cv::Mat t2w = connectedFrame->getTw();


            cv::Mat vectorBaseLine = t1w - t2w;
            const float baseline = cv::norm(vectorBaseLine);

            {
                //compute median depth from connectedFrame
            }

            cv::Mat F21 = computeF12(m_newFrame, connectedFrame);
            //Logger<std::string>::LogInfoI("Mapper matching Frames: " + std::to_string(m_newFrame->getID()) + " and " + std::to_string(connectedFrame->getID()));

            std::vector<PtPair> matchedFeatures;
            int nMatches = Matcher::matchByEpipolarDist(m_newFrame, connectedFrame, F21, matchedFeatures);

            //Logger<std::string>::LogInfoI("Mapper triangulation Frames: " + std::to_string(m_newFrame->getID()) + " and " + std::to_string(connectedFrame->getID()) + " potential new points matched: " + std::to_string(nMatches));

            //triangulate matches
            if(nMatches > 0)
            {

                std::vector<cv::KeyPoint> kpts2 = connectedFrame->getKeyPoints();

                cv::Mat p3D;
                for(size_t i = 0; i < matchedFeatures.size(); i++)
                {

                    size_t idx1 = matchedFeatures[i].first;
                    size_t idx2 = matchedFeatures[i].second;

                    cv::KeyPoint kp1 = kpts1[idx1];
                    cv::KeyPoint kp2 = kpts2[idx2];

                    //back-project points into rays with depth=1
                    cv::Mat r1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx)*invFx, (kp1.pt.y-cy)*invFy, 1.0);
                    cv::Mat r2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx)*invFx, (kp2.pt.y-cy)*invFy, 1.0);

                    //find parallax between rays
                    float rParallax = r1.dot(r2)/ (cv::norm(r1)*cv::norm(r2));
                    if(rParallax > 0.9998)
                        continue;

                    cv::Mat P1 = cv::Mat::eye(3,4,CV_32F);
                    R1c.copyTo(P1.rowRange(0, 3).colRange(0, 3));
                    t1c.copyTo(P1.rowRange(0, 3).col(3));
                    //P1 = K * P1;

                    cv::Mat P2 = cv::Mat::eye(3, 4, CV_32F);
                    R2c.copyTo(P2.rowRange(0, 3).colRange(0, 3));
                    t2c.copyTo(P2.rowRange(0, 3).col(3));
                    //P2 = K * P2;

                    cv::Mat A(4, 4, CV_32F);
                    A.row(0) = r1.at<float>(0) * P1.row(2) - P1.row(0);
                    A.row(1) = r1.at<float>(1) * P1.row(2) - P1.row(1);
                    A.row(2) = r2.at<float>(0) * P2.row(2) - P2.row(0);
                    A.row(3) = r2.at<float>(1) * P2.row(2) - P2.row(1);

                    cv::Mat U, S, Vt;
                    cv::SVD::compute(A, S, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                    p3D = Vt.row(3).t();
                    p3D = p3D.rowRange(0, 3) / p3D.at<float>(3);
                    cv::Mat p3Dt = p3D.t();

                    //check point behind camera and too far from camera
                    float z1 = R1c.row(2).dot(p3Dt)+t1c.at<float>(2);
                    if((z1<=0) || (z1 >= maxDepth))
                        continue;
                    float z2 = R2c.row(2).dot(p3Dt)+t2c.at<float>(2);
                    if((z2<=0) || (z2 >= maxDepth))
                        continue;

                    //check re-projection error
                    const float p1x = R1c.row(0).dot(p3Dt) + t1c.at<float>(0);
                    const float p1y = R1c.row(1).dot(p3Dt) + t1c.at<float>(1);
                    const float p1zInv = 1.0f/z1;
                    const float u1  = (p1x*fx*p1zInv)+cx;
                    const float v1  = (p1y*fy*p1zInv)+cy;
                    float ex1 = u1-kp1.pt.x;
                    float ey1 = v1-kp1.pt.y;
                    float e1  = ex1*ex1+ey1*ey1;
                    if(e1 > 5.991)
                        continue;

                    const float p2x = R2c.row(0).dot(p3Dt) + t2c.at<float>(0);
                    const float p2y = R2c.row(1).dot(p3Dt) + t2c.at<float>(1);
                    const float p2zInv = 1.0f/z2;
                    const float u2  = (p2x*fx*p2zInv)+cx;
                    const float v2  = (p2y*fy*p2zInv)+cy;
                    float ex2 = u2-kp2.pt.x;
                    float ey2 = v2-kp2.pt.y;
                    float e2  = ex2*ex2+ey2*ey2;
                    if(e2 > 5.991)
                        continue;


                    size_t descriptor = m_newFrame->getSimDescriptors()[idx1];
                    preMapPoints.emplace_back(p3D, descriptor,(e1+e2/2.0f),idx1,idx2,connectedFrame);

                    nMapPoints++;
                }
            }

        }

        if(!preMapPoints.empty())
        {
            //sort through least erroneous map poitns and keep
            std::sort(preMapPoints.begin(), preMapPoints.end(), [](MapPointData& p1, MapPointData& p2) {return p1.error < p2.error;});

            size_t counter = 0;
            for(auto& pt : preMapPoints)
            {
                MapPoint* mapPoint = new MapPoint(pt.position, m_newFrame, m_map, pt.descriptor, pt.error);
                mapPoint->addFrameView(m_newFrame, pt.idx1);
                mapPoint->addFrameView(pt.connectedFrame,pt.idx2);

                m_newFrame->addMapPoint(mapPoint, pt.idx1);
                pt.connectedFrame->addMapPoint(mapPoint,pt.idx2);

                m_map->addMapPoint(mapPoint);
                m_newMapPoints.push_back(mapPoint);
                counter++;
                if(counter >= maxNewMapPoints)
                    break;
            }
        }
    return nMapPoints;
    }

    bool Mapping::deleteMapPoints()
    {
        return false;
    }

    bool Mapping::searchNeighbors(int n)
    {

        const std::vector<Frame*> nFrames = m_newFrame->getNConnectedFrames(n);
        const unsigned long int newFrameID = m_newFrame->getID();
        std::vector<Frame*> targetFrames;
        std::vector<std::pair<int,int> > matches;
        size_t nMatches =0;

        for(std::vector<Frame*>::const_iterator it = nFrames.begin(), itEnd = nFrames.end();it!=itEnd; it++)
        {
            Frame* nFrame = *it;
            if(nFrame->getActiveFuseID() == m_newFrame->getID())
                continue;

            nFrame->setActiveFuseID(m_newFrame->getID());
            targetFrames.push_back(nFrame);

            //extend to neighbours
            const std::vector<Frame*> n2Frames = nFrame->getNConnectedFrames(5);
            for(std::vector<Frame*>::const_iterator it2 = n2Frames.begin(),it2End=n2Frames.end();it2!=it2End;it2++)
            {
                Frame* n2Frame = *it2;
                if(n2Frame->getActiveFuseID() == m_newFrame->getID() || n2Frame->getID() == m_newFrame->getID())
                    continue;

                n2Frame->setActiveFuseID(m_newFrame->getID());
                targetFrames.push_back(n2Frame);
            }
        }

        //fetch this frame's map points
        std::vector<MapPoint*> mapPoints = m_newFrame->getMapPoints();

        //attempt to match and fuse current keyframe map points with other keyframes (target) map points
        for(std::vector<Frame*>::iterator it = targetFrames.begin(), itEnd = targetFrames.end(); it!=itEnd; it++)
        {
            Frame* targetFrame = *it;

            nMatches = Matcher::fuseByProjection(targetFrame, mapPoints, matches);
        }

        nMatches = 0;
        matches.clear();

        //attempt to match and fuse other keyframes (target) map points with current keyframe map points
        std::vector<MapPoint*> targetMapPoints;
        for(std::vector<Frame*>::iterator it = targetFrames.begin(), itEnd = targetFrames.end(); it!=itEnd; it++)
        {
            Frame* frame = *it;
            std::vector<MapPoint*> otherFrameMapPoints = frame->getMapPoints();

            for(std::vector<MapPoint*>::iterator mpIt = otherFrameMapPoints.begin(), mpItEnd = otherFrameMapPoints.end(); mpIt != mpItEnd; mpIt++)
            {
                MapPoint* targetMapPoint = *mpIt;
                if(targetMapPoint == nullptr)
                    continue;
                if(targetMapPoint->getActiveFuseID() == newFrameID)
                    continue;
                targetMapPoint->setActiveFuseID(newFrameID);
                targetMapPoints.push_back(targetMapPoint);
            }
        }
        nMatches = Matcher::fuseByProjection(m_newFrame, targetMapPoints, matches);
        if(nMatches > 0)
            return true;
        return false;
    }

    void Mapping::removeFrames()
    {
        //iterate through connected frames to this frame
        //iterate through its map points and iterate through frameviews of each map point
        //check that the descriptor from connected frame is the same as the map point descriptor

        size_t nMapPoints = 0;
        size_t nCoviewedCount = 3;
        size_t nRedundancy = 0;
        const size_t minFrameViews = 3;

        std::vector<Frame*> connectedFrames = m_newFrame->getConnectedFrames();
        for(std::vector<Frame*>::iterator it = connectedFrames.begin(), itEnd = connectedFrames.end(); it!=itEnd; it++)
        {
            nMapPoints = 0;
            Frame* connectedFrame = (*it);

            //we skip both first frames
            if(connectedFrame->getID() == 0 || connectedFrame->getID()== 1)
                continue;

            std::vector<MapPoint*> mapPoints = connectedFrame->getMapPoints();
            for(size_t i = 0; i < connectedFrame->getN(); i++)
            {
                MapPoint* mapPoint = mapPoints[i];
                if(mapPoint!= nullptr)
                {
                    size_t mapPointDescriptor = mapPoint->getDescriptor();
                    nMapPoints++;

                    //get frameviews from this map point
                    std::map<Frame*, size_t> frameViews = mapPoint->getFrameViews();
                    if(frameViews.size() > minFrameViews)
                    {
                        nCoviewedCount=0;
                        //now check that the same map point is seen from both frames (views)
                        for(std::map<Frame*, size_t>::const_iterator itFVs=frameViews.begin(), itFVsEnd=frameViews.end(); itFVs!=itFVsEnd; itFVs++)
                        {
                            Frame* frameView = (*itFVs).first;
                            if(frameView->getID() ==  connectedFrame->getID())
                                continue;

                            size_t idx = (*itFVs).second;
                            size_t frameMapPointDescriptor = frameView->getSimDescriptors()[idx];

                            if(mapPointDescriptor == frameMapPointDescriptor)
                                nCoviewedCount++;
                            if(nCoviewedCount > minFrameViews)
                                break;
                        }

                        if(nCoviewedCount >= minFrameViews)
                            nRedundancy++;
                    }

                }
            }
            if(nRedundancy > 0.9*nMapPoints)
            {
                //remove this frame
                Logger<std::string>::LogWarning("removing redundant frame: " + std::to_string(connectedFrame->getID()));
                connectedFrame->removeFrame();
                m_map->removeFrame(connectedFrame);
                m_tracker->removeFrame(connectedFrame);
            }
        }

    }

    void Mapping::insertFrame(Frame *frame)
    {
        std::unique_lock<std::mutex> lock(m_frameMutex);
        m_availableFrames.push_back(frame);
        m_abortBA = true;
    }

    void Mapping::setTracker(std::shared_ptr<Tracking> tracker)
    {
        m_tracker = tracker;
    }

    void Mapping::setViewer(std::shared_ptr<Viewer> viewer)
    {
        m_viewer = viewer;
    }

