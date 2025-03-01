#include <regex>
#ifndef TRACKING_H

#include "tracking.h"
#include "slamUtils.h"

Tracking::Tracking(SlamParams* slamParams) : m_slamParams(slamParams)
{
    m_trackingState = TrackingStates::NO_FRAME;;
    m_MVG.updateParams(m_slamParams);
    initializeKMatrix();
}

Tracking::Tracking(std::shared_ptr<Map> map, SlamParams* slamParams) : m_map(map), m_slamParams(slamParams)
{
    m_trackingState = TrackingStates::NO_FRAME;
    m_MVG.updateParams(m_slamParams);
    m_minJumpFrames = static_cast<char>(m_slamParams->optimizationParams.minJumpFrames);
    m_maxJumpFrames = static_cast<char>(m_slamParams->optimizationParams.maxJumpFrames);
    initializeKMatrix();

}

bool Tracking::run(const VertexFeatures& vertexFeatures, const double currentTimestamp, unsigned long int mFrameCount)
{
    bool ok = false;

    switch(m_trackingState)
    {
        case (TrackingStates::TRACKING):
        {
            const int maxBadTrack = m_slamParams->errorParams.maxBadTrackCnt;
            //TODO: to be removed, this is for debugging purposes
            auto timerStart = std::chrono::high_resolution_clock::now();

            m_newFrame = Frame(vertexFeatures, currentTimestamp, m_trackFrameCnt, m_slamParams);
            if (trackNextFrame())
            {
                updateTrackedFrame();
            } else
            {
                //attempt to recover tracking
                m_badTrackingCount++;
                m_trackWithMotion = false;
                m_viewer->setSquareUpdateFlag(4);

                if (m_badTrackingCount > maxBadTrack)
                {
                    Logger<std::string>::LogError("Tracking Lost!");
                    m_trackingState = TrackingStates::LOST;
                }
                ok = false;

            }

            auto timerEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = timerEnd - timerStart;

            m_viewer->setSquareUpdateFlag(2);
            ok = true;
            break;
        }

        case(TrackingStates::NOT_INITIALIZED):
        {
            m_newFrame = Frame(vertexFeatures, currentTimestamp, m_trackFrameCnt, m_slamParams);

            //TODO: REMOVE CONDITION, ONLY FOR DEBUGGING AND GUARANTEE FIRST 2 FRAMES INITIALIZE
            //if(m_oldFrame.getID()==0 && m_newFrame.getID()==1)
            if (initialize())
                ok = true;
            else
                ok = false;

            break;
        }

        case(TrackingStates::NO_FRAME):
        {
            m_oldFrame = Frame(vertexFeatures, currentTimestamp, m_trackFrameCnt, m_slamParams);
            if(m_oldFrame.isInitialized())
                m_trackingState = TrackingStates::NOT_INITIALIZED;
            break;
        }

        case(TrackingStates::LOST):
        {
            //tracking state is lost, attempt to re-localize
            auto timerStart = std::chrono::high_resolution_clock::now();
            m_newFrame = Frame(vertexFeatures, currentTimestamp, m_trackFrameCnt, m_slamParams);
            Frame* recoveryFrame = nullptr;
            if(trackLostFrame(recoveryFrame))
            {
                Logger<std::string>::LogInfoIV("Tracking recovered. Updating relocalization.");
                if(updateTrackedFrame())
                {
                    m_trackingState = TrackingStates::TRACKING;
                    Logger<std::string>::LogInfoIV("Update successul.");
                    return true;
                }
            }
            Logger<std::string>::LogError("Tracking still lost");
            m_viewer->setSquareUpdateFlag(3);

            break;
        }

        case(TrackingStates::STOP_TRACKING):
        {
            Logger<std::string>::LogInfoI("Tracker stopped.");
            ok = false;
            break;
        }
    }

    m_trackFrameCnt++;
    return ok;
}


bool Tracking::initialize()
{
    //match features (always with previous frame), must be more than min matches
    if (m_newFrame.MatchWithFrame(m_oldFrame))
    {
        if (m_MVG.initializePoseAndMap(m_newFrame, m_oldFrame, m_initP3Dw, m_prunedMatchesIdx))
        {
            Logger<std::string>::LogInfoIII("Initialized with frames: " + std::to_string(m_oldFrame.getID()) + " and " + std::to_string(m_newFrame.getID()));
            if (createInitialMap())
            {
                m_trackingState = TRACKING;

                return true;
            }
            else
                Logger<std::string>::LogError("Failed to create map");
        }
        else
        {
            ;
        }
    }
    else
    {
        Logger<std::string>::LogWarning("No Feature Matches!");
    }
    return false;
}

bool Tracking::createInitialMap()
{
     int nMapPoints = 0;

     std::vector<MapPoint*> createdMapPoints;
    //Instantiate the two new frames
    Frame* pPreviousFrame = new Frame(m_oldFrame);
    Frame* pCurrentFrame = new Frame(m_newFrame);

    //TODO: ****NEEDS TO BE REMOVED WHEN USED OUTSIDE SIMULATION****
    //This is only used for (virtual SLAM) to set map scale and offset to origin
    glm::mat4 vc1 = m_oldFrame.getVirtualCamPose();

    glm::mat4 vc2 = m_newFrame.getVirtualCamPose();
    float lc1c2 = glm::length(glm::vec3(vc1[3])-glm::vec3(vc2[3]));

    //now replace previous frame which should be origin t and identity R
    cv::Mat previousVirtualPose = cv::Mat::eye(4, 4, CV_32F);
    convertGLM2CV(vc1,previousVirtualPose);

    cv::Mat currentVirtualPose =  cv::Mat::eye(4, 4, CV_32F);
    convertGLM2CV(vc2,currentVirtualPose);
    cv::Mat relativeCurrentPose = previousVirtualPose.inv() * currentVirtualPose;

    //scale by scale factor
    cv::Mat translationVec = relativeCurrentPose(cv::Rect(3, 0, 1, 3)).clone();
    translationVec *= lc1c2;

    m_map->insertFrame(pPreviousFrame);
    m_map->insertFrame(pCurrentFrame);

    //create map points and associate to Frames
    for (size_t i = 0; i < m_prunedMatchesIdx.size(); i++)
    {
        if(m_prunedMatchesIdx[i].first == -1)
            continue;

        //get estimated positions
        cv::Mat posw(m_initP3Dw[i]);

        //feature descriptor to store in map point
        size_t descriptor = pCurrentFrame->getSimDescriptors()[m_prunedMatchesIdx[i].second];

        //create map point from it
        MapPoint *pMapPoint = new MapPoint(posw, pCurrentFrame, m_map, descriptor);


        //now add this map point to the frames (use matches as idx: previous=first, current=second
        pPreviousFrame->addMapPoint(pMapPoint, m_prunedMatchesIdx[i].first);
        pCurrentFrame->addMapPoint(pMapPoint, m_prunedMatchesIdx[i].second);

        m_oldFrame.addMapPoint(pMapPoint,m_prunedMatchesIdx[i].first);
        m_newFrame.addMapPoint(pMapPoint, m_prunedMatchesIdx[i].second);


        createdMapPoints.push_back(pMapPoint);

        //to each map point add the frame view and its corresponding index pt
        pMapPoint->addFrameView(pPreviousFrame, m_prunedMatchesIdx[i].first);
        pMapPoint->addFrameView(pCurrentFrame, m_prunedMatchesIdx[i].second);


        pMapPoint->setLastFrameViewID(pCurrentFrame->getID());
        //ORB_SLAM refines descriptors here... Skipping
        //pMapPoint

        //add map point to map
        m_map->addMapPoint(pMapPoint);
        nMapPoints++;
    }

    //Maybe skipping this for now
    pPreviousFrame->setConnections();
    pCurrentFrame->setConnections();

    const int iterations = m_slamParams->optimizationParams.GBAIterations;
    //Optimize
    Optimizer::globalBundleAdjustment(m_map, iterations);

    float medianDepth = pCurrentFrame->computeMedianDepth();

    //this is rare, but if it happens, need to cleanup
    if(medianDepth<0.0f)
    {
        for(auto* mp : createdMapPoints) delete mp;
        delete pCurrentFrame;
        delete pPreviousFrame;
        return false;
    }
    float invDepth = lc1c2;

    cv::Mat newT = pCurrentFrame->getPosec().col(3).rowRange(0,3) *invDepth;
    pCurrentFrame->sett(newT);
    pCurrentFrame->updateTransform();

    std::vector<MapPoint*> mapPoints = pCurrentFrame->getMapPoints();
    for(size_t i = 0; i < pCurrentFrame->getN(); i++)
    {
        MapPoint* mp = mapPoints[i];
        if(mp != nullptr)
        {
            glm::vec3 mpPos = glm::vec3(mp->getPosition().at<float>(0)*invDepth,mp->getPosition().at<float>(1)*invDepth,mp->getPosition().at<float>(2)*invDepth);
            //mpPos += glm::vec3(vc1[3]);
            cv::Mat newMpPos = cv::Mat::eye(3, 1, CV_32F);
            newMpPos.at<float>(0) = mpPos[0];
            newMpPos.at<float>(1) = mpPos[1];
            newMpPos.at<float>(2) = mpPos[2];
            mp->setPos(newMpPos);
        }
    }

    Logger<std::string>::LogInfoIII("Map created: " + std::to_string(nMapPoints) + " map points");

    //insert into mapper
    m_mapper->insertFrame(pCurrentFrame);
    m_mapper->insertFrame(pPreviousFrame);

    m_oldFrame = *pCurrentFrame;
    m_referenceFrame = pCurrentFrame;

    m_frameSequence.push_back(pPreviousFrame);
    m_frameSequence.push_back(pCurrentFrame);

    m_lastKFrameID = pCurrentFrame->getImageFrameNumber();

    //m_mapper->updateViewer();
    return true;
}

void Tracking::setMapping(std::shared_ptr<Mapping> mapping)
{
    m_mapper = mapping;
}

void Tracking::setViewer(std::shared_ptr<Viewer> viewer)
{
    m_viewer = viewer;
}

bool Tracking::trackNextFrame()
{
    std::vector<MapPoint*> mapPointMatches;
    const int minMatches = m_slamParams->featureParams.minRefFrameMatches;
    const float minInlierRatio = m_slamParams->errorParams.minInlierRatio;
    bool tracked = false;
    //track using previous motion estimation
    if (m_trackWithMotion)
    {
        //while(1);
        //set initial estimated pose of frame based on previous frame and motion
        m_newFrame.setPose(m_motion * m_oldFrame.getPosec());
        Logger<std::string>::LogInfoI("Next frame: " + std::to_string(m_newFrame.getID()) + " tracking with motion and projection method. ");

        //Attempt to match by projection, otherwise brute-force match all
        size_t matches = Matcher::matchAndAddByProjection(&m_newFrame, m_referenceFrame);
        if(matches > minMatches)
        {
            m_trackInliers = Optimizer::pnpOnFrame(m_newFrame, m_slamParams->optimizationParams.PNPIterations);

            if(m_trackInliers > static_cast<float>(matches) * minInlierRatio)
            {
                tracked = true;
            }
            else
            {
                Logger<std::string>::LogWarning("Failed motion and projection PNP on Frame: " + std::to_string(m_newFrame.getID()) + " inliers: " + std::to_string(m_trackInliers));
            }

        }
        else
        {
            //BF match map point descriptors to new frame captured feature descriptors and perform pose estimation
            Logger<std::string>::LogInfoI("Now attempting tracking with motion and all-matches method: ");
            size_t matches = Matcher::matchAndAddAll(m_newFrame, m_referenceFrame) > minMatches;
            if (matches > minMatches)
            {

                //perform PNP using:
                // matched map points (fixed 3D), new measurements and previous pose as initial estimate
                m_newFrame.setPose(m_referenceFrame->getPosec());
                m_trackInliers = Optimizer::Optimizer::pnpOnFrame(m_newFrame, m_slamParams->optimizationParams.PNPIterations);

                if(m_trackInliers > static_cast<float>(matches) * minInlierRatio)
                {
                    tracked = true;
                }
                else
                {
                    Logger<std::string>::LogWarning("Failed motion and all-matches PNP on Frame: " + std::to_string(m_newFrame.getID()) + " inliers: " + std::to_string(m_trackInliers));
                }
            }
            else
            {
                Logger<std::string>::LogWarning("Frame: " + std::to_string(m_newFrame.getID()) + " tracking with motion and all-matches method FAILED. Ref. Frame: " + std::to_string(m_referenceFrame->getID()));
            }
        }
    }

        //track from static frame
    else
    {
        Logger<std::string>::LogInfoI("Next frame: " + std::to_string(m_newFrame.getID()) + " tracking no-motion and all-matches method");

        //BF match map point descriptors to new frame captured feature descriptors and perform pose estimation
        size_t matches = Matcher::matchAndAddAll(m_newFrame, m_referenceFrame);
        if (matches > minMatches)
        {
            //perform PNP using:
            // matched map points (fixed 3D), new measurements and previous pose as initial estimate
            m_newFrame.setPose(m_referenceFrame->getPosec());
            m_trackInliers = Optimizer::Optimizer::pnpOnFrame(m_newFrame, m_slamParams->optimizationParams.PNPIterations);

            if(m_trackInliers > static_cast<float>(matches) * minInlierRatio)
            {
                tracked = true;
            }
            else
            {
                Logger<std::string>::LogWarning("Frame: " + std::to_string(m_newFrame.getID()) + "Failed no-motion and all-matches PNP on Frame: ");
            }
        }
        else
        {
            Logger<std::string>::LogWarning("Frame: " + std::to_string(m_newFrame.getID()) + " tracking no-motion and all-matches method FAILED. Ref. Frame: " + std::to_string(m_referenceFrame->getID()));
        }
    }

    if(tracked)
    {
        m_trackInliers = updateFrameOutliers();

        if (m_trackInliers > minMatches)
        {
            Logger<std::string>::LogInfoIII("Frame " + std::to_string(m_newFrame.getID()) + " tracked. Inliers: " + std::to_string(m_trackInliers));
            return true;
        }
    }
    Logger<std::string>::LogError("Frame " + std::to_string(m_newFrame.getID()) + " tracking failed. Inliers: " + std::to_string(m_trackInliers));

    return false;
}

void Tracking::updateLocalMap()
{
    //after initial pose estimation on new frame (based on matched descriptors)
    //find all frames that view every map point from the new frame and add to a local map

    //counter for every frame that views map points (some frames view more map points than others)
    //the number of co-viewed map points are connection weights with new frame
    std::map<Frame*,size_t> frameViewsCounter;
    std::vector<MapPoint*> mapPoints = m_newFrame.getMapPoints();
    size_t N = mapPoints.size();

    for(size_t i = 0; i < N; i++)
    {
        MapPoint* mp = mapPoints[i];

        if((mp == nullptr) || (mp->isBad()))
            continue;

        const std::map<Frame*,size_t> frameViews = mp->getFrameViews();
        for(std::map<Frame*,size_t>::const_iterator  it = frameViews.begin(), itend = frameViews.end(); it!=itend; it++)
        {
            frameViewsCounter[it->first]++;
        }
    }

    //Add frames to local map (local frames), frame and number of co-viewed map points
    //check frame with most co-views (connections)
    size_t max = 0;
    Frame* frameMaxViews = nullptr;
    m_localFrames.clear();
    m_localFrames.reserve(3 * frameViewsCounter.size());
    for(std::map<Frame*,size_t>::const_iterator it=frameViewsCounter.begin(),itend=frameViewsCounter.end();it!=itend;it++)
    {
        if(it->second > max)
        {
            max=it->second;
            frameMaxViews = it->first;
        }

        m_localFrames.push_back(it->first);
    }

    //set reference frame as frame with most connections

    if(frameMaxViews != nullptr)
    {
        Logger<std::string>::LogInfoI("Update Local Map: setting reference frame: " + std::to_string(frameMaxViews->getID()));
        m_referenceFrame = frameMaxViews;
    }

    //add map points to a local map
    //for every frame stored in the local map iterate through its map points and add the map points to the local map points
    for(std::vector<Frame*>::const_iterator it=m_localFrames.begin(),itend=m_localFrames.end(); it != itend; it++)
    {
        Frame* frame = *it;
        const std::vector<MapPoint*> localMps = frame->getMapPoints();
        for(std::vector<MapPoint*>::const_iterator itMP=localMps.begin(),itEndMP=localMps.end();itMP!=itEndMP;itMP++)
        {
            MapPoint* mapPoint = *itMP;
            if((mapPoint == nullptr) || (mapPoint->isBad()))
                continue;
            if(mapPoint->getFrameUpdateID() == m_newFrame.getID())
                continue;
            mapPoint->setFrameUpdateID(m_newFrame.getID());
            m_localMapPoints.push_back(mapPoint);
        }
    }

}

size_t Tracking::trackLocalMap()
{
    if (m_localFrames.empty() || m_localMapPoints.empty())
        return false;


    //mark map points from local map that belong to new frame to avoid repetition when matching
    size_t inFrustumCounter = 0;
    std::vector<MapPoint *> mapPoints = m_newFrame.getMapPoints();
    for (std::vector<MapPoint *>::const_iterator it = mapPoints.begin(), itend = mapPoints.end(); it != itend; it++)
    {
        MapPoint *mapPoint = *it;
        if((mapPoint == nullptr) || (mapPoint->isBad()))
            continue;

        mapPoint->increaseVisible();
        mapPoint->setLastFrameViewID(m_newFrame.getID());
    }

    //now check if any of the map points from the local map has not been matched already to newest frame
    //first check if in viewing frustum
    std::vector<MapPoint*> mapPointsInFrustum;
    for (size_t i = 0; i < m_localMapPoints.size(); i++)
    {
        MapPoint *mapPoint = m_localMapPoints[i];

        //assume most map points are already matched
        if (mapPoint->getLastFrameViewID() == m_newFrame.getID())
            continue;

        //now for every map point from local map, check if in viewing frustum of new frame
        if (isInFrustum(&m_newFrame, mapPoint))
        {
            mapPoint->increaseVisible();
            mapPoint->setLastFrameViewID(m_newFrame.getID());
            inFrustumCounter++;
            mapPointsInFrustum.push_back(mapPoint);
        }
    }

    m_localMapPoints.clear();
    m_localFrames.clear();

    //finally do a fast matching, by using grids partitioning from the frames
    if (inFrustumCounter)
    {
        return Matcher::matchAndAddByProjection(&m_newFrame, mapPointsInFrustum);
    }

    return 0;
}

bool Tracking::isInFrustum(Frame *frame, MapPoint *mp)
{
    const cv::Mat ptWorld = mp->getPosition();
    const cv::Mat ptCam = frame->getRc() * ptWorld + frame->getTc();
    const float ptX = ptCam.at<float>(0);
    const float ptY = ptCam.at<float>(1);
    const float ptZ = ptCam.at<float>(2);

    if(ptZ < 0.0f)
        return false;

    auto& K = frame->getCamPrjParams();
    const float invz = 1.0f/ptZ;
    const float u=K.fx * ptX * invz + K.cx;
    const float v=K.fy * ptY * invz + K.cy;

    if(u < 0 || u > K.width)
        return false;
    if(v < 0 || v > K.height)
        return false;

    mp->setLastImageCoord(cv::Point2i (u,v));
    return true;
}

Tracking::~Tracking()
{
    if(m_referenceFrame != nullptr)
    {
        delete m_referenceFrame;
        m_referenceFrame = nullptr;
    }
}

void Tracking::initializeKMatrix()
{
    m_K = cv::Mat::eye(3, 3, CV_32F);

    if (m_slamParams != nullptr)
    {
        m_K.at<float>(0, 0) = m_slamParams->camParams.fx;
        m_K.at<float>(1, 1) = m_slamParams->camParams.fy;
        m_K.at<float>(0, 2) = m_slamParams->camParams.cx;
        m_K.at<float>(1, 2) = m_slamParams->camParams.cy;
    }
}

bool Tracking::createNewFrame()
{

    const unsigned int minFrames = (m_frameSequence.size() < 3) ? 2 : 3;

    const unsigned int refTrackedMapPoints = m_referenceFrame->getNTrackedMapPoints(2);
    const unsigned int newTrackedMapPoints = m_newFrame.getNTrackedMapPoints(2);
    const unsigned int GBAJumpFrames = m_slamParams->optimizationParams.GBAJumpFrames;

    bool mapperReady     = checkUpdateFlag();
    bool minFramesPassed = m_lastKFrameID + static_cast<unsigned long int>(m_minJumpFrames) < m_newFrame.getImageFrameNumber();
    bool GBAReady        = m_lastKFrameID + static_cast<unsigned long int>(GBAJumpFrames) < m_newFrame.getImageFrameNumber();
    std::string minFramesPassedString = (minFramesPassed ) ? "true" : "false";
    std::string mapperReadyString = (mapperReady ) ? "true" : "false";

    float displacement = cv::norm(m_newFrame.getTw() - m_oldFrame.getTw());
    bool displaced = (displacement > m_slamParams->optimizationParams.minDisplacement);
    if(!displaced)
        m_stillCount++;

    Logger<std::string>::LogInfoII("displacement: " + std::to_string(displacement));

    if(minFramesPassed && mapperReady && displaced)
    {
        Logger<std::string>::LogInfoIII("new KF: frame " + std::to_string(m_newFrame.getID()) +
                                        " tracked map points: " + std::to_string(newTrackedMapPoints) +
                                        "\n reference frame " + std::to_string(m_referenceFrame->getID()) +
                                        ": tracked map points: " + std::to_string(refTrackedMapPoints));

        Frame *frame = new Frame(m_newFrame);
        m_lastKFrameID = m_newFrame.getImageFrameNumber();
        m_referenceFrame = frame;
        m_mapper->insertFrame(frame);
        m_map->insertFrame(frame);
        m_oldFrame = &m_newFrame;
        m_frameSequence.push_back(frame);

        if(GBAReady)
        {
            const int iterations = m_slamParams->optimizationParams.GBAIterations;
            Logger<std::string>::LogInfoIII("Performing GBA!");
            //Optimize
            Optimizer::globalBundleAdjustment(m_map, iterations);

        }
        return true;
    }
    else
    {
        Logger<std::string>::LogWarning("No KF: frame " + std::to_string(m_newFrame.getID()) +
                                        " tracked map points: " + std::to_string(newTrackedMapPoints) +
                                        " minFramePassed: " + minFramesPassedString +
                                        " mapperReady: " + mapperReadyString +
                                        "\n reference frame " + std::to_string(m_referenceFrame->getID()) +
                                        ": tracked map points: " + std::to_string(refTrackedMapPoints));
    }
    return false;
}

size_t Tracking::updateFrameOutliers()
{
    size_t trackedCount = 0;
    //set outlier slamParams
    for (size_t i = 0; i < m_newFrame.getN(); i++)
    {
        MapPoint *mapPoint = m_newFrame.getMapPoints()[i];

        if (mapPoint != nullptr)
            if (mapPoint->getFrameViews().size() > 0)
            {
                if (m_newFrame.getOutliers()[i])
                {
                    mapPoint = nullptr;
                } else
                {
                    trackedCount++;
                }
            }
    }

    return trackedCount;
}

void Tracking::clearUpdateFlag()
{
    std::lock_guard<std::mutex> lock(m_mutexUpdate);
    m_frameUpdateFlag = false;
}

void Tracking::setUpdateFlag()
{
    std::lock_guard<std::mutex> lock(m_mutexUpdate);
    m_frameUpdateFlag = true;
}

bool Tracking::trackLostFrame(Frame* frame)
{
    bool tracked = false;
    auto minMatches = m_slamParams->featureParams.minMatches;
    size_t matchedFeatures  = Matcher::matchOldFrames(&m_newFrame, m_frameSequence, 50, frame);
    if(matchedFeatures >= minMatches)
    {
        //perform PNP using:
        // matched map points (fixed 3D), new measurements and previous pose as initial estimate
        m_newFrame.setPose(m_referenceFrame->getPosec());
        m_trackInliers = Optimizer::Optimizer::pnpOnFrame(m_newFrame, m_slamParams->optimizationParams.PNPIterations);
        auto minTracked = m_slamParams->featureParams.minTracked;
        if(m_trackInliers > static_cast<float>(matchedFeatures) * 0.95)
        {
            tracked = true;
        }
        else
        {
            Logger<std::string>::LogWarning("Failed LOST recovery tracking PNP on Frame: " + std::to_string(m_newFrame.getID()) + " inliers: " + std::to_string(m_trackInliers));
        }
    }

    return tracked;
}

void Tracking::removeFrame(Frame *frame)
{
    auto it = std::find(m_frameSequence.begin(), m_frameSequence.end(), frame);
    if(it != m_frameSequence.end())
    {
        m_frameSequence.erase(it);
    }
}

bool Tracking::updateTrackedFrame()
{
    bool ok = false;
    //build a local map (every frame that views map points from new frame and every of the frame's map points)
    updateLocalMap();

    //Now check matching map points from local map and new frame
    //when points are tracked and matched, these map points are added to new frame
    m_matchedtoLocalMap = trackLocalMap();

    //if non-zero, this is bonus, does not need to be extra though
    if(m_matchedtoLocalMap > 0)
    {
        //if new map points are matched perform further pose estimation
        Logger<std::string>::LogInfoI("Frame: " + std::to_string(m_newFrame.getID()) + ": " + std::to_string(m_matchedtoLocalMap) + " map points matched and added. ");
        Logger<std::string>::LogInfoI("inliers before local map matching: " + std::to_string(m_trackInliers));
        m_trackInliers = Optimizer::Optimizer::pnpOnFrame(m_newFrame, m_slamParams->optimizationParams.PNPIterations);
        Logger<std::string>::LogInfoI("inliers after local map matching: " + std::to_string(m_trackInliers));

        //TODO: Go back to older pose if inlier ratio is too low (since PNP will set camera to new pose)

    }

    // Update motion model
    if(!m_oldFrame.getPosec().empty())
    {
        cv::Mat motionTransform = cv::Mat::eye(4,4,CV_32F);
        motionTransform = m_oldFrame.getPosew().clone();
        m_motion = m_newFrame.getPosec() * motionTransform;
        m_trackWithMotion = true;
    }

    //check if create new keyframe
    if(createNewFrame())
    {
        ok = true;
    }
    else
    {
        //m_referenceFrame = &m_oldFrame;
        TFrame* tFrame = new TFrame();
        tFrame->setPose(m_newFrame.getPosec());
        tFrame->setFrameID(m_newFrame.getID());
		tFrame->setTimeStamp(m_newFrame.getTimeStamp());												
        m_map->insertTFrame(tFrame);
        m_tFrameSequence.emplace_back(tFrame);

        //if no movement, do global BA
        if(m_stillCount > 10)
        {
            //Logger<std::string>::LogInfoIV("\n\n\n performing GBA!!! \n\n\n ");
            //Optimizer::globalBundleAdjustment(m_map,m_slamParams->optimizationParams.GBAIterations);
            m_stillCount = 0;
        }
    }

    m_badTrackingCount = 0;
    return ok;
}

void Tracking::convertGLM2CV(const glm::mat4 &glmMat, cv::Mat &cvMat)
{
    const float* glmData = glm::value_ptr(glmMat);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            cvMat.at<float>(i, j) = glmData[j * 4 + i];
        }
    }
}


#endif // !TRACKING_H
