#include "slam.h"

Slam::Slam(SlamParams* params, AAssetManager *assetManager) : m_slamParams(params), m_assetManager(assetManager)
{
    m_slamState = RUNNING;

    //initialize slam main objects and threads
    m_map =  std::make_shared<Map>();
    m_tracker = std::make_shared<Tracking>(m_map, m_slamParams);
    assert(m_tracker != nullptr && "Tracker initialization failed");
    Logger<std::string>::LogInfoI("tracker initialized.");

    bool runViewer = static_cast<bool>(m_slamParams->viewerParams.runViewer);
    m_viewer = std::make_shared<Viewer>(m_slamParams, assetManager);
    assert(m_viewer != nullptr && "Viewer initialization failed");
    m_viewer->setMap(m_map);
    m_viewer->initialize();
    Logger<std::string>::LogInfoI("viewer initialized.");

    m_mapper = std::make_shared<Mapping>(m_map, m_slamParams);
    assert(m_mapper != nullptr && "Mapper initialization failed");
    m_mappingThread = std::make_shared<std::thread>(&Mapping::run, m_mapper);
    Logger<std::string>::LogInfoI("mapper initialized.");

    //set pointers
    m_tracker->setMapping(m_mapper);
    m_tracker->setViewer(m_viewer);
    m_mapper->setTracker(m_tracker);
    m_mapper->setViewer(m_viewer);

    //give enough time for initializations (probably not needed)
    std::this_thread::sleep_for(std::chrono::milliseconds (200));
}

void Slam::processFrame(const VertexFeatures& vertexFeatures)
{
    std::lock_guard<std::mutex> lock(m_frameMutex);
    m_frameAvailable = true;
    m_frameData = vertexFeatures;
    m_sourceFrameCnt++;
    m_frameCondVariable.notify_one();
}

void Slam::run()
{
    Logger<std::string>::LogInfoIV("Slam is now running.");
    while(!m_stop)
    {
        while (m_slamState == RUNNING)
        {
            //will hold  here until a new frame is available
            std::unique_lock<std::mutex> lock(m_frameMutex);
            while (!m_frameAvailable)
            {
                m_frameCondVariable.wait(lock);
            }

            VertexFeatures vertexFeatures;
            vertexFeatures.m_pts = m_frameData.m_pts;
            vertexFeatures.m_pose = m_frameData.m_pose;
            m_frameAvailable = false;
            lock.unlock();
            m_tracker->run(vertexFeatures, m_timestamp, m_sourceFrameCnt);
        }
    }
}

void Slam::shutdown()
{
    m_slamState = SlamState::STOPPED;

    if (m_mappingThread && m_mappingThread->joinable())
    {
        m_mappingThread->join();
    }

    if (m_viewerThread && m_viewerThread->joinable())
    {
        m_viewerThread->join();
    }

}

void Slam::renderViewer()
{
    m_viewer->render();
}

bool Slam::updateViewer()
{
    if(m_viewer->checkMapUpdateFlag())
    {
        m_viewer->update();
        m_viewer->clearMapUpdateFlag();
        return true;
    }
    return false;
}

void Slam::stopSLAM()
{
    Logger<std::string>::LogInfoI("Stopping SLAM...");
    m_slamState = SlamState::STOPPED;
    m_stop = true;


    m_tracker->stopTracking();
    m_mapper->stopMapping();

    if (m_mappingThread && m_mappingThread->joinable())
    {
        m_mappingThread->join();
        Logger<std::string>::LogInfoI("Mapper thread terminated.");
    }

    m_viewer->stopViewer();

}


bool SlamManager::initializeSLAM()
{
    bool okay = true;

    if(m_slamParams != nullptr)
    {
        std::cout << "creating SLAM instances." << std::endl;

        m_slam = std::make_shared<Slam>(m_slamParams, m_assetManager);

    }

    if(okay)
    {
        m_slamInitialized = okay;
        m_slamThread = std::make_shared<std::thread>(&Slam::run, m_slam);
    }
    return okay;
}


void SlamManager::updateFrame()
{
    if((m_slamInitialized) && (m_slam->getSlamState() == SlamState::RUNNING))
    {
        // cv::Mat displayImage;
        // displayImage = cv::Mat(m_slamParams->viewerParams.width, m_slamParams->viewerParams.height, CV_8UC3, cv::Scalar(0, 0, 0));

        //fetch: projection of vertexes as features (id + uv coord.) and the virtual cam position (used for scale ambiguity)
        VertexFeatures vertexFeatures;
        //if(m_camManager->GetvertexFeatures(vertexFeatures))
        if(m_slam->getViewer()->renderVtxFeatures(vertexFeatures))
        {
            //std::cout << "vtx features: " << std::to_string(vertexFeatures.m_pts.size()) << std::endl;
            // for(auto& pt : vertexFeatures.m_pts)
            //     displayImage.at<cv::Vec3b>(pt.y, pt.z) = cv::Vec3b(255.0f, 255.0f, 255.0f);
            // cv::imshow("cam display",displayImage);
            // cv::waitKey(1);
            // Logger<std::string>::LogInfoI("vertices: " + std::to_string(vertexFeatures.m_pts.size()));
            m_slam->processFrame(vertexFeatures);
        }

        //update slam viewer map
        //m_slam->updateViewer();
    }
}

void SlamManager::renderSLAM()
{
    if(m_slamInitialized)
        m_slam->renderViewer();
}

void SlamManager::stopSLAM()
{
    m_slam->stopSLAM();

    Logger<std::string>::LogInfoI("Slam running rendering-only mode.");
    if (m_slamThread && m_slamThread->joinable())
    {
        //Logger<std::string>::LogInfoI("Slam Thread terminated.");
        //m_slamThread->join();
    }
}

void SlamManager::startSLAM()
{
    if(!m_slamInitialized)
        if(initializeSLAM())
        {
            Logger<std::string>::LogInfoI("SLAM initialized.");
        }
}

