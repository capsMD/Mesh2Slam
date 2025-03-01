#include "slam.h"

Slam::Slam(SlamParams* params, AAssetManager *assetManager) : m_slamParams(params), m_assetManager(assetManager)
{
    bool initializationOk = true;
    m_slamState = RUNNING;

    try {
        m_map = std::make_shared<Map>();
        if (!m_map) {
            Logger<std::string>::LogError("map initialization failed.");
            initializationOk = false;
        }
        Logger<std::string>::LogInfoI("map initialized.");

        m_tracker = std::make_shared<Tracking>(m_map, m_slamParams);
        if (!m_tracker) {
            Logger<std::string>::LogError("tracker initialization failed.");
            initializationOk = false;
        }
        Logger<std::string>::LogInfoI("tracker initialized.");

        m_viewer = std::make_shared<Viewer>(m_slamParams, assetManager);
        if (!m_viewer) {
            Logger<std::string>::LogError("viewer initialization failed.");
            initializationOk = false;
        }
        m_viewer->setMap(m_map);
        m_viewer->initialize();
        Logger<std::string>::LogInfoI("viewer initialized.");

        m_mapper = std::make_shared<Mapping>(m_map, m_slamParams);
        if (!m_mapper) {
            Logger<std::string>::LogError("mapper initialization failed.");
            initializationOk = false;
        }
        Logger<std::string>::LogInfoI("mapper initialized.");

        if (!initializationOk) {
            m_slamState = SlamState::STOPPED;
            Logger<std::string>::LogError("SLAM initialization failed due to one or more components.");
            throw std::runtime_error("SLAM initialization failed");
        }

        // Set pointers only if all components are initialized
        m_tracker->setMapping(m_mapper);
        m_tracker->setViewer(m_viewer);
        m_mapper->setTracker(m_tracker);
        m_mapper->setViewer(m_viewer);

        m_mappingThread = std::make_shared<std::thread>(&Mapping::run, m_mapper);
        Logger<std::string>::LogInfoI("Mapping thread started.");

        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Optional
    } catch (const std::exception& e)
    {
        m_slamState = SlamState::STOPPED;
        Logger<std::string>::LogError("Exception during SLAM initialization: " + std::string(e.what()));
        throw; // Re-throw to caller
    }
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
    m_tracker->stopTracking();
    m_mapper->stopMapping();
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_frameAvailable = true;
        m_frameCondVariable.notify_one();
    }
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

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_frameAvailable = true;
        m_frameCondVariable.notify_one();
    }

    if (m_mappingThread && m_mappingThread->joinable())
    {
        m_mappingThread->join();
        Logger<std::string>::LogInfoI("Mapper thread terminated.");
    }
    m_viewer->stopViewer();
}

bool SlamManager::initializeSLAM()
{
    bool initializationOk = true;

    if(m_slamParams == nullptr)
    {
        Logger<std::string>::LogError("SLAM parameters not found.");
        initializationOk = false;
    }

    try
    {
        std::cout << "creating SLAM instances." << std::endl;
        m_slam = std::make_shared<Slam>(m_slamParams, m_assetManager);
    }
    catch(const std::exception &e)
    {
        Logger<std::string>::LogError("Failed to initialize SLAM.");
        initializationOk = false;
    }

    if(initializationOk)
    {
        m_slamInitialized.store(true);
        m_slamThread = std::make_shared<std::thread>(&Slam::run, m_slam);
    }
    return initializationOk;
}

void SlamManager::updateFrame()
{
    if((m_slamInitialized) && (m_slam->getSlamState() == SlamState::RUNNING))
    {
        VertexFeatures vertexFeatures;
        if(m_slam->getViewer()->renderVtxFeatures(vertexFeatures))
            m_slam->processFrame(vertexFeatures);
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
        m_slamThread->join();
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

