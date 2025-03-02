
#include "viewer.h"


Viewer::Viewer(SlamParams* params, AAssetManager *assetManager): m_slamParams(params),m_assetManager(assetManager)
{
    m_width = m_slamParams->viewerParams.width;
    m_height = m_slamParams->viewerParams.height;
    m_title = m_slamParams->viewerParams.title;
    m_frameColor = m_slamParams->viewerParams.frameColor;
    m_framePathColor = m_slamParams->viewerParams.framePathColor;
    m_tFrameColor = m_slamParams->viewerParams.tFrameColor;
    m_pointCloudColor = m_slamParams->viewerParams.pointCloudColor;
    m_scaleFactor = m_slamParams->viewerParams.scaleFactor;
    m_featuresMaxDepth = m_slamParams->featureParams.featuresMaxDepth;
    m_meshColor = m_slamParams->viewerParams.meshColor;
    m_forceOriginStart = m_slamParams->viewerParams.forceOriginStart;

    m_featuresMaxDepth *= m_scaleFactor;
}

Viewer::~Viewer() {
    shutdown();
    for (std::map<unsigned long int, FrameGizmo*>::iterator it = m_frames.begin(), itEnd = m_frames.end(); it != itEnd; ++it) {
        it->second->deleteBuffers(); // Clean OpenGL resources
        delete it->second;           // Delete the FrameGizmo*
    }
    m_frames.clear();
    if (m_pointCloud) {
        m_pointCloud->deleteBuffers();
        delete m_pointCloud;
    }
    if (m_tframePoints) {
        m_tframePoints->deleteBuffers();
        delete m_tframePoints;
    }
    if (m_framesPath) {
        m_framesPath->deleteBuffers();
        delete m_framesPath;
    }
    if (m_guiSquare) {
        m_guiSquare->deleteBuffers();
        delete m_guiSquare;
    }
    for (std::map<std::string, Shader*>::iterator it = m_shaders.begin(), itEnd = m_shaders.end(); it != itEnd; ++it) {
        if (it->second) {
            it->second->cleanup(); // Use public cleanup()
            delete it->second;
        }
    }
    m_shaders.clear();
    if (m_mesh) {
        m_mesh->cleanup(); // Use public cleanup() from TriangleMesh
        delete m_mesh;
    }
    if (m_vtxFeature) {
        m_vtxFeature->cleanup(); // Use public cleanup()
        delete m_vtxFeature;
    }
    // mp_pointClouds (unique_ptrs) clears itself automatically
}


bool Viewer::initialize()
{

    Logger<std::string>::LogInfoI("Initializing viewer...");
    std::cout << "Initializing viewer..." << std::endl;
    printVersions();
    initializeProjectionMatrix();

    initializeShaders();
    std::cout << "shaders Initialized." << std::endl;

    initializeMapPoints();
    std::cout << "cloudmap Initialized." << std::endl;

    initializeGizmos();
    //std::cerr << "gizmos Initialized." << std::endl;

    update();
    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
    return true;
}

void Viewer::render()
{

    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
    m_mMatrix = glm::mat4(1.0f);
    setMatrices();

    //render frames
    auto& basicShader = m_shaders.find("basicShader")->second;
    basicShader->use();
    for(std::map<unsigned long int, FrameGizmo* >::iterator it = m_frames.begin(); it != m_frames.end(); it++)
    {
        m_mMatrix = it->second->getPose();
        setMatrices();
        basicShader->setUniform("vRGB", m_frameColor);
        basicShader->setUniform("mvpMatrix", m_mvpMatrix);
        it->second->render();
    }
    glUseProgram(0);


    //render frame axis gizmos
    auto& colorVtxShader = m_shaders.find("colorVtxShader")->second;
    colorVtxShader->use();
    for(auto it = m_frames.begin(); it != m_frames.end(); it++)
    {
        m_mMatrix = it->second->getPose();
        setMatrices();
        colorVtxShader->setUniform("mvpMatrix", m_mvpMatrix);
        it->second->renderAxisGizmos();
    }
    glUseProgram(0);

    //render frame paths
    if(m_framesPath->getNumberOfElements()>1) {
        basicShader->use();
        m_mMatrix = glm::mat4(1.0f);
        setMatrices();
        basicShader->setUniform("vRGB", m_framePathColor);
        basicShader->setUniform("mvpMatrix", m_mvpMatrix);
        m_framesPath->render();
        glUseProgram(0);
    }

    //render tframes
    if(m_tframePoints->getNumberOfElements()>0) {
        basicShader->use();
        //m_mMatrix = glm::mat4(1.0f) * m_scaleFactor;
        m_mMatrix = glm::mat4(1.0f);
        m_mMatrix[3].w = 1.0f;
        setMatrices();
        basicShader->setUniform("vRGB", m_tFrameColor);
        basicShader->setUniform("mvpMatrix", m_mvpMatrix);
        m_tframePoints->render();
        glUseProgram(0);
    }


    //render points
    if(m_pointCloud->getNumberOfElements()>0) {
        basicShader->use();
        //m_mMatrix = glm::mat4(1.0f) * m_scaleFactor;
        m_mMatrix = glm::mat4(1.0f);
        m_mMatrix[3].w = 1.0f;
        setMatrices();
        basicShader->setUniform("vRGB", m_pointCloudColor);
        basicShader->setUniform("mvpMatrix", m_mvpMatrix);
        m_pointCloud->render();
        glUseProgram(0);
    }


    //render mesh
    if(m_mesh != nullptr)
    {
        auto& worldPtsShader = m_shaders.find("worldPtsShader")->second;
        worldPtsShader->use();
        glm::mat4 meshScale (1.0f);
        meshScale*= m_scaleFactor;
        meshScale[3].w = 1.0f;
        m_mMatrix = glm::mat4(1.0f);
        m_mMatrix *= meshScale;
        setMatrices();
        worldPtsShader->setUniform("mvpMatrix", m_mvpMatrix);
        worldPtsShader->setUniform("mvMatrix", m_vMatrix);
        worldPtsShader->setUniform("u_ColorNear", glm::vec3(0.3f,0.3f,0.3f));
        worldPtsShader->setUniform("u_ColorFar", glm::vec3(0.8f,0.8f,0.8f));
        worldPtsShader->setUniform("u_MaxDistance", 0.020f);

        m_mesh->render(0);
        glUseProgram(0);
    }




    //render GUI Squares
    glDisable(GL_DEPTH_TEST);
    basicShader->use();
    glm::mat4 idMatrix(1.0f);
    basicShader->setUniform("vRGB", m_guiSquare->getColor());
    basicShader->setUniform("mvpMatrix", idMatrix);
    m_guiSquare->render();
    glUseProgram(0);

    glEnable(GL_DEPTH_TEST);
}

bool Viewer::renderVtxFeatures(VertexFeatures& vtxFeatures)
{
    if(!m_vtxFeaturesInitialized)
        return false;

    if(!m_originCaptured)
    {
        m_vMatrix = glm::mat4(1.0f);
        m_originCaptured = true;
    }

    size_t N = m_vtxFeature->getSize();
    //render frames
    auto& vtxFeatureShader = m_shaders.find("vtxFeatureShader")->second;
    vtxFeatureShader->use();
    vtxFeatureShader->setUniform("mMatrix", m_mMatrix);
    vtxFeatureShader->setUniform("vMatrix", m_vMatrix);
    vtxFeatureShader->setUniform("pMatrix", m_p);
    vtxFeatureShader->setUniform("screenSize", glm::vec2(m_width, m_height));
    vtxFeatureShader->setUniform("maxDepth", m_featuresMaxDepth);


    GLint maxGroupSize;
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxGroupSize); // GPU limit (e.g., 1024)
    GLuint workGroupSize = 256; // Match shader’s local_size_x (check vtxFeatureShader.comp)
    if (workGroupSize > maxGroupSize) workGroupSize = maxGroupSize;
    GLuint numGroups = (N + workGroupSize - 1) / workGroupSize; // Enough groups for N items

    glDispatchCompute(numGroups, 1, 1);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "Compute failed: " << err << std::endl;
        return false;
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_vtxFeature->readOutputBuffer(vtxFeatures.m_pts);
    vtxFeatures.m_pose = m_vMatrix;
    return !vtxFeatures.m_pts.empty();
    return(!vtxFeatures.m_pts.empty());
}

void Viewer::update()
{
    if(!m_stop)
    {
        if (!m_map) return;
        //fetch frames and pts form map
        std::vector<MapPoint*> p_mapPoints = m_map->getMapPoints();
        std::vector<Frame*> p_frames = m_map->getFrames();
        std::vector<TFrame*> p_tframes = m_map->getTFrames();

        std::vector<glm::vec3> points;
        size_t N = p_mapPoints.size();
        points.reserve(N);

        std::vector<glm::vec3> points2;

        for(auto& mp : p_mapPoints)
        {
            if(mp == nullptr)
                continue;

            points.emplace_back(mp->getPosition().at<float>(0),
                                -mp->getPosition().at<float>(1),
                                -mp->getPosition().at<float>(2));
        }
        //update already initialized buffers
        m_pointCloud->updatePoints(points);
        updateFrames(p_frames);
        updateTFrames(p_tframes);
        updatePaths();
    }


}

void Viewer::printVersions()
{
    const GLubyte *renderer = glGetString(GL_RENDERER);
    const GLubyte *vendor = glGetString(GL_VENDOR);
    const GLubyte *version = glGetString(GL_VERSION);
    const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

    GLint major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);


    printf("GL Vendor              : %s\n", vendor);
    printf("GL Renderer            : %s\n", renderer);
    printf("GL Version (string)    : %s\n", version);
    printf("GL Version (integeger) : %d.%d\n", major, minor);
    printf("GLSL Version           : %s\n", glslVersion);


    //query for supported extensions of the current OpenGL implementation
    bool logExtensions = false;
    if (logExtensions)

    {
        GLint nExtensions;
        glGetIntegerv(GL_NUM_EXTENSIONS, &nExtensions);

        for (int i = 0; i < nExtensions; i++)
            printf("%s\n", glGetStringi(GL_EXTENSIONS, i));
    }
}

void Viewer::shutdown()
{
    //TODO: Need to adapt this code
    if (m_eglDisplay != EGL_NO_DISPLAY)
    {
        eglMakeCurrent(m_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (m_eglContext != EGL_NO_CONTEXT)
        {
            eglDestroyContext(m_eglDisplay, m_eglContext);
            m_eglContext = EGL_NO_CONTEXT;
        }
        if (m_eglSurface != EGL_NO_SURFACE)
        {
            eglDestroySurface(m_eglDisplay, m_eglSurface);
            m_eglSurface = EGL_NO_SURFACE;
        }
        eglTerminate(m_eglDisplay);
        m_eglDisplay = EGL_NO_DISPLAY;
    }

//    if (m_window != nullptr) {
//        SDL_DestroyWindow(m_window);
//        m_window = nullptr;
//        SDL_Quit();
//    }
}

void Viewer::exit()
{
    shutdown();
}

void Viewer::initializeShaders()
{
    GLuint shaderProgram = glCreateProgram();
    Shader* shaderSimpleWhite = new Shader();
    shaderSimpleWhite->setHandle(shaderProgram);

    std::cout << "attempting to compiles SLAM shaders..." << std::endl;
    shaderSimpleWhite->compile(GL_VERTEX_SHADER, "shaders/basicShader.vert", m_assetManager);
    shaderSimpleWhite->compile(GL_FRAGMENT_SHADER, "shaders/basicShader.frag",m_assetManager);
    shaderSimpleWhite->link();
    m_shaders["basicShader"] = shaderSimpleWhite;
    std::cout << "basic shader loaded." << std::endl;

    shaderProgram = glCreateProgram();
    Shader* shaderSimpleColor = new Shader();
    shaderSimpleColor->setHandle(shaderProgram);
    shaderSimpleColor->compile(GL_VERTEX_SHADER, "shaders/coloredVtxShader.vert",m_assetManager);
    shaderSimpleColor->compile(GL_FRAGMENT_SHADER, "shaders/coloredVtxShader.frag",m_assetManager);
    shaderSimpleColor->link();
    m_shaders["colorVtxShader"] = shaderSimpleColor;
    std::cout << "colored shader loaded." << std::endl;

    //world pts shader
    shaderProgram = glCreateProgram();
    Shader* worldPtsShader = new Shader();
    worldPtsShader->setHandle(shaderProgram);
    worldPtsShader->compile(GL_VERTEX_SHADER, "shaders/worldPtsShader.vert",m_assetManager);
    worldPtsShader->compile(GL_FRAGMENT_SHADER, "shaders/worldPtsShader.frag",m_assetManager);
    worldPtsShader->link();
    m_shaders["worldPtsShader"] = worldPtsShader;




    //compute shader
    shaderProgram = glCreateProgram();
    Shader* vtxFeature = new Shader();
    vtxFeature->setHandle(shaderProgram);
    vtxFeature->compile(GL_COMPUTE_SHADER, "shaders/vtxFeatureShader.comp", m_assetManager);
    vtxFeature->link();
    m_shaders["vtxFeatureShader"] = vtxFeature;
    std::cout << "compute shader loaded." << std::endl;
}

void Viewer::initializeBuffers()
{
    glGenFramebuffers(1, &m_renderFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_renderFBO);

    glGenRenderbuffers(1, &m_depthFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, m_depthFBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_width, m_height);

    GLenum drawBuffers[] = { GL_NONE};
    glDrawBuffers(1, drawBuffers);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Viewer::setModelManager(ModelManager* modelManager)
{
    if(modelManager== nullptr)
    {
        std::cout << "Model manager is null!" << std::endl;
    }
    m_modelManager = modelManager;
    std::cout << "Setting model manager to viewer" << std::endl;
    if(m_mesh)
    {
        m_mesh->cleanup();
        delete m_mesh;
    }
    m_mesh = m_modelManager->getModelMesh();
    if(m_mesh == nullptr)
    {
        std::cout << "Oh no, model mesh is nullptr" << std::endl;
        return;
    }
    initializeVtxFeature(modelManager->getModelPts());
}

void Viewer::initializeMapPoints()
{
    m_pointCloud = new PointCloud();
    m_pointCloud->initializeEmptyBuffer();

    m_tframePoints = new PointCloud();
    m_tframePoints->initializeEmptyBuffer();

}

void Viewer::setMatrices()
{
    //because here we use OpenXR's matrices
    //m_vMatrix = mp_activeCamera->getViewMatrix();
    //m_pMatrix = mp_activeCamera->getProjectionMatrix();
    m_mvpMatrix = m_pMatrix * m_vMatrix * m_mMatrix;
}

void Viewer::initializeGizmos()
{
    m_framesPath = new PathGizmo();
    m_framesPath->initializeEmptyBuffer();
    m_guiSquare = new GUISquaresGizmo();
    m_guiSquare->initialize();
    m_guiSquare->updateColor(glm::vec3(0.0f,0.0f,1.0f));
}

void Viewer::initializeVtxFeature(const std::vector<glm::vec4>& modelVertexes)
{
    if(m_vtxFeature)
    {
        m_vtxFeature->cleanup();
        delete m_vtxFeature;
    }
    if(m_vtxFeature == nullptr)
        m_vtxFeature = new VtxFeature();

    std::cout << "Setting vtx features: " << std::to_string(modelVertexes.size()) << std::endl;

    //TODO: this should return a bool, to assert it was properly initialized
    m_vtxFeature->initialize(modelVertexes);
    m_vtxFeaturesInitialized = true;
}

void Viewer::updateFrames(const std::vector<Frame *>& newFrames)
{
    const size_t N = newFrames.size();

    //update map of frames, if frame not included, include it
    for(size_t n = 0; n < N; n++)
    {
        unsigned long int id = newFrames[n]->getID();
        //if new frame

        glm::mat4 pose(1.0f);
        cv::Mat framePose = newFrames[n]->getPosew();

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                pose[i][j] = framePose.at<float>(i, j);

        pose = glm::transpose(pose);

        glm::mat3 R(1.0f);
        R[0] = pose[0];
        R[1] = pose[1];
        R[2] = pose[2];

        glm::vec3 t(0.0f);
        t = pose[3] * 1.0f;
        t = ViewerUtil::convertTranslation(t,glm::vec3 (1,-2,-3));

        // Create a rotation matrix for a 180-degree rotation around the x-axis
        glm::mat3 rot180x(1.0f);
        rot180x[1][1] = -1.0f;
        rot180x[2][2] = -1.0f;

        // Apply the rotation to correct the orientation
        R = rot180x * R;

        // Reconstruct the pose matrix with corrected rotation and original translation
        pose[0] = glm::vec4(R[0], 0.0f);
        pose[1] = glm::vec4(R[1], 0.0f);
        pose[2] = glm::vec4(R[2], 0.0f);
        pose[3] = glm::vec4(t, 1.0f);

        glm::mat4 scaleMatrix = glm::mat4(1.0f) * m_scaleFactor;
        scaleMatrix[3].w = 1.0f;
        pose = pose * scaleMatrix;


        if(!m_frames.count(id))
        {
            FrameGizmo* tempFrame = new FrameGizmo(0, pose, id);
            tempFrame->initialize();

            if(!m_frames.empty())
            {
                tempFrame->setParentNode(std::prev(m_frames.end())->second);
            }
            m_frames[newFrames[n]->getID()] = tempFrame;

        }
        else
        {
            m_frames[id]->setPose(pose);
        }
    }
}

void Viewer::updateTFrames(const std::vector<TFrame* >& tFrames)
{
    const size_t N = tFrames.size();
    std::vector<glm::vec3> pts = std::vector(N,glm::vec3(0.0f));

    //update map of frames, if frame not included, include it
    for(size_t n = 0; n < N; n++)
    {
        unsigned long int id = tFrames[n]->getFrameID();
        //if new frame

        glm::mat4 pose(1.0f);
        cv::Mat framePose = tFrames[n]->getPosew();

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                pose[i][j] = framePose.at<float>(i, j);

        pose = glm::transpose(pose);

        glm::vec3 t(0.0f);
        t = pose[3];
        t = ViewerUtil::convertTranslation(t, glm::vec3(1, -2, -3));

        pts.emplace_back(t);
    }
    m_tframePoints->updatePoints(pts);
}

void Viewer::updatePaths()
{
    std::vector<glm::vec3> pathNodes;

    for(std::map<unsigned long int,FrameGizmo* >::iterator it = m_frames.begin(); it!=m_frames.end();it++)
    {
        FrameGizmo* frameGizmo = it->second;
        if(frameGizmo->getParentNode()!= nullptr)
        {
            pathNodes.push_back(frameGizmo->getParentNode()->getPosition());
            pathNodes.push_back(frameGizmo->getPosition());
        }
    }

    m_framesPath->updatePoints(pathNodes);
}

void Viewer::setMapUpdateFlag()
{
    m_mapUpdateFlag.store(true);
}

void Viewer::setSquareUpdateFlag(const char& state)
{
    std::unique_lock<std::mutex> lock(m_viewerMutex);
    {
        switch (state) {
            case 1: //startup
            {
                m_guiSquare->updateColor(glm::vec3(0.0f, 0.0f, 1.0f));
                break;
            }
            case 2://tracking
            {
                m_guiSquare->updateColor(glm::vec3(0.0f, 1.0f, 0.0f));
                break;
            }
            case 3://lost
            {
                m_guiSquare->updateColor(glm::vec3(1.0f, 0.0f, 0.0f));
                break;
            }
            case 4://recovery
            {
                m_guiSquare->updateColor(glm::vec3(0.0f, 1.0f, 1.0f));
                break;
            }
        }
    }
}

bool Viewer::checkMapUpdateFlag()
{
    return m_mapUpdateFlag.load();
}

void Viewer::clearMapUpdateFlag()
{
    m_mapUpdateFlag.store(false);
}


void Viewer::stopViewer()
{
    std::lock_guard<std::mutex> lock(m_mutexUpdate);
    m_stop = true;
    Logger<std::string>::LogInfoI("Viewer updates stopped.");
}

bool Viewer::setActiveCamera(Camera* camera)
{
    if(camera != nullptr)
    {
        m_activeCamera = camera;
        return true;
    }
    return false;
}

bool Viewer::getVtxFeatures(const float maxDepth,std::vector<cv::Point3i>& pts)
{
    pts.clear();
    bool ok = false;
    size_t count = 0;

    if(m_mesh == nullptr)
        return false;

    //assuming all matrices are ready
    std::vector<glm::vec3> points = m_mesh->getMeshData()->points;
    size_t N = points.size();
    for (size_t i = 0; i < N; i++)
    {
        //opengl
        glm::vec3 dataVtxPos = points[i];

        //model
        glm::vec4 vtxM = m_mMatrix * glm::vec4(dataVtxPos, 1.0);

        //get vtx wrt camera for depth only
        glm::vec4 mvVtx = m_vMatrix * vtxM;

        //get projection to plane for u and v only
        glm::vec4 mvpvtx = m_p * m_vMatrix * vtxM;

        //to NDC
        float x = (mvpvtx.x / mvpvtx.w);
        float y = (mvpvtx.y / mvpvtx.w);
        float z  = - (mvVtx.z);

        //now x and y to screen coordinates
        float u = (x * 0.5f + 0.5f) * m_width;
        float v = (y * -0.5f+0.5f)  * m_height;

        if ((z > maxDepth) || (z < 0.0f))
            continue;
        if ((u <= 0 || u >= m_width) || (v <= 0 || v >= m_height))
            continue;

        pts.push_back(cv::Point3i (i,u,v));
        count++;
    }

    ok = static_cast<bool>(count);

    return ok;
}

void Viewer::loadMesh(const std::string &meshFile, AAssetManager *assetManager)
{
    if(m_mesh)
    {
        m_mesh->cleanup();
        delete m_mesh;
    }
    m_mesh = ObjMesh::load(meshFile.c_str(), assetManager);
}

void Viewer::initializeProjectionMatrix()
{
    m_p = glm::perspective(glm::radians(m_fov),
                           static_cast<float>(m_width)/m_height,
                           m_near,
                           m_far);

    std::cout << "Printing m_p matrix: " << std::endl;

    for(size_t i = 0; i < 4; i++)
        for(size_t j = 0; j < 4; j++)
            std::cout << std::to_string(m_p[j][i]) << std::endl;

    m_k = glm::mat4(1.0f);
    m_k[0][0] = 512;
    m_k[1][1] = 512;
    m_k[0][2] = 512;
    m_k[1][2] = 512;

}


void ViewerUtil::convertToGL(const std::vector<glm::vec3> &points, std::vector<GLfloat> &glPoints)
{
    glPoints.clear();
    for(size_t i = 0; i < points.size(); i++)
    {
        glPoints.push_back(points[i].x);
        glPoints.push_back(points[i].y);
        glPoints.push_back(points[i].z);
    }
}
//********************************************************  SHADER ********************************************************


bool Shader::link()
{
    if (m_isLinked)
    {
        std::cout<< "Shader program has already been linked!" << std::endl;
        return false;
    }
    if (m_shaderProgram <= 0)
    {
        std::cout<<"Link error: Shader program has not been created!"<< std::endl;
        return false;
    }

    //atach compiled shaders
    for (size_t iLoop = 0; iLoop < m_compiledShaders.size(); iLoop++)
        glAttachShader(m_shaderProgram, m_compiledShaders[iLoop]);


    //link
    glLinkProgram(m_shaderProgram);

    GLint status;
    glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetProgramiv(m_shaderProgram, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetProgramInfoLog(m_shaderProgram, infoLogLength, NULL, strInfoLog);
        fprintf(stderr, "Linker failure: %s\n", strInfoLog);
        std::cout << strInfoLog << std::endl;
        delete[] strInfoLog;
        return false;
    }

        //Link successful, get uniforms and set linked
    else
    {
        findUniformLocations();
        findAttributeLocations();
        m_isLinked = true;
    }

    //in either case, detach shader objects
    detachAndDeleteShaders();

    return true;
}

std::string Shader::readFile(const std::string &path) {
    std::string shaderSource;
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Error! File not found or could not be opened! " + path << std::endl;
        return "";
    } else {
        std::cout << "Shader File found: " + path << std::endl;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    f.close();
    int length = 0;
    if (ss) {
        ss.seekg(0, ss.end);
        length = ss.tellg();
        ss.seekg(0, ss.beg);
        shaderSource = ss.str();
        return shaderSource;
    } else {
        std::string error = "Error! File not found or could not be opened!" + path;
        return "";
    }
}

bool Shader::compile(GLenum shaderType, const std::string &shaderSrcFile) {
    std::string shaderSource;
    GLuint shader = glCreateShader(shaderType);
    shaderSource = readFile(shaderSrcFile); // Calls the fixed readFile below
    const char* strFileData = shaderSource.c_str();

    if ((unsigned int)m_shaderProgram <= 0) {
        m_shaderProgram = glCreateProgram();
        if (m_shaderProgram == 0) {
            std::cout << "Unable to create shader program." << std::endl;
        }
        return false;
    }

    glShaderSource(shader, 1, &strFileData, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLint infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);

        const char* strShaderType;
        switch (shaderType) {
            case GL_VERTEX_SHADER: strShaderType = "vertex"; break;
            case GL_GEOMETRY_SHADER: strShaderType = "geometry"; break;
            case GL_FRAGMENT_SHADER: strShaderType = "fragment"; break;
            case GL_COMPUTE_SHADER: strShaderType = "compute"; break;
        }

        fprintf(stderr, "Compile failure in %s shader:\n%s\n", strShaderType, strInfoLog);
        delete[] strInfoLog;

        std::cout << "Compile failure in" + std::string(strShaderType) + "in shader: " + strInfoLog << std::endl;

        return false;
    }

    m_compiledShaders.push_back(shader);
    return true;
}

//used by android
bool Shader::compile(GLenum shaderType, const std::string &shaderSrcFile, AAssetManager *assetManager)
{
    std::string shaderSource;
    GLuint shader = glCreateShader(shaderType);
    std::cout << "Reading Slam shaders: " + shaderSrcFile << std::endl;
    if(assetManager == nullptr)
        std::cout << "assetManager is fucked" << std::endl;

    shaderSource = ReadTextFile(shaderSrcFile,assetManager);
    std::cout << "Text file has been read!" << std::endl;
    if(shaderSource.empty())
        std::cout << "Error reading file!" << std::endl;
    const char* strFileData = shaderSource.c_str();

    if ((unsigned int) m_shaderProgram <= 0) {
        m_shaderProgram = glCreateProgram();
        if (m_shaderProgram == 0) {
            std::cout << "Unable to create shader program." << std::endl;
        }
    }

    glShaderSource(shader, 1, &strFileData, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);

        const char* strShaderType;
        switch (shaderType)
        {
            case GL_VERTEX_SHADER: strShaderType = "vertex"; break;
            case GL_GEOMETRY_SHADER: strShaderType = "geometry"; break;
            case GL_FRAGMENT_SHADER: strShaderType = "fragment"; break;
        }

        fprintf(stderr, "Compile failure in %s shader:\n%s\n", strShaderType, strInfoLog);
        delete[] strInfoLog;

        std::cout << "Compile failure in" + std::string(strShaderType) + "in shader: " + strInfoLog << std::endl;

        return false;
    }

    m_compiledShaders.push_back(shader);
    return true;
}

void Shader::detachAndDeleteShaders()
{
    GLint numberOfShaders = 0;
    glGetProgramiv(m_shaderProgram, GL_ATTACHED_SHADERS, &numberOfShaders);
    std::vector<GLuint> shaderNames(numberOfShaders);
    glGetAttachedShaders(m_shaderProgram, numberOfShaders, NULL, shaderNames.data());
    for (GLuint attachedShader : shaderNames)
    {
        glDetachShader(m_shaderProgram, attachedShader);
        glDeleteShader(attachedShader);
    }
}

void Shader::findUniformLocations()
{
    m_uniformLocations.clear();

    GLint i;
    GLint count;
    GLint size; // size of the variable
    GLenum type; // type of the variable (float, vec3 or mat4, etc)

    const GLsizei bufSize = 64; // maximum name length
    GLchar name[bufSize]; // variable name in GLSL
    GLsizei length; // name length
    glGetProgramiv(m_shaderProgram, GL_ACTIVE_UNIFORMS, &count);
    for (i = 0; i < count; i++)
    {
        glGetActiveUniform(m_shaderProgram, (GLuint)i, bufSize, &length, &size, &type, name);

        printf("Uniform #%d Type: %u Name: %s\n", i, type, name);
        m_uniformLocations[name] = glGetUniformLocation(m_shaderProgram, name);
    };


}

void Shader::findAttributeLocations()
{

    GLint i;
    GLint count;
    GLint size; // size of the variable
    GLenum type; // type of the variable (float, vec3 or mat4, etc)

    const GLsizei bufSize = 16; // maximum name length
    GLchar name[bufSize]; // variable name in GLSL
    GLsizei length; // name length
    glGetProgramiv(m_shaderProgram, GL_ACTIVE_ATTRIBUTES, &count);
    printf("Active Attributes: %d\n", count);

    for (i = 0; i < count; i++)
    {
        glGetActiveAttrib(m_shaderProgram, (GLuint)i, bufSize, &length, &size, &type, name);

        printf("Attribute #%d Type: %u Name: %s\n", i, type, name);
        m_attributeLocations[name] = glGetAttribLocation(m_shaderProgram, name);;
    }
}

void Shader::setUniform(const char *name, const glm::mat4 &m)
{
    GLint loc = getUniformLocation(name);
    glUniformMatrix4fv(loc, 1, GL_FALSE, &m[0][0]);
}

void Shader::setUniform(const char *name, const glm::vec3 &v)
{
    GLint loc = getUniformLocation(name);
    glUniform3f(loc, v.x, v.y, v.z);
}

void Shader::setUniform(const char *name, const glm::vec2 &v)
{
    GLint loc = getUniformLocation(name);
    glUniform2f(loc, v.x, v.y);
}

void Shader::setUniform(const char *name, float val)
{
    GLint loc = getUniformLocation(name);
    glUniform1f(loc, val);
}

int Shader::getUniformLocation(const char *name)
{
    //traverse map
    auto index = m_uniformLocations.find(name);

    //not found, means uniform was not included
    if (index == m_uniformLocations.end())
    {
        //case uniform is present, include in map
        if (GLint loc = glGetUniformLocation(m_shaderProgram, name) > 0)
        {
            m_uniformLocations[name] = loc;
            return loc;
        }
        else
        {
            std::string output = name;
            output = "uniform: " + output + " not found!";
            fprintf(stderr, "%s", output.c_str());
            return -1;
        }
    }
    return index->second;
}

bool Shader::setHandle(GLuint handle)
{
    if(handle !=0)
    {
        m_shaderProgram = handle;
        return true;
    }
    else
        std::cerr << "invalid shader program!" << std::endl;
    return false;
}

void Shader::cleanup()
{
    detachAndDeleteShaders();
    glDeleteProgram(m_shaderProgram);
    m_shaderProgram = 0;
}


//********************************************************  CAMERA ********************************************************

void Camera::setTransform()
{
    glm::mat4 posMat(1.0f);
    posMat[3] = glm::vec4(-m_position, 1.0f);

    glm::mat4 rotMat(1.0);
    rotMat[0] = glm::vec4(m_right, 0.0f);
    rotMat[1] = glm::vec4(m_up, 0.0f);
    rotMat[2] = glm::vec4(-m_forward, 0.0f);
    rotMat = glm::transpose(rotMat);

    m_transform = rotMat * posMat;
    m_viewMatrix = m_transform;
}



//********************************************************  GL ELEMENTS ********************************************************

void GLPrimitive::initialize()
{

}

void GLPrimitive::loadPoints(const std::vector<glm::vec3> &points, const std::vector<glm::vec3> &pointsColor)
{
    std::vector<GLfloat> glPoints;
    std::vector<GLfloat> glPointsColor;

    ViewerUtil::convertToGL(points,glPoints);
    ViewerUtil::convertToGL(pointsColor,glPointsColor);
    initializeBuffers(&glPoints,&glPointsColor);
}

void GLPrimitive::loadPoints(const std::vector<glm::vec3> &points)
{
    std::vector<GLfloat> glPoints;
    ViewerUtil::convertToGL(points,glPoints);
    initializeBuffers(&glPoints);
}

void GLPrimitive::initializeBuffers(std::vector<GLfloat> *glPoints, std::vector<GLfloat> *glpointsColors)
{
    if(glPoints == nullptr || glpointsColors== nullptr)
        return;

    GLuint posBuffer = 0, colorBuffer = 0;
    int vtxPosAttributeIndex = 0;
    int vtxColorAttributeIndex = 1;
    m_N = glPoints->size();


    glGenBuffers(1, &posBuffer);
    m_buffers.push_back(posBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
    glBufferData(GL_ARRAY_BUFFER, glPoints->size() * sizeof(GLfloat), glPoints->data(), GL_STATIC_DRAW);

    glGenBuffers(1, &colorBuffer);
    m_buffers.push_back(colorBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, glpointsColors->size() * sizeof(GLfloat), glpointsColors->data(), GL_STATIC_DRAW);

    glGenVertexArrays( 1, &m_vao );
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
    glVertexAttribPointer(vtxPosAttributeIndex, 3, GL_FLOAT, GL_FALSE, 0, 0 );
    glEnableVertexAttribArray(vtxPosAttributeIndex);  // Vertex position

    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glVertexAttribPointer(vtxColorAttributeIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(vtxColorAttributeIndex);

    glBindVertexArray(0);
}

void GLPrimitive::initializeBuffers(std::vector<GLfloat> *glPoints)
{

    if(glPoints == nullptr)
        return;

    GLuint posBuffer = 0, colorBuffer = 0;
    m_N = glPoints->size()/3;

    glGenBuffers(1, &posBuffer);
    m_buffers.push_back(posBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
    glBufferData(GL_ARRAY_BUFFER, glPoints->size() * sizeof(GLfloat), glPoints->data(), GL_STATIC_DRAW);

    glGenVertexArrays( 1, &m_vao );
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0 );
    glEnableVertexAttribArray(0);  // Vertex position

    glBindVertexArray(0);
}

void GLPrimitive::render() const
{
    if (m_vao != 0)
    {
        glBindVertexArray(m_vao);
        glDrawArrays(GL_LINES, 0, m_N);
        glBindVertexArray(0);
    }
}

void GLPrimitive::deleteBuffers()
{
    if( m_buffers.size() > 0 )
    {
        glDeleteBuffers( (GLsizei)m_buffers.size(), m_buffers.data() );
        m_buffers.clear();
    }

    if( m_vao != 0 )
    {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
}

void GLPrimitive::updateBuffer(const std::vector<GLfloat> *glPoints)
{
    m_N = glPoints->size();
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, glPoints->size() * sizeof(GLfloat), glPoints->data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);
}

void GLPrimitive::initializeEmptyBuffer()
{

// Generate VAO
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

// Generate VBO
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);


// Initialize VBO with an empty buffer
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

// Set up vertex attributes (if needed)
// For example, assuming points are 3D coordinates stored as floats:
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr);
    glEnableVertexAttribArray(0);

// Unbind VAO and VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    m_buffers.push_back(m_vbo);

}

void GLPrimitive::initializeBuffers(const std::vector<GLfloat> *points, const std::vector<GLuint> *indices)
{
    if( ! m_buffers.empty() ) deleteBuffers();

    m_N = points->size()/2;
    int vtxAttributeIndex = 0;


    // Must have data for indices, points
    if( indices== nullptr || points == nullptr  )
    {
        return;
    }


    GLuint indexBuf = 0, posBuf = 0, normBuf = 0, tcBuf = 0, tangentBuf = 0, vertexColorsBuf = 0;
    glGenBuffers(1, &indexBuf);
    m_buffers.push_back(indexBuf);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuf);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices->size() * sizeof(GLuint), indices->data(), GL_STATIC_DRAW);

    glGenBuffers(1, &posBuf);
    m_buffers.push_back(posBuf);
    glBindBuffer(GL_ARRAY_BUFFER, posBuf);
    glBufferData(GL_ARRAY_BUFFER, points->size() * sizeof(GLfloat), points->data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuf);

    // Position
    glBindBuffer(GL_ARRAY_BUFFER, posBuf);
    glVertexAttribPointer(vtxAttributeIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(vtxAttributeIndex);  // Vertex position
    vtxAttributeIndex++;

    glBindVertexArray(0);
}

void AxisGizmo::initialize()
{
    std::vector<glm::vec3> points;
    points.push_back(glm::vec3(0, 0, 0));
    points.push_back(glm::vec3(1, 0, 0));

    points.push_back(glm::vec3(0, 0, 0));
    points.push_back(glm::vec3(0, -1, 0));

    points.push_back(glm::vec3(0, 0, 0));
    points.push_back(glm::vec3(0, 0, 1));

    std::vector<glm::vec3> pointsColors;
    pointsColors.push_back(glm::vec3(1, 0, 0));
    pointsColors.push_back(glm::vec3(1, 0, 0));

    pointsColors.push_back(glm::vec3(0, 1, 0));
    pointsColors.push_back(glm::vec3(0, 1, 0));

    pointsColors.push_back(glm::vec3(0, 0, 1));
    pointsColors.push_back(glm::vec3(0, 0, 1));
    loadPoints(points,pointsColors);
}

void FrameGizmo::initialize()
{
    //update position and rotation matrix
    m_R[0] = m_pose[0];
    m_R[1] = m_pose[1];
    m_R[2] = m_pose[2];
    m_t    = m_pose[3];


    std::vector<glm::vec3> points;
    //top
    points.push_back(glm::vec3(-0.5, 0.5, 0.5));
    points.push_back(glm::vec3( 0.5, 0.5, 0.5));

    //right
    points.push_back(glm::vec3( 0.5, 0.5, 0.5));
    points.push_back(glm::vec3( 0.5, -0.5, 0.5));

    //bottom
    points.push_back(glm::vec3( 0.5, -0.5, 0.5));
    points.push_back(glm::vec3( -0.5, -0.5, 0.5));

    //left
    points.push_back(glm::vec3( -0.5, -0.5, 0.5));
    points.push_back(glm::vec3( -0.5, 0.5, 0.5));

    //center
    points.push_back(glm::vec3(0.0, 0.0, 0.0));
    points.push_back(glm::vec3(-0.5, 0.5, 0.5));

    points.push_back(glm::vec3(0.0, 0.0, 0.0));
    points.push_back(glm::vec3(0.5, 0.5, 0.5));

    points.push_back(glm::vec3(0.0, 0.0, 0.0));
    points.push_back(glm::vec3(0.5, -0.5, 0.5));

    points.push_back(glm::vec3(0.0, 0.0, 0.0));
    points.push_back(glm::vec3(-0.5, -0.5, 0.5));


    loadPoints(points);
}

void FrameGizmo::setParentNode(FrameGizmo* frameGizmo)
{
    m_parent = frameGizmo;
}

void FrameGizmo::renderAxisGizmos() const
{
    m_axisGizmo.render();
}

void PathGizmo::updatePoints(const std::vector<glm::vec3>& points)
{
    m_N = points.size();
    std::vector <GLfloat> glPoints;
    ViewerUtil::convertToGL(points,glPoints);
    updateBuffer(&glPoints);
}

void PathGizmo::render() const
{
    GLPrimitive::render();
}

void TrailGizmo::render() const
{
    if(m_vao == 0) return;

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, m_N);
    glBindVertexArray(0);
}

void PointCloud::render() const
{
    if(m_vao == 0) return;

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, m_N);
    glBindVertexArray(0);
}

//TODO: use single conversion for points and frames etc.
void PointCloud::updatePoints(const std::vector<glm::vec3> &points)
{
    std::vector <GLfloat> glPoints;
    ViewerUtil::convertToGL(points,glPoints);
    updateBuffer(&glPoints);
}

void GUISquaresGizmo::initialize()
{
    //update position and rotation matrix
    m_R[0] = m_pose[0];
    m_R[1] = m_pose[1];
    m_R[2] = m_pose[2];
    m_t    = m_pose[3];


    std::vector<glm::vec3> points;

    const float leftOffset = 0.5f;
    const float bottomOffset = 0.7f;
    points.push_back(glm::vec3(-1.0f + leftOffset, -1.0f + leftOffset, 0.0)); //bottom-left
    points.push_back(glm::vec3(-1.0f + bottomOffset, -1.0f + leftOffset, 0.0)); //bottom-right
    points.push_back(glm::vec3(-1.0f + bottomOffset, -1.0f + bottomOffset, 0.0)); //top-right
    points.push_back(glm::vec3(-1.0f + leftOffset, -1.0f + bottomOffset, 0.0)); //top-left


    std::vector <GLfloat> glPoints;
    ViewerUtil::convertToGL(points,glPoints);

    std::vector<GLuint> indices = {0, 1, 2, 2, 3, 0};
    initializeBuffers(&glPoints, &indices);

}

void GUISquaresGizmo::render() const
{
    if(m_vao == 0) return;

    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, m_N, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}


struct Quaternion
{

    float x;
    float y;
    float z;
    float w;

    Quaternion(float _x, float _y, float _z, float _w)
    {

        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }


    static Quaternion conjugate(float _x, float _y, float _z, float _w)
    {
        Quaternion ret(-_x, -_y, -_z, _w);
        return ret;
    }


    Quaternion normalize(Quaternion &quaternion)
    {
        float mag = glm::sqrt(quaternion.w*quaternion.w + quaternion.x*quaternion.x + quaternion.y*quaternion.y + quaternion.z*quaternion.z);

        Quaternion normQuaternion(quaternion.x / mag, quaternion.y / mag, quaternion.z / mag, quaternion.w / mag);
        return normQuaternion;
    }
};

inline Quaternion operator*(const Quaternion& l, const Quaternion& r)
{
    const float w = (l.w * r.w) - (l.x * r.x) - (l.y * r.y) - (l.z * r.z);
    const float x = (l.x * r.w) + (l.w * r.x) + (l.y * r.z) - (l.z * r.y);
    const float y = (l.y * r.w) + (l.w * r.y) + (l.z * r.x) - (l.x * r.z);
    const float z = (l.z * r.w) + (l.w * r.z) + (l.x * r.y) - (l.y * r.x);

    Quaternion ret(x, y, z, w);

    return ret;
}

inline Quaternion operator*(const Quaternion& q, const glm::vec3& v)
{
    const float w = -(q.x * v.x) - (q.y * v.y) - (q.z * v.z);
    const float x = (q.w * v.x) + (q.y * v.z) - (q.z * v.y);
    const float y = (q.w * v.y) + (q.z * v.x) - (q.x * v.z);
    const float z = (q.w * v.z) + (q.x * v.y) - (q.y * v.x);

    Quaternion ret(x, y, z, w);

    return ret;
}

inline static glm::vec3 rotateAngleAxis(glm::vec3 &vector, float angle, glm::vec3 &axis)
{
    float halfSinAngle = glm::sin((angle / 2.0f));

    float x = axis[0] * halfSinAngle;
    float y = axis[1] * halfSinAngle;
    float z = axis[2] * halfSinAngle;
    float w = glm::cos(angle / 2.0f);

    Quaternion RotationQ(x, y, z, w);
    Quaternion ConjugateQ = Quaternion::conjugate(x, y, z, w);

    Quaternion result = RotationQ * vector*ConjugateQ;

    glm::vec3 ret(result.x, result.y, result.z);

    return ret;
}

void ModelManager::updateSettings()
{
    //Next();
}

void ModelManager::Next()
{

}

void ModelManager::Update(float t)
{

}

void ModelManager::loadMesh(const std::string &path, AAssetManager *assetManager)
{
    if(m_mesh == nullptr)
        m_mesh = new ObjMesh();
    else
        m_mesh->clearMesh();

    m_mesh = ObjMesh::load(path.c_str(),assetManager);
    auto meshPts = m_mesh->getMeshData()->points;
    m_pts.clear();
    m_pts.reserve(meshPts.size());
    for(auto& pt : meshPts)
    {
        m_pts.emplace_back(pt.x, pt.y, pt.z, 1.0);
    }
}




void TriangleMesh::initBuffers(
        std::vector<GLuint> * indices,
        std::vector<GLfloat> * points,
        std::vector<GLfloat> * normals,
        std::vector<GLfloat> * texCoords,
        std::vector<GLfloat> * tangents,
        std::vector<GLfloat> * vertexColors
) {

    if( ! buffers.empty() ) deleteBuffers();

    nVerts = (GLuint)points->size();
    int vtxAttributeIndex = 0;


    // Must have data for indices, points, and normals
    if( indices == nullptr || points == nullptr || normals == nullptr )
    {
        return;
    }


    GLuint indexBuf = 0, posBuf = 0, normBuf = 0, tcBuf = 0, tangentBuf = 0, vertexColorsBuf = 0;
    glGenBuffers(1, &indexBuf);
    buffers.push_back(indexBuf);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuf);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices->size() * sizeof(GLuint), indices->data(), GL_STATIC_DRAW);

    glGenBuffers(1, &posBuf);
    buffers.push_back(posBuf);
    glBindBuffer(GL_ARRAY_BUFFER, posBuf);
    glBufferData(GL_ARRAY_BUFFER, points->size() * sizeof(GLfloat), points->data(), GL_STATIC_DRAW);

    glGenBuffers(1, &normBuf);
    buffers.push_back(normBuf);
    glBindBuffer(GL_ARRAY_BUFFER, normBuf);
    glBufferData(GL_ARRAY_BUFFER, normals->size() * sizeof(GLfloat), normals->data(), GL_STATIC_DRAW);


    if (texCoords != nullptr)
    {
        glGenBuffers(1, &tcBuf);
        buffers.push_back(tcBuf);
        glBindBuffer(GL_ARRAY_BUFFER, tcBuf);
        glBufferData(GL_ARRAY_BUFFER, texCoords->size() * sizeof(GLfloat), texCoords->data(), GL_STATIC_DRAW);
    }

    if (tangents != nullptr)
    {
        glGenBuffers(1, &tangentBuf);
        buffers.push_back(tangentBuf);
        glBindBuffer(GL_ARRAY_BUFFER, tangentBuf);
        glBufferData(GL_ARRAY_BUFFER, tangents->size() * sizeof(GLfloat), tangents->data(), GL_STATIC_DRAW);
    }


    if (vertexColors != nullptr)
    {
        glGenBuffers(1, &vertexColorsBuf);
        buffers.push_back(vertexColorsBuf);
        glBindBuffer(GL_ARRAY_BUFFER, vertexColorsBuf);
        glBufferData(GL_ARRAY_BUFFER, vertexColors->size() * sizeof(GLfloat), vertexColors->data(), GL_STATIC_DRAW);
    }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuf);

    // Position
    glBindBuffer(GL_ARRAY_BUFFER, posBuf);
    glVertexAttribPointer(vtxAttributeIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(vtxAttributeIndex);  // Vertex position
    vtxAttributeIndex++;

    // Normal
    glBindBuffer(GL_ARRAY_BUFFER, normBuf);
    glVertexAttribPointer(vtxAttributeIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(vtxAttributeIndex);  // Normal
    vtxAttributeIndex++;
    // Tex coords
    if (texCoords != nullptr)
    {
        glBindBuffer(GL_ARRAY_BUFFER, tcBuf);
        glVertexAttribPointer(vtxAttributeIndex, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(vtxAttributeIndex);  // Tex coord
        vtxAttributeIndex++;
    }

    if (tangents != nullptr)
    {
        glBindBuffer(GL_ARRAY_BUFFER, tangentBuf);
        glVertexAttribPointer(vtxAttributeIndex, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(vtxAttributeIndex);  // Tangents
        vtxAttributeIndex++;
    }

    // Tex coords
    if (vertexColors != nullptr)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vertexColorsBuf);
        glVertexAttribPointer(vtxAttributeIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(vtxAttributeIndex);  // Tex coord
    }

    glBindVertexArray(0);
}

void TriangleMesh::render(int renderType) const {
    if(vao == 0) return;

    glBindVertexArray(vao);
    if(renderType == 0)
        glDrawArrays(GL_POINTS, 0, nVerts);
    else if(renderType == 1)
        glDrawElements(GL_TRIANGLES, nVerts, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

TriangleMesh::~TriangleMesh() {
    deleteBuffers();
}

void TriangleMesh::deleteBuffers() {
    if( buffers.size() > 0 ) {
        glDeleteBuffers( (GLsizei)buffers.size(), buffers.data() );
        buffers.clear();
    }

    if( vao != 0 ) {
        glDeleteVertexArrays(1, &vao);
        vao = 0;
    }
}


ObjMesh::ObjMesh() : drawAdj(false)
{ }

void ObjMesh::render(int renderType) const {
    if( drawAdj ) {
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES_ADJACENCY, nVerts, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    } else {
        TriangleMesh::render(renderType);
    }
}

ObjMesh* ObjMesh::load( const char * fileName, bool center, bool genTangents) {

    ObjMesh* mesh = new ObjMesh();

    //ObjMeshData meshData;
    mesh->m_meshData = std::make_shared<ObjMeshData>();
    mesh->m_meshData->load(fileName);
    // Generate normals
    mesh->m_meshData->generateNormalsIfNeeded();

    // Generate tangents?
    if( genTangents ) mesh->m_meshData->generateTangents();

    // Convert to GL format
    GlMeshData glMesh;
    mesh->m_meshData->toGlMesh(glMesh);


    // Load into VAO
    mesh->initBuffers(
            & (glMesh.faces), & glMesh.points, & glMesh.normals,
            glMesh.texCoords.empty() ? nullptr : (& glMesh.texCoords),
            glMesh.tangents.empty() ? nullptr : (& glMesh.tangents),
            glMesh.vertexColors.empty() ? nullptr : (& glMesh.vertexColors)
    );

    std::cout << "Loaded mesh from: " << fileName
              << " vertices = " << (glMesh.points.size() / 3)
              << " triangles = " << (glMesh.faces.size() / 3)
              <<  std::endl;

    return mesh;
}

ObjMesh* ObjMesh::load( const char * fileName, AAssetManager *assetManager )
{
    ObjMesh* mesh = new ObjMesh();

    //ObjMeshData meshData;
    mesh->m_meshData = std::make_shared<ObjMeshData>();
    mesh->m_meshData->load(fileName, assetManager);
    // Generate normals
    mesh->m_meshData->generateNormalsIfNeeded();

    // Generate tangents?
    //if( genTangents ) mesh->m_meshData->generateTangents();

    // Convert to GL format
    GlMeshData glMesh;
    mesh->m_meshData->toGlMesh(glMesh);


    // Load into VAO
    mesh->initBuffers(
            & (glMesh.faces), & glMesh.points, & glMesh.normals,
            glMesh.texCoords.empty() ? nullptr : (& glMesh.texCoords),
            glMesh.tangents.empty() ? nullptr : (& glMesh.tangents),
            glMesh.vertexColors.empty() ? nullptr : (& glMesh.vertexColors)
    );

    std::cout << "Loaded mesh from: " << fileName
              << " vertices = " << (glMesh.points.size() / 3)
              << " triangles = " << (glMesh.faces.size() / 3)
              <<  std::endl;

    return mesh;
}

ObjMesh* ObjMesh::loadWithAdjacency( const char * fileName, bool center ) {

    ObjMesh* mesh = new ObjMesh();

    ObjMeshData meshData;
    meshData.load(fileName);

    // Generate normals
    meshData.generateNormalsIfNeeded();

    // Convert to GL format
    GlMeshData glMesh;
    meshData.toGlMesh(glMesh);


    mesh->drawAdj = true;
    glMesh.convertFacesToAdjancencyFormat();

    // Load into VAO
    mesh->initBuffers(
            & (glMesh.faces), & glMesh.points, & glMesh.normals,
            glMesh.texCoords.empty() ? nullptr : (& glMesh.texCoords),
            glMesh.tangents.empty() ? nullptr : (& glMesh.tangents)
    );

    std::cout << "Loaded mesh from: " << fileName
              << " vertices = " << (glMesh.points.size() / 3)
              << " triangles = " << (glMesh.faces.size() / 3) << std::endl;

    return mesh;
}

void ObjMeshData::load(const char * fileName) {
    std::ifstream objStream(fileName, std::ios::in);

    if (!objStream) {
        std::cerr << "Unable to open OBJ file: " << fileName << std::endl;
        exit(1);
    }

    std::cout << "loading data mesh... " << std::endl;
    std::string line, token;
    std::getline(objStream, line);
    while (!objStream.eof()) {
        // Remove comment if it exists
        size_t pos = line.find_first_of("#");
        if (pos != std::string::npos) {
            line = line.substr(0, pos);
        }
        trimString(line);

        if (line.length() > 0) {
            std::istringstream lineStream(line);

            lineStream >> token;

            if (token == "v") {
                float x, y, z;
                lineStream >> x >> y >> z;
                glm::vec3 p(x, y, z);
                points.push_back(p);
                glm::vec3 c((float)((std::rand() % 256) / 255.0f), (float)((std::rand() % 256) / 255.0f), (float)((std::rand() % 256) / 255.0f));
                vertexColors.push_back(c);
            }
            else if (token == "vt") {
                // Process texture coordinate
                float s, t;
                lineStream >> s >> t;
                texCoords.push_back(glm::vec2(s, t));
            }
            else if (token == "vn") {
                float x, y, z;
                lineStream >> x >> y >> z;
                normals.push_back(glm::vec3(x, y, z));
            }
            else if (token == "f") {
                std::vector<std::string> parts;
                while (lineStream.good()) {
                    std::string s;
                    lineStream >> s;
                    parts.push_back(s);
                }

                // Triangulate as a triangle fan
                if (parts.size() > 2) {
                    ObjVertex firstVert(parts[0], this);
                    for (int i = 2; i < parts.size(); i++) {
                        faces.push_back(firstVert);
                        faces.push_back(ObjVertex(parts[i - 1], this));
                        faces.push_back(ObjVertex(parts[i], this));
                    }
                }
            }
        }
        getline(objStream, line);
    }
    objStream.close();

}

void ObjMeshData::load(const char* fileName,AAssetManager *assetManager)
{
    std::string meshFile = ReadTextFile(fileName, assetManager);

    std::istringstream objStream(meshFile);

    if (!objStream) {
        std::cerr << "Unable to open OBJ file: " << fileName << std::endl;
        exit(1);
    }

    std::string line, token;
    std::getline(objStream, line);
    while (!objStream.eof()) {
        // Remove comment if it exists
        size_t pos = line.find_first_of("#");
        if (pos != std::string::npos) {
            line = line.substr(0, pos);
        }
        trimString(line);

        if (line.length() > 0) {
            std::istringstream lineStream(line);

            lineStream >> token;

            if (token == "v") {
                float x, y, z;
                lineStream >> x >> y >> z;
                glm::vec3 p(x, y, z);
                points.push_back(p);
                glm::vec3 c((float)((std::rand() % 256) / 255.0f), (float)((std::rand() % 256) / 255.0f), (float)((std::rand() % 256) / 255.0f));
                vertexColors.push_back(c);
            }
            else if (token == "vt") {
                // Process texture coordinate
                float s, t;
                lineStream >> s >> t;
                texCoords.push_back(glm::vec2(s, t));
            }
            else if (token == "vn") {
                float x, y, z;
                lineStream >> x >> y >> z;
                normals.push_back(glm::vec3(x, y, z));
            }
            else if (token == "f") {
                std::vector<std::string> parts;
                while (lineStream.good()) {
                    std::string s;
                    lineStream >> s;
                    parts.push_back(s);
                }

                // Triangulate as a triangle fan
                if (parts.size() > 2) {
                    ObjVertex firstVert(parts[0], this);
                    for (int i = 2; i < parts.size(); i++) {
                        faces.push_back(firstVert);
                        faces.push_back(ObjVertex(parts[i - 1], this));
                        faces.push_back(ObjVertex(parts[i], this));
                    }
                }
            }
        }
        getline(objStream, line);
    }

    size_t ptsSize = points.size();
    size_t vtxColorsSize = vertexColors.size();
    size_t texCoordsSisze = texCoords.size();
    size_t normalsSize = normals.size();

    std::cout << "mesh data loaded:" << " pts size: "  << std::to_string(ptsSize)
              << " pts size: "        << std::to_string(ptsSize)
              << " vtx color size: "  << std::to_string(vtxColorsSize)
              << " tex Color size: "  << std::to_string(texCoordsSisze)
              << " normals size: "    << std::to_string(normalsSize) << std::endl;
}


ObjMeshData::ObjVertex::ObjVertex(std::string &vertString, ObjMeshData * mesh) : pIdx(-1), nIdx(-1), tcIdx(-1) {
    size_t slash1, slash2;
    slash1 = vertString.find("/");
    pIdx = std::stoi(vertString.substr(0, slash1));
    if (pIdx < 0) pIdx += mesh->points.size();
    else pIdx--;
    if (slash1 != std::string::npos) {
        slash2 = vertString.find("/", slash1 + 1);
        if (slash2 > slash1 + 1) {
            tcIdx = std::stoi(vertString.substr(slash1 + 1, slash2 - slash1 - 1));
            if (tcIdx < 0) tcIdx += mesh->texCoords.size();
            else tcIdx--;
        }
        nIdx = std::stoi(vertString.substr(slash2 + 1));
        if (nIdx < 0) nIdx += mesh->normals.size();
        else nIdx--;
    }
}

void ObjMeshData::generateNormalsIfNeeded() {
    if( normals.size() != 0 ) return;

    normals.resize(points.size());

    for( GLuint i = 0; i < faces.size(); i += 3) {
        const glm::vec3 & p1 = points[faces[i].pIdx];
        const glm::vec3 & p2 = points[faces[i+1].pIdx];
        const glm::vec3 & p3 = points[faces[i+2].pIdx];

        glm::vec3 a = p2 - p1;
        glm::vec3 b = p3 - p1;
        glm::vec3 n = glm::normalize(glm::cross(a,b));

        normals[faces[i].pIdx] += n;
        normals[faces[i+1].pIdx] += n;
        normals[faces[i+2].pIdx] += n;

        // Set the normal index to be the same as the point index
        faces[i].nIdx = faces[i].pIdx;
        faces[i+1].nIdx = faces[i+1].pIdx;
        faces[i+2].nIdx = faces[i+2].pIdx;
    }

    for( GLuint i = 0; i < normals.size(); i++ ) {
        normals[i] = glm::normalize(normals[i]);
    }
}

void ObjMeshData::generateTangents() {
    std::vector<glm::vec3> tan1Accum(points.size());
    std::vector<glm::vec3> tan2Accum(points.size());
    tangents.resize(points.size());

    // Compute the tangent std::vector
    for( GLuint i = 0; i < faces.size(); i += 3 )
    {
        const glm::vec3 &p1 = points[faces[i].pIdx];
        const glm::vec3 &p2 = points[faces[i+1].pIdx];
        const glm::vec3 &p3 = points[faces[i+2].pIdx];

        const glm::vec2 &tc1 = texCoords[faces[i].tcIdx];
        const glm::vec2 &tc2 = texCoords[faces[i+1].tcIdx];
        const glm::vec2 &tc3 = texCoords[faces[i+2].tcIdx];

        glm::vec3 q1 = p2 - p1;
        glm::vec3 q2 = p3 - p1;
        float s1 = tc2.x - tc1.x, s2 = tc3.x - tc1.x;
        float t1 = tc2.y - tc1.y, t2 = tc3.y - tc1.y;
        float r = 1.0f / (s1 * t2 - s2 * t1);
        glm::vec3 tan1( (t2*q1.x - t1*q2.x) * r,
                        (t2*q1.y - t1*q2.y) * r,
                        (t2*q1.z - t1*q2.z) * r);
        glm::vec3 tan2( (s1*q2.x - s2*q1.x) * r,
                        (s1*q2.y - s2*q1.y) * r,
                        (s1*q2.z - s2*q1.z) * r);
        tan1Accum[faces[i].pIdx] += tan1;
        tan1Accum[faces[i+1].pIdx] += tan1;
        tan1Accum[faces[i+2].pIdx] += tan1;
        tan2Accum[faces[i].pIdx] += tan2;
        tan2Accum[faces[i+1].pIdx] += tan2;
        tan2Accum[faces[i+2].pIdx] += tan2;
    }

    for( GLuint i = 0; i < points.size(); ++i )
    {
        const glm::vec3 &n = normals[i];
        glm::vec3 &t1 = tan1Accum[i];
        glm::vec3 &t2 = tan2Accum[i];

        // Gram-Schmidt orthogonalize
        tangents[i] = glm::vec4(glm::normalize( t1 - (glm::dot(n,t1) * n) ), 0.0f);
        // Store handedness in w
        tangents[i].w = (glm::dot( glm::cross(n,t1), t2 ) < 0.0f) ? -1.0f : 1.0f;
    }
}

void ObjMeshData::toGlMesh(GlMeshData & data) {
    data.clear();

    std::map<std::string, GLuint> vertexMap;
    //try to load only components instead of faces
    if(!faces.size())
    {
        if(points.size())
        {
            for(auto& pt : points)
            {
                data.points.push_back(pt.x);
                data.points.push_back(pt.y);
                data.points.push_back(pt.z);
            }
        }
    }
    else
    {
        for (auto &vert: faces)
        {
            auto vertStr = vert.str();
            auto it = vertexMap.find(vertStr);
            if (it == vertexMap.end())
            {
                auto vIdx = data.points.size() / 3;

                auto &pt = points[vert.pIdx];
                data.points.push_back(pt.x);
                data.points.push_back(pt.y);
                data.points.push_back(pt.z);

                auto &n = normals[vert.nIdx];
                data.normals.push_back(n.x);
                data.normals.push_back(n.y);
                data.normals.push_back(n.z);

                auto &c = vertexColors[vert.pIdx];
                data.vertexColors.push_back(c.x);
                data.vertexColors.push_back(c.y);
                data.vertexColors.push_back(c.z);

                if (!texCoords.empty())
                {
                    auto &tc = texCoords[vert.tcIdx];
                    data.texCoords.push_back(tc.x);
                    data.texCoords.push_back(tc.y);
                }

                if (!tangents.empty())
                {
                    // We use the point index for tangents
                    auto &tang = tangents[vert.pIdx];
                    data.tangents.push_back(tang.x);
                    data.tangents.push_back(tang.y);
                    data.tangents.push_back(tang.z);
                    data.tangents.push_back(tang.w);
                }

                data.faces.push_back((GLuint) vIdx);
                vertexMap[vertStr] = (GLuint) vIdx;
            } else
            {
                data.faces.push_back(it->second);
            }
        }
    }
}

void GlMeshData::convertFacesToAdjancencyFormat()
{
    // Elements with adjacency info
    std::vector<GLuint> elAdj(faces.size() * 2);

    // Copy and make room for adjacency info
    for( GLuint i = 0; i < faces.size(); i+=3)
    {
        elAdj[i*2 + 0] = faces[i];
        elAdj[i*2 + 1] = std::numeric_limits<GLuint>::max();
        elAdj[i*2 + 2] = faces[i+1];
        elAdj[i*2 + 3] = std::numeric_limits<GLuint>::max();
        elAdj[i*2 + 4] = faces[i+2];
        elAdj[i*2 + 5] = std::numeric_limits<GLuint>::max();
    }

    // Find matching edges
    for( GLuint i = 0; i < elAdj.size(); i+=6)
    {
        // A triangle
        GLuint a1 = elAdj[i];
        GLuint b1 = elAdj[i+2];
        GLuint c1 = elAdj[i+4];

        // Scan subsequent triangles
        for(GLuint j = i+6; j < elAdj.size(); j+=6)
        {
            GLuint a2 = elAdj[j];
            GLuint b2 = elAdj[j+2];
            GLuint c2 = elAdj[j+4];

            // Edge 1 == Edge 1
            if( (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2) )
            {
                elAdj[i+1] = c2;
                elAdj[j+1] = c1;
            }
            // Edge 1 == Edge 2
            if( (a1 == b2 && b1 == c2) || (a1 == c2 && b1 == b2) )
            {
                elAdj[i+1] = a2;
                elAdj[j+3] = c1;
            }
            // Edge 1 == Edge 3
            if ( (a1 == c2 && b1 == a2) || (a1 == a2 && b1 == c2) )
            {
                elAdj[i+1] = b2;
                elAdj[j+5] = c1;
            }
            // Edge 2 == Edge 1
            if( (b1 == a2 && c1 == b2) || (b1 == b2 && c1 == a2) )
            {
                elAdj[i+3] = c2;
                elAdj[j+1] = a1;
            }
            // Edge 2 == Edge 2
            if( (b1 == b2 && c1 == c2) || (b1 == c2 && c1 == b2) )
            {
                elAdj[i+3] = a2;
                elAdj[j+3] = a1;
            }
            // Edge 2 == Edge 3
            if( (b1 == c2 && c1 == a2) || (b1 == a2 && c1 == c2) )
            {
                elAdj[i+3] = b2;
                elAdj[j+5] = a1;
            }
            // Edge 3 == Edge 1
            if( (c1 == a2 && a1 == b2) || (c1 == b2 && a1 == a2) )
            {
                elAdj[i+5] = c2;
                elAdj[j+1] = b1;
            }
            // Edge 3 == Edge 2
            if( (c1 == b2 && a1 == c2) || (c1 == c2 && a1 == b2) )
            {
                elAdj[i+5] = a2;
                elAdj[j+3] = b1;
            }
            // Edge 3 == Edge 3
            if( (c1 == c2 && a1 == a2) || (c1 == a2 && a1 == c2) )
            {
                elAdj[i+5] = b2;
                elAdj[j+5] = b1;
            }
        }
    }

    // Look for any outside edges
    for( GLuint i = 0; i < elAdj.size(); i+=6)
    {
        if( elAdj[i+1] == std::numeric_limits<GLuint>::max() ) elAdj[i+1] = elAdj[i+4];
        if( elAdj[i+3] == std::numeric_limits<GLuint>::max() ) elAdj[i+3] = elAdj[i];
        if( elAdj[i+5] == std::numeric_limits<GLuint>::max() ) elAdj[i+5] = elAdj[i+2];
    }

    // Copy all data back into el
    faces = elAdj;
}

// Initialization function
void VtxFeature::initializeBuffers(const std::vector<glm::vec4>& points)
{
    std::cout << "initializing vtx feature buffers!! " << std::to_string(points.size()) <<  std::endl;
    if (points.empty()) return;

    GLuint posBuffer = 0, indexBuffer = 0, outputBuffer = 0, counterBuffer = 0;
    int vtxPosBinding = 0;
    int indexBinding = 1;
    int outputBinding = 2;
    int counterBinding = 3;
    m_N = points.size();

    //fill indices
    std::vector<unsigned int> pointsIndexes = std::vector<unsigned int>(m_N, 0);
    std::iota(pointsIndexes.begin(), pointsIndexes.end(), 0);

    // Create and bind the vertex position buffer
    glGenBuffers(1, &posBuffer);
    m_buffers.push_back(posBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, posBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4)*m_N, points.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, vtxPosBinding, posBuffer);

    // Create and bind the index buffer
    glGenBuffers(1, &indexBuffer);
    m_buffers.push_back(indexBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(unsigned int)*m_N, pointsIndexes.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, indexBinding, indexBuffer);

    // Create and bind the output buffer
    glGenBuffers(1, &outputBuffer);
    m_buffers.push_back(outputBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::ivec4) * m_N, nullptr, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, outputBinding, outputBuffer);

    // Unbind the buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void VtxFeature::readOutputBuffer(std::vector<cv::Point3i>& vtxFeatures)
{

    // Assuming m_buffers[2] is the output buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_buffers[2]);
    size_t bufferSize = sizeof(glm::ivec4) * m_N;
    void* ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bufferSize, GL_MAP_READ_BIT);

    if (ptr)
    {
        vtxFeatures.clear();
        glm::ivec4* debugData = static_cast<glm::ivec4*>(ptr);
        for (size_t i = 0; i < m_N; ++i)
        {
            if(   debugData[i].x == 0 && debugData[i].y == 0 && debugData[i].z == 0)
                continue;
            vtxFeatures.emplace_back(debugData[i].x, debugData[i].y, debugData[i].z);
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    } else {
        std::cerr << "Failed to map buffer" << std::endl;
    }


    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // Unbind
    glUseProgram(0);

}

void VtxFeature::initialize(const std::vector<glm::vec4>& points)
{
    deleteBuffers();
    initializeBuffers(points);
}

void VtxFeature::deleteBuffers()
{
    if( m_buffers.size() > 0 )
    {
        glDeleteBuffers( (GLsizei)m_buffers.size(), m_buffers.data() );
        m_buffers.clear();
    }

    if( m_vao != 0 )
    {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
}

