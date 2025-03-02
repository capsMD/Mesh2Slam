

#ifndef VIEWER_H
#define VIEWER_H

//system
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <memory>
#include <thread>
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <map>

//replaces EGL + SDL (used by Opengl ES)
#include <gfxwrapper_opengl.h>

//OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

//maths library
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>


//needed to read files in Android
#include <HelperFunctions.h>

//own
#include "slamUtils.h"
#include "slamParams.h"
#include "map.h"


class TriangleMesh
{
public:

    virtual void initBuffers(
            std::vector<GLuint> * indices,
            std::vector<GLfloat> * points,
            std::vector<GLfloat> * normals,
            std::vector<GLfloat> * texCoords = nullptr,
            std::vector<GLfloat> * tangents = nullptr,
            std::vector<GLfloat>* vertexColors = nullptr

    );
    void cleanup(){deleteBuffers();}
protected:

    GLuint nVerts;     // Number of vertices
    GLuint vao;        // The Vertex Array Object

    // Vertex buffers
    std::vector<GLuint> buffers;


    virtual void deleteBuffers();

public:
    virtual ~TriangleMesh();
    virtual void render(int renderType) const;
    GLuint getVao() const { return vao; }
    GLuint getElementBuffer() { return buffers[0]; }
    GLuint getPositionBuffer() { return buffers[1]; }
    GLuint getNormalBuffer() { return buffers[2]; }
    GLuint getTcBuffer() { if( buffers.size() > 3) return buffers[3]; else return 0; }
    GLuint getNumVerts() { return nVerts; }

};

// Helper classes used for loading
class GlMeshData {
public:
    std::vector <GLfloat> points;
    std::vector <GLfloat> normals;
    std::vector <GLfloat> texCoords;
    std::vector <GLuint> faces;
    std::vector <GLfloat> tangents;
    std::vector <GLfloat> vertexColors;

    void clear() {
        points.clear();
        normals.clear();
        texCoords.clear();
        faces.clear();
        tangents.clear();
        vertexColors.clear();
    }
    void convertFacesToAdjancencyFormat();
};

class ObjMeshData
{
public:
    class ObjVertex {
    public:
        int pIdx;
        int nIdx;
        int tcIdx;

        ObjVertex() {
            pIdx = -1;
            nIdx = -1;
            tcIdx = -1;
        }

        ObjVertex(std::string& vertString, ObjMeshData* mesh);
        std::string str() {
            return std::to_string(pIdx) + "/" + std::to_string(tcIdx) + "/" + std::to_string(nIdx);
        }
    };

    std::vector <glm::vec3> points;
    std::vector <glm::vec3> normals;
    std::vector <glm::vec2> texCoords;
    std::vector <ObjVertex> faces;
    std::vector <glm::vec4> tangents;
    std::vector <glm::vec3> vertexColors;

    ObjMeshData() { }

    void generateNormalsIfNeeded();
    void generateTangents();
    void load(const char* fileName);
    void load(const char* fileName,AAssetManager *assetManager );
    void toGlMesh(GlMeshData& data);
};

class ObjMesh : public TriangleMesh {

public:
    ObjMesh();
    ~ObjMesh(){ std::cout << "deleting mesh" << std::endl; }
    static ObjMesh* load(const char * fileName, bool center = false, bool genTangents = false);
    static ObjMesh* load(const char * fileName, AAssetManager *assetManager);
    static ObjMesh* loadWithAdjacency(const char * fileName, bool center = false);
    const std::shared_ptr<ObjMeshData>& getMeshData() const { return m_meshData; }
    void render(int renderType) const override;
    void clearMesh(){deleteBuffers();}

private:
    bool drawAdj;
    std::shared_ptr<ObjMeshData> m_meshData;
};

class ModelManager
{
public:
    ObjMesh* getModelMesh(void) { return m_mesh; }
    const std::vector<glm::vec4>& getModelPts(void) const {return m_pts;}
    void loadMesh(const std::string &path, AAssetManager *assetManager);
    void updateSettings();
    void Update(float t);
    void Next();
private:

    std::vector<glm::vec4> m_pts;
    ObjMesh* m_mesh{nullptr};

    glm::vec3 m_startPosition{0.0f};
    glm::vec3 m_endPosition{0.0f};
    glm::vec3 m_startOrientation{0.0f};
    glm::vec3 m_endOrientation{0.0f};
};

enum class EventTypes
{
    Empty =0,
    WindowClose,
    WindowResize,
    WindowFocus,
    WindowLostFocus,
    WindowMoved,
    AppTick,
    AppUpdate,
    AppRender,
    KeyPressed,
    KeyReleased,
    MousePressed,
    MouseReleased,
    MouseMoved,
    MouseScrolled,
    GUIUpdate,
    MapUpdate

};

inline static glm::vec3 rotateAngleAxis(glm::vec3 &vector, float angle, glm::vec3 &axis);

class ViewerUtil final {
public:
    static void convertToGL(const std::vector<glm::vec3>& points, std::vector<GLfloat>& glPoints);
inline static glm::vec3 convertTranslation(const glm::vec3 &t, const glm::vec3 &v)
{
    /*
    Do: M  * t

            | 0 -1 0 |   | X |	  |-Y |
            | 0  0 1 | * | Y | =  | Z |
            | -1 0 0 |	 | Z |	  |-X |
    */
    glm::mat3 M(0.0);
    M[0][abs(v[0])-1] = (v[0] < 0) ? -1 : 1;
    M[1][abs(v[1])-1] = (v[1] < 0) ? -1 : 1;
    M[2][abs(v[2])-1] = (v[2] < 0) ? -1 : 1;
    return M * t;
}

inline static glm::mat3 convertOrientation(const glm::mat3 &R, const glm::vec3 &v)
{
    /* Example:
        From: x1,y1,z1          to -z2,-y2,z2        use( -3,-1,1)
                                     _ _ _ _ z2
      y1|   /z1                    / |
        |  /                      /  |
        | /                      /   |
        |/_ _ _ _ _x1        x2 /    | y2

     Do: M^T * R:
    | 0   0  -1 |   | Xx  Yx  Zx |	  | -Zx  -Yx  Xx |
    | 0  -1   0 | * | Xy  Yy  Zy | =  | -Zy  -Yy  Xy |
    | 1   0   0 |   | Xz  Yz  Zz |	  | -Zz  -Yz  Xz |
    */

    glm::mat3 M(0.0);
    M[0][abs(v[0])-1] = (v[0] < 0) ? -1 : 1;
    M[1][abs(v[1])-1] = (v[1] < 0) ? -1 : 1;
    M[2][abs(v[2])-1] = (v[2] < 0) ? -1 : 1;
    return R*M;
}

private:
    // non-instantiable, non-copyable
    ViewerUtil() = delete;
    ViewerUtil& operator=(const ViewerUtil&) = delete;
    ViewerUtil(const ViewerUtil&) = delete;
};

class Shader
{
public:
    Shader(){}
    ~Shader(){if(m_shaderProgram){detachAndDeleteShaders();
            glDeleteProgram(m_shaderProgram);}}
//not allowed copies
    Shader(const Shader&) = delete;
    Shader& operator = (const Shader&) = delete;

    void use() {if(m_shaderProgram > 0 && m_isLinked) glUseProgram(m_shaderProgram);}
    bool link();
    bool compile(GLenum shaderType, const std::string& shaderFile);
    bool compile(GLenum shaderType, const std::string &shaderSrcFile, AAssetManager *assetManager);
    int  getHandle() const {return m_shaderProgram;}
    bool setHandle(GLuint handle);
    bool isLinked() const {return m_isLinked;}
    void setUniform(const char *name, const glm::mat4 &m);
    void setUniform(const char *name, const glm::vec2 &v);
    void setUniform(const char *name, const glm::vec3 &v);
    void setUniform(const char *name, float val);
    void cleanup();
private:
    int getUniformLocation(const char* name);
    void detachAndDeleteShaders();
    static std::string readFile(const std::string& path);
    void findUniformLocations();
    void findAttributeLocations();
private:
    GLuint m_shaderProgram;
    std::vector<GLuint> m_compiledShaders;
    std::map<std::string, GLuint> m_uniformLocations;
    std::map<std::string, GLuint> m_attributeLocations;
    bool m_isLinked{false};
};

class GLPrimitive
{
public:
    GLPrimitive(){}
    GLPrimitive(const glm::mat4& pose) : m_pose(pose){}
    virtual ~GLPrimitive() {}
    virtual void render() const;
    void setTransform(const glm::mat4& newPose) {}
    void setScale(const float v) {m_scale = v;}
    const float getScale(void) const {return m_scale;}
    const glm::mat4& getPose(void) const {return m_pose; }
    void setPose(const glm::mat4& pose) {m_pose = pose; m_t = glm::vec3(pose[3]);}
    const glm::vec3& getPosition(void) const {return m_t;}
    virtual void deleteBuffers();
    virtual void initializeEmptyBuffer();
    void clear(){deleteBuffers();}
    void loadPoints(const std::vector<glm::vec3> &points);
    GLuint getNumberOfElements() const {return m_N;}
protected:
    virtual void initializeBuffers(std::vector<GLfloat>* points);
    virtual void initializeBuffers(std::vector<GLfloat>* glPoints, std::vector<GLfloat>* glpointsColors);
    virtual void initializeBuffers(const std::vector<GLfloat> *points, const std::vector<GLuint> *indices);
    virtual void updateBuffer(const std::vector<GLfloat> *points);
    virtual void loadPoints(const std::vector<glm::vec3> &points,const std::vector<glm::vec3> &pointsColor);


protected:
    GLsizei m_N;
    GLuint m_vao;
    GLuint m_vbo;
    std::vector<GLuint> m_buffers;
    glm::mat4 m_pose;
    glm::mat3 m_R;
    glm::vec3 m_t;
private:
    virtual void initialize();
private:

    float m_scale{1.0f};
};

class AxisGizmo : public GLPrimitive
{
public:
    AxisGizmo(){initialize();};

private:
    void initialize();
};

class FrameGizmo : public GLPrimitive
{
public:
    FrameGizmo(){};
    FrameGizmo(const char frameType) : m_frameType(frameType){};
    FrameGizmo(const char frameType, const glm::mat4& pose, unsigned long int id) : GLPrimitive(pose), m_frameType(frameType), m_ID(id){};
    void renderAxisGizmos() const;
    void setParentNode(FrameGizmo* f);
    FrameGizmo* getParentNode(void) const {return m_parent;}
    void removeParentNode(void) {m_parent = nullptr;}
    void initialize() override;

private:
    char m_frameType{0};
    unsigned long int m_ID{0};
    AxisGizmo m_axisGizmo;
    FrameGizmo* m_parent{nullptr};
};

class PathGizmo : public GLPrimitive
{
public:
    PathGizmo() {}
    void updatePoints(const std::vector<glm::vec3>& points);
    void render() const override;
private:
    bool m_isInitialized{false};
};

class TrailGizmo : public GLPrimitive
{
public:
    TrailGizmo(){}
    ~TrailGizmo(){}
    void render() const override;
};

class PointCloud : public GLPrimitive
{
public:
    ~PointCloud(){deleteBuffers();}
    void loadPoints(const std::vector<glm::vec3>& points);
    void updatePoints(const std::vector<glm::vec3>& points);
    void render() const override;
    void setColor(const glm::vec3 color){ m_color=color;}
    const glm::vec3 getColor(void) const {return m_color;}
private:
    glm::vec3 m_color{1.0f, 1.0f, 1.0f};
};

class GUISquaresGizmo : public GLPrimitive
{
public:
    ~GUISquaresGizmo(){deleteBuffers();}
    void render() const override;
    void initialize() override;
    const glm::vec3 getColor() const{std::unique_lock<std::mutex> lock; return m_color;}
    void updateColor(const glm::vec3& color){m_color = color;}
private:
    glm::vec3 m_color{glm::vec3(0.0f,1.0f,0.0f)};
    std::mutex m_updateMutex;
};

class Camera
{
public:
    Camera(){*this= Camera(m_width,m_height,glm::vec3(0),glm::vec3 (1,0,0),glm::vec3(0,0,1));};
    Camera(int width, int height, const glm::vec3 pos, const glm::vec3 target, const glm::vec3 up);
    void onLook(int x, int y);
    void onMove(int key, int mode);
    void setIsFree(bool free) {m_isFree = free;}
    const glm::mat4& getViewMatrix() {setTransform(); return m_viewMatrix;}
    const glm::mat4& getProjectionMatrix() {setTransform(); return m_projectionMatrix;}
    const glm::mat4& getViewProjectionMatrix(){setTransform(); m_viewProjectionMatrix = m_viewMatrix*m_projectionMatrix; return m_viewProjectionMatrix;}
    void setMoveFactor(const float f) {m_moveFactor=f;}
private:
    void update();
    void updateMove();
    void updateAim();
    void setTransform();
    void setProjectionTransform();
    void setPosition(const glm::vec3& pos) {m_position = pos;};

private:
    glm::vec3 m_right;
    glm::vec3 m_up;
    glm::vec3 m_forward;

    float m_horAngle{0.0f };
    float m_verAngle{0.0f };
    float m_minMaxVerAngle{-85};

    int m_mouseX{0};
    int m_mouseY{0};

    float m_moveFactor{0.075f };
    float m_horFactor{1.75f };
    float m_verFactor{1.75f };

    bool m_isInitialized{false };
    bool m_isFree{false};

    glm::mat4 m_viewMatrix;
    glm::mat4 m_projectionMatrix;
    glm::mat4 m_viewProjectionMatrix;

    glm::mat4 m_transform;
    glm::vec3 m_position;
    glm::mat3 m_orientation;

    int m_width{640};
    int m_height{480};

    float m_fov{ 60.0f };
    float m_ar{ 1.0f };
    float m_near{ 1.0f };
    float m_far{ 100.0f };

};

class VtxFeature
{
public:
    ~VtxFeature(){deleteBuffers();}
    void readOutputBuffer(std::vector<cv::Point3i>& vtxFeatures);
    void initialize(const std::vector<glm::vec4>& points);
    std::vector<GLuint> getBuffers() {return m_buffers;}
    size_t getSize() const {return m_N;}
    void cleanup(){deleteBuffers();}
private:
    void initializeBuffers(const std::vector<glm::vec4>& points);
    void deleteBuffers();

private:
    size_t m_N{0};
    GLuint m_vao{0};
    GLuint m_vbo{0};
    std::vector<GLuint> m_buffers;
    glm::mat4 m_pose{glm::mat4(1.0f)};
};

class Viewer
{
public:

    //Viewer(SlamParams* slamParams);
    Viewer(SlamParams* slamParams, AAssetManager *assetManager);
    //Viewer(int width, int height, const std::string& title);
    ~Viewer();
    bool initialize();
    void update();
    void stopViewer();
    void exit();
	void setMap(Map* map) { m_map = map;}
    void setModelManager(ModelManager* modelManager);
    void setScale(const float scale){m_scaleFactor = scale;}
    void setMapUpdateFlag();
    bool checkMapUpdateFlag();
    void clearMapUpdateFlag();

    void setViewMatrix(const glm::mat4 view){m_vMatrix = view;}
    void setProjectionMatrix(const glm::mat4 proj){ m_pMatrix = proj;}
    void setModelMatrix(const glm::mat4 model){m_mMatrix = model;}
    void setSquareUpdateFlag(const char& state);

    void render();
    bool renderVtxFeatures(VertexFeatures& vtxFeatures);
	void initializeVtxFeature(const std::vector<glm::vec4>& modelVertexes);
    bool setActiveCamera(Camera* camera);
    

    void loadMesh(const std::string& mesh, AAssetManager *assetManager);
    bool getVtxFeatures(const float maxDepth,std::vector<cv::Point3i>& pts);
    glm::mat4 getP(void) const {return m_p;}
    void setMatrices();

    void debugGetComputeShaderTime() {if(m_computeShader_done){std::cout << "time: " << std::to_string(m_computeShader_dtAvg) << std::endl;}}
private:
    void initializeProjectionMatrix();
    void initializeShaders();
    void initializeBuffers();
    void initializeMapPoints();
    void initializeGizmos();
    void printVersions();
    //void updatePoints(std::vector<glm::vec3>& pts);
    void updateFrames(const std::vector<Frame*>& pFrames);
    void updateTFrames(const std::vector<TFrame*>& tFrames);
    void updatePaths();
    void shutdown();

private:

    //Viewer visuals
    int m_width{640};
    int m_height{480};
    std::string m_title{"No title"};
    float m_scaleFactor{1.0f};
    float m_fov{90.0f};
    float m_far{1000.0f};
    float m_near{1.0f};
	float m_featuresMaxDepth{10.0f};
    glm::mat4 m_p;
    glm::mat4 m_k;

    glm::vec3 m_frameColor{glm::vec3(0.0f)};
    glm::vec3 m_framePathColor{glm::vec3(0.0f)};
    glm::vec3 m_pointCloudColor{glm::vec3(0.0f)};
    glm::vec3 m_tFrameColor{glm::vec3(0.0f)};
    glm::vec3 m_meshColor{glm::vec3(0.0f)};
    
    EGLDisplay  m_eglDisplay;
    EGLConfig   m_eglConfig;
    EGLContext  m_eglContext;
    EGLSurface  m_eglSurface;

    Map* m_map{nullptr};
    ObjMesh* m_mesh{nullptr};

    std::map<unsigned long int,FrameGizmo* > m_frames;
    PointCloud* m_tframePoints{nullptr};
    PointCloud* m_pointCloud{nullptr};
    PathGizmo* m_framesPath{nullptr};
    GUISquaresGizmo* m_guiSquare{nullptr};

    std::map<std::string, Shader* > m_shaders;
    Camera* m_activeCamera{nullptr};
    ModelManager* m_modelManager{nullptr};
    VtxFeature* m_vtxFeature{nullptr};

    //TODO: Temporary to test point clouds for every frame
    std::vector<PointCloud*> mp_pointClouds;

    GLuint m_renderFBO, m_depthFBO;

    glm::mat4 m_mMatrix{glm::mat4(1.0f)};
    glm::mat4 m_vMatrix{glm::mat4(1.0f)};
    glm::mat4 m_pMatrix{glm::mat4(1.0f)};
    glm::mat4 m_mvpMatrix{glm::mat4(1.0f)};

    SlamParams* m_slamParams{ NULL };
    std::atomic<bool> m_mapUpdateFlag{false};
	std::atomic<bool> m_vtxFeaturesInitialized{false};
    std::mutex m_mutexUpdate;
	std::condition_variable m_cv;
    std::mutex m_viewerMutex;
    bool m_forceOriginStart{true};
    bool m_originCaptured{false};
    std::atomic<bool> m_stop{false};
    AAssetManager *m_assetManager;

    //for debugging
    double m_computeShader_dtAvg{0.0};
    double m_computeShader_total{0.0};
    size_t m_computeShader_nSamples{0};
    std::atomic<bool> m_computeShader_done{false};
};



#endif //VIEWER_H
