//
// Created by caps on 14.10.23.
//

#include "optimizer.h"
#include <random>


void Optimizer::globalBundleAdjustment(std::shared_ptr< Map> pMap, const int iterations)
{
    std::vector<Frame*> vpFrames = pMap->getFrames();
    std::vector<MapPoint*> vpMapPoints = pMap->getMapPoints();
    bundleAdjustN(vpFrames, vpMapPoints,  true,iterations);
}

int Optimizer::pnpOnFrame(Frame &frame, const int iterations)
{
    bool ok = false;
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = frame.getCamPrjParams().fx;
    K.at<float>(1,1) = frame.getCamPrjParams().fy;
    K.at<float>(0,2) = frame.getCamPrjParams().cx;
    K.at<float>(1,2) = frame.getCamPrjParams().cy;

    const float fx = frame.getCamPrjParams().fx;
    const float fy = frame.getCamPrjParams().fy;
    const float cx = frame.getCamPrjParams().cx;
    const float cy = frame.getCamPrjParams().cy;

    //optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();

    std::unique_ptr<g2o::BlockSolver_6_3> solver_ptr (new g2o::BlockSolver_6_3(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    optimizer.setAlgorithm(solver);

    int nBad = 0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(toSE3Quat(frame.getPosec()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);
    // Set MapPoint vertices
    const int N = frame.getN();

    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    std::vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    const float deltaMono = sqrt(5.991);

    std::vector<MapPoint*> mapPoints = frame.getMapPoints();

    for(int i=0; i<N; i++)
    {
        MapPoint* mapPoint = mapPoints[i];
        if(mapPoint != nullptr)
        {
                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = frame.getKeyPoints()[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                e->setInformation(Eigen::Matrix2d::Identity()*1.0f);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = fx;
                e->fy = fy;
                e->cx = cx;
                e->cy = cy;
                cv::Mat Xw = mapPoint->getPosition();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
        }
    }


    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const int its[4]={10,10,10,10};

    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(toSE3Quat(frame.getPosec()));
        optimizer.initializeOptimization(0);
        optimizer.optimize(iterations);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            const float chi2 = e->chi2();
            //std::cout << chi2 << std::endl;
            if(chi2>chi2Mono[it])
            {
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }
        if(optimizer.edges().size()<10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = toCVMat(SE3quat_recov);
    frame.setPose(pose);

    MapPoint* mapPoint;
    float totalError = 0.0f;
    nBad = 0;
    for (size_t i = 0; i < N; i++)
    {
        mapPoint = mapPoints[i];

        if((mapPoint == nullptr) || (mapPoint->isBad()))
            continue;

        const cv::Mat p3Dw = mapPoint->getPosition();
        const cv::Mat p3Dc = frame.getRc() * p3Dw + frame.getTc();

        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float invzc = 1.0f / p3Dc.at<float>(2);

        const float u = fx * x * invzc + cx;
        const float v = fy * y * invzc + cy;

        const float error = (u - frame.getKeyPoints()[i].pt.x) * (u - frame.getKeyPoints()[i].pt.x) + (v - frame.getKeyPoints()[i].pt.y) * (v - frame.getKeyPoints()[i].pt.y);

        totalError += error;

        if (error > 5.991)
        {
            frame.getOutliers()[i] = true;
            nBad++;
            mapPoint->increaseBad();
            mapPoint = nullptr;
            // if(mapPoint->getBadCount() >= 3)
            // {
            //     mapPoint->setBad();
            //     mapPoint = nullptr;
            // }
        }
        else
        {
            mapPoint->increaseVisible();
        }
    }

    return (N - nBad);
}

void Optimizer::bundleAdjust(std::shared_ptr<Map> pMap, Frame *newFrame, bool *stop, SlamParams* slamParams)
{
    int iterations1 = slamParams->optimizationParams.BAIterations1;
    int iterations2 = slamParams->optimizationParams.BAIterations2;


    //ID used to avoid repeating elements for BA
    unsigned long int activeBAID = newFrame->getID();
    //used to insert data as vertices for optimization
    unsigned long int maxID = 0;
    const float thHuberMono = sqrt(5.991);

    const float fx = newFrame->getCamPrjParams().fx;
    const float fy = newFrame->getCamPrjParams().fy;
    const float cx = newFrame->getCamPrjParams().cx;
    const float cy = newFrame->getCamPrjParams().cy;

    newFrame->setActiveBAID(activeBAID);

    //Fetch all MPs from current frame, then find all frames that are connected/local
    std::list<Frame*> localFrames;

    //push self (frame) to the list
    localFrames.push_back(newFrame);

    //add all connected frames (local)
    //const std::vector<Frame*> connectedFrames = newFrame->getConnectedFrames();
    const std::vector<Frame*> connectedFrames = newFrame->getNConnectedFrames(slamParams->optimizationParams.SearchN);
    for(size_t i = 0; i < connectedFrames.size(); i++)
    {
        Frame* frame = connectedFrames[i];
        if(frame != nullptr)
        {
            frame->setActiveBAID(activeBAID);
            localFrames.push_back(frame);
        }
    }

    //add map points
    //for every frame in local map, fetch map points and add to local map points (udpdate Active BA id)

    std::list<MapPoint*> localMapPoints;
    for(std::list<Frame*>::iterator it = localFrames.begin(), itEnd = localFrames.end(); it!=itEnd; it++)
    {
        std::vector<MapPoint*> pts = (*it)->getMapPoints();
        for(size_t i = 0; i < pts.size(); i++)
        {
            MapPoint* mapPoint = pts[i];
            if(mapPoint != nullptr)
            {
                if (mapPoint->getActiveBAID() != activeBAID)
                {
                    mapPoint->setActiveBAID(activeBAID);
                    localMapPoints.push_back(mapPoint);
                }
            }
        }
    }

//    add also other frames that views the map-points from the local map (as fixed)
//    same as map points, avoid reptition
    std::list<Frame*> fixedFrames;
    for(std::list<MapPoint*>::iterator it=localMapPoints.begin(), itEnd=localMapPoints.end(); it!=itEnd; it++)
    {
        std::map<Frame *, size_t>  frameViews = (*it)->getFrameViews();
        for(std::map<Frame*,size_t>::iterator it = frameViews.begin(), itEnd=frameViews.end();it!=itEnd;it++)
        {
            Frame* frame = (*it).first;
            if(frame->getActiveBAID()!=activeBAID)
            {
                frame->setActiveBAID(activeBAID);
                fixedFrames.push_back(frame);
            }
        }
    }


    //optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();

    std::unique_ptr<g2o::BlockSolver_6_3> solver_ptr (new g2o::BlockSolver_6_3(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    optimizer.setAlgorithm(solver);

    //add data as vertices
    //start with frames from local map
    for(std::list<Frame*>::iterator it=localFrames.begin(), itEnd=localFrames.end();it!=itEnd;it++)
    {
        Frame* frame = *it;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(toSE3Quat(frame->getPosec()));
        vSE3->setId(frame->getID());
        vSE3->setFixed(frame->getID()==0);
        optimizer.addVertex(vSE3);
        if(frame->getID() > maxID)
            maxID = frame->getID();
    }

 //   add fixed frames
    for(std::list<Frame*>::iterator it=fixedFrames.begin(), itEnd=fixedFrames.end();it!=itEnd;it++)
    {
        Frame* frame = *it;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(toSE3Quat(frame->getPosec()));
        vSE3->setId(frame->getID());
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(frame->getID() > maxID)
            maxID = frame->getID();
    }


    //reserve enough edges
    const int Nedges = ((localFrames.size()+fixedFrames.size()) * localMapPoints.size());
    std::vector<g2o::EdgeSE3ProjectXYZ*> edges;
    std::vector<MapPoint*>  edgeMapPoints;
    std::vector<Frame*>     edgeFrames;

    edges.reserve(Nedges);
    edgeFrames.reserve(Nedges);
    edgeMapPoints.reserve(Nedges);



    for(std::list<MapPoint*>::iterator mit=localMapPoints.begin(), mitEnd=localMapPoints.end();mit!=mitEnd;mit++)
    {
        MapPoint* mapPoint = *mit;
        g2o::VertexPointXYZ* vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(toEigenVector3D(mapPoint->getPosition()));
        unsigned long int id = mapPoint->getID() + maxID + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const std::map<Frame*,size_t> frameViews = mapPoint->getFrameViews();

        //get frame from local map, make edge-connection
        for(std::map<Frame*,size_t>::const_iterator fit=frameViews.begin(),fitEnd=frameViews.end();fit!=fitEnd;fit++)
        {
            Frame* frame = (*fit).first;
            int idx = 0;

            //get index of map point on frame
            if(mapPoint->isInFrameView(frame))
                idx = mapPoint->getFrameViews()[frame];

            else
                continue;

            //fetch image plane 2d pt
            cv::KeyPoint kp = frame->getKeyPoints()[idx];
            Eigen::Matrix<double,2,1> pt2D;
            pt2D << kp.pt.x, kp.pt.y;

            g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
            edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame->getID())));
            edge->setMeasurement(pt2D);
            edge->setInformation(2.0*Eigen::Matrix2d::Identity());

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            edge->setRobustKernel(rk);
            rk->setDelta(thHuberMono);

            edge->fx = fx;
            edge->fy = fy;
            edge->cx = cx;
            edge->cy = cy;

            optimizer.addEdge(edge);
            edges.push_back(edge);
            edgeMapPoints.push_back(mapPoint);
            edgeFrames.push_back(frame);
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(iterations1);

    //check inliers
    for(size_t i = 0; i < edges.size(); i++)
    {
        g2o::EdgeSE3ProjectXYZ* edge = edges[i];
        MapPoint* mapPoint = edgeMapPoints[i];

        //failing criteria
        if(mapPoint== nullptr)
            continue;
        float chi2Error = edge->chi2();

        bool depthPositive = edge->isDepthPositive();
        if(chi2Error>5.991 || !depthPositive)
            edge->setLevel(1);

        edge->setRobustKernel(0);
    }

    optimizer.initializeOptimization(0);
    optimizer.optimize(iterations2);

    std::vector<std::pair<Frame*,MapPoint*> > removeEdges;
    removeEdges.reserve(edges.size());

    //check inliers and set frames - map points to delete
    for(size_t i = 0; i < edges.size(); i++)
    {
        g2o::EdgeSE3ProjectXYZ* edge = edges[i];
        MapPoint* mapPoint = edgeMapPoints[i];

        //failing criteria
        if(mapPoint== nullptr)
            continue;
        float chi2Error = edge->chi2();
        if(edge->chi2()>5.991 || !edge->isDepthPositive())
        {
            Frame* frame = edgeFrames[i];
            removeEdges.push_back(std::make_pair(frame,mapPoint));
        }
    }

    //Remove Data
    std::unique_lock<std::mutex> lock(pMap->getMutexMap());
    for(size_t i = 0; i < removeEdges.size(); i++)
    {
        Frame* frame        = removeEdges[i].first;
        MapPoint* mapPoint  = removeEdges[i].second;
        frame->removeMapPoint(mapPoint);
        mapPoint->removeFrameView(frame);
    }

    //Fetch optimized data
    //Frames
    for(std::list<Frame*>::iterator it=localFrames.begin(),itEnd=localFrames.end();it!=itEnd;it++)
    {
        Frame* frame = *it;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame->getID()));
        g2o::SE3Quat sE3Quat = vSE3->estimate();
        frame->setPose(toCVMat(sE3Quat));
    }

    //Point
    for(std::list<MapPoint*>::iterator it=localMapPoints.begin(),itend=localMapPoints.end();it!=itend;it++)
    {
        MapPoint* mapPoint = *it;
        g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(mapPoint->getID() + maxID + 1));
        mapPoint->setPos(toCVMat(vPoint->estimate()));
        //update normal and depth
    }

}

void Optimizer::bundleAdjustN(const std::vector<Frame *> &vpFrames, const std::vector<MapPoint *> &vpMapPoints, const bool isSetRobust, const int iterations)
{
    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);
    unsigned long int maxID = 0;

    if(vpFrames.size() <= 0)
        return;

    const float fx = vpFrames[0]->getCamPrjParams().fx;
    const float fy = vpFrames[0]->getCamPrjParams().fy;
    const float cx = vpFrames[0]->getCamPrjParams().cx;
    const float cy = vpFrames[0]->getCamPrjParams().cy;

    std::vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMapPoints.size());

    //optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();

    std::unique_ptr<g2o::BlockSolver_6_3> solver_ptr (new g2o::BlockSolver_6_3(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices
    for(size_t i =0; i < vpFrames.size(); i++)
    {
        Frame* frame = vpFrames[i];
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(toSE3Quat(frame->getPosec()));
        vSE3->setId(frame->getID());
        vSE3->setFixed(frame->getID()==0);
        optimizer.addVertex(vSE3);
        if(frame->getID() > maxID)
            maxID = frame->getID();
    }

    // Set MapPoint vertices
    for(size_t i=0; i<vpMapPoints.size(); i++)
    {
        MapPoint* mapPoint = vpMapPoints[i];
        if((mapPoint == nullptr) || (mapPoint->isBad()))
            continue;

        g2o::VertexPointXYZ* vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(toEigenVector3D(mapPoint->getPosition()));
        const int id = mapPoint->getID()+maxID+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const std::map<Frame*,size_t> frameViews = mapPoint->getFrameViews();

        int nEdges = 0;
        //SET EDGES
        for(std::map<Frame*,size_t>::const_iterator mit=frameViews.begin(); mit!=frameViews.end(); mit++)
        {

            Frame* frame = mit->first;
            if(frame == nullptr || frame->getID()>maxID)
                continue;

            nEdges++;

            //fetch image plane 2d pt
            cv::KeyPoint kp = frame->getKeyPoints()[mit->second];
            Eigen::Matrix<double,2,1> pt2D;
            pt2D << kp.pt.x, kp.pt.y;

            g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();

            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame->getID())));
            edge->setMeasurement(pt2D);
            edge->setInformation(Eigen::Matrix2d::Identity());


            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            edge->setRobustKernel(rk);
            rk->setDelta(thHuber2D);


            edge->fx = fx;
            edge->fy = fy;
            edge->cx = cx;
            edge->cy = cy;

            optimizer.addEdge(edge);


        }

        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(iterations);

    // Recover optimized data

    //Keyframes
    for(size_t i=0; i<vpFrames.size(); i++)
    {
        Frame* frame = vpFrames[i];
        if(frame == nullptr)
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame->getID()));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        frame->setPose(toCVMat(SE3quat));
    }

    //Points
    for(size_t i=0; i<vpMapPoints.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* mapPoint = vpMapPoints[i];

        if((mapPoint == nullptr) || (mapPoint->isBad()))
            continue;
        g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(mapPoint->getID()+maxID+1));


        mapPoint->setPos(toCVMat(vPoint->estimate()));
        //mapPoint->UpdateNormalAndDepth();

    }


}



static inline g2o::SE3Quat toSE3Quat(const cv::Mat &cvT)
{
    Eigen::Matrix<double,3,3> R;
    R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
         cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
         cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

    return g2o::SE3Quat(R,t);
}

static inline cv::Mat toCVMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
    return toCVMat(eigMat);
}

static inline cv::Mat toCVMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

static inline cv::Mat toCVMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

static inline cv::Mat toCVMat(const Eigen::Matrix<double,3,1> &m)
{
    cv::Mat cvMat(3,1,CV_32F);
    for(int i=0;i<3;i++)
        cvMat.at<float>(i)=m(i);

    return cvMat.clone();
}
