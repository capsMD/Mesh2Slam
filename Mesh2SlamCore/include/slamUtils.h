//
// Created by caps80 on 10.10.23.
//

#ifndef CAPSLAM_UTILS_H
#define CAPSLAM_UTILS_H

#include <iostream>
#include <chrono>
#include <ctime>
#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <glm/glm.hpp>

//needed to read files in Android
#include <HelperFunctions.h>


#include "slamParams.h"

#define BLACK_TEXT   "\033[30m"
#define RED_TEXT     "\033[31m"
#define GREEN_TEXT   "\033[32m"
#define YELLOW_TEXT  "\033[33m"
#define BLUE_TEXT    "\033[34m"
#define MAGENTA_TEXT "\033[35m"
#define CYAN_TEXT    "\033[36m"
#define WHITE_TEXT   "\033[37m"
#define RESET_TEXT   "\033[0m"
#define WIN_FOREGROUND_WHITE (FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)
#define WIN_FOREGROUND_YELLOW (FOREGROUND_RED | FOREGROUND_GREEN)

template<typename T>
class Logger
{

public:
    static void LogInfoI(const T& msg)
    {
        //standard color
        std::string color = BLUE_TEXT;
        std::string prefix = color;
        log(prefix,msg);
    }

    static void LogInfoII(const T& msg)
    {
        //standard color
        std::string color = WHITE_TEXT;
        std::string prefix = color;
        log(prefix,msg);
    }

    static void LogInfoIII(const T& msg)
    {
        //standard color
        std::string color = GREEN_TEXT;
        std::string prefix = color;
        log(prefix,msg);
    }

    static void LogInfoIV(const T& msg)
    {
        //standard color
        std::string color = CYAN_TEXT;
        std::string prefix = color;
        log(prefix,msg);
    }
    static void LogTime(const T& msg)
    {
        //standard color
        std::string color = CYAN_TEXT;
        std::string prefix = color + "[TIME] ";
        log(prefix,msg);
    }

    static void LogWarning(const T& msg)
    {
        //standard color
        std::string color = YELLOW_TEXT ;
        std::string prefix = color;
        log(prefix,msg);
    }

    static void LogError(const T& msg)
    {
        //standard color
        std::string color = RED_TEXT ;
        std::string prefix = color;
        log(prefix,msg);
    }

private:

    static void log(const std::string& prefix, const T& msg)
    {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::clog << prefix << msg << WHITE_TEXT <<std::endl;
    }
};

static bool readImgSimPts(const std::string& file,
                          std::vector<cv::Point3i>& imgPts,
                          const unsigned long int frameCount,
                          const unsigned long int maxPts,
                          AAssetManager *assetManager)
{
    imgPts.clear();
    std::vector<cv::Point3i> tempPts;
    //std::ifstream imgFile;
    std::string fileNumber = std::to_string(frameCount);
    std::string fileName = file + fileNumber + ".txt";

    std::string fileContent = ReadTextFile(fileName, assetManager);

    if(fileContent.empty())
        return false;

    std::istringstream imgFile(fileContent);

        std::string line;
        unsigned long int lineNumber = 0;
        while(std::getline(imgFile,line))
        {
            //read number by number, separated by delimiter
            std::istringstream s(line);
            std::string number;
            size_t counter = 0;
            int firstElement = 0;
            std::vector<int> v(3,0);
            while (std::getline(s, number, ','))
            {
                std::istringstream n(number);
                int d;
                n >> d;
                v[counter] = d;
                counter++;

                if(counter > 2)
                {
                //idx0 = ptMeshIndex, idx1=ptx, idx2=pty
                cv::Point3i pt(v[0],v[1],v[2]);
                tempPts.push_back(pt);

                }
            }
        }
        size_t npts = tempPts.size();
        if(tempPts.size() > maxPts)
        {
            imgPts.reserve(maxPts);
            imgPts = std::vector<cv::Point3i>(tempPts.begin(),tempPts.begin()+maxPts);
        }
        else
            imgPts = tempPts;

        return true;

}
static bool readvtxPts(const std::string& file, std::vector<cv::Point3f>& imgPts, const unsigned long int frameCount, const unsigned long int maxPts)
{
    imgPts.clear();
    std::vector<cv::Point3f> tempPts;
    std::ifstream imgFile;
    std::string fileNumber = std::to_string(frameCount);
    std::string fileName = file + fileNumber + ".txt";
    imgFile.open(fileName, std::ios::in);

    if(imgFile.is_open())
    {
        std::string line;
        unsigned long int lineNumber = 0;
        while(std::getline(imgFile,line))
        {
            //read number by number, separated by delimiter
            std::istringstream s(line);
            std::string number;
            size_t counter = 0;
            std::vector<float> v(4,0);
            while (std::getline(s, number, ','))
            {
                std::istringstream n(number);
                float d;
                n >> d;
                v[counter] = d;
                counter++;

                if(counter > 3)
                {
                    //idx0 = ptMeshIndex, idx1=ptx, idx2=pty
                    cv::Point3f pt(v[1],v[2],v[3]);
                    tempPts.push_back(pt);

                }
            }
        }
        size_t npts = tempPts.size();
        if(tempPts.size() > maxPts)
        {
            imgPts.reserve(maxPts);
            imgPts = std::vector<cv::Point3f>(tempPts.begin(),tempPts.begin()+maxPts);
        }
        else
            imgPts = tempPts;

        imgFile.close();
        return true;
    }
    else
    {
        //Logger::Log("Error Reading file: " + file, 3);
        return false;
    }

}
static bool readR(const std::string& file, cv::Mat& R, const unsigned long int frameCount)
{
    R = cv::Mat::eye(3,3,CV_32F);
    std::ifstream File;
    std::string fileNumber = std::to_string(frameCount);
    std::string fileName = file + fileNumber + ".txt";
    File.open(fileName, std::ios::in);
    if(File.is_open())
    {
        std::string line;
        unsigned long int lineNumber = 0;
        std::vector<std::vector<float> > matrix;
        while(std::getline(File,line))
        {
            std::istringstream s(line);
            std::string number;
            size_t counter = 0;
            std::vector<float> row;
            while(std::getline(s,number,','))
            {
                std::istringstream n(number);
                float d;
                n >> d;
                row.push_back(d);
                counter++;
                if(counter > 2)
                {
                    counter = 0;
                    matrix.push_back(row);
                }
            }
        }

        size_t rows = matrix.size();
        if(rows == 3)
        {
            size_t cols = matrix[0].size();
            if(cols == 3)
            {
                for (size_t i = 0; i < rows; i++)
                    for (size_t j = 0; j < cols; j++)
                        R.at<float>(i, j) = matrix[i][j];

                File.close();
                return true;
            }
        }

    }
    File.close();
    return false;
}
static bool readt(const std::string& file, cv::Mat& t, const unsigned long int frameCount)
{
    t = cv::Mat::eye(3,1,CV_32F);
    std::ifstream File;
    std::string fileNumber = std::to_string(frameCount);
    std::string fileName = file + fileNumber + ".txt";
    File.open(fileName, std::ios::in);

    if(File.is_open())
    {
        std::string line;
        unsigned long int lineNumber = 0;
        while(std::getline(File,line))
        {
            //read number by number, separated by delimiter
            std::istringstream s(line);
            std::string number;
            size_t counter = 0;
            std::vector<float> v(3,0.0f);
            while (std::getline(s, number, ','))
            {
                std::istringstream n(number);
                float d;
                n >> d;
                t.at<float>(counter) = d;
                counter++;

                if(counter > 2)
                {
                   File.close();
                   return true;

                }
            }
        }
    }
    File.close();
    return false;
}
static bool readImg2DMeasurementsFile(const std::string& file, std::vector<cv::Point2i>& imgPts, const unsigned long int frameCount)
{
    bool result = false;

    std::ifstream imgMFile;
    imgMFile.open(file, std::ios::in);
    if(imgMFile.is_open())
    {
        std::string line;
        unsigned long int lineNumber = 0;
        while(std::getline(imgMFile,line))
        {
            if(lineNumber>frameCount)
                break;
            if(lineNumber == frameCount)
            {
                //read number by number, separated by delimiter
                std::istringstream s(line);
                std::string number;
                char counter = 0;
                int firstElement = 0;
                while (std::getline(s, number, ','))
                {
                    std::istringstream n(number);
                    int d;
                    n >> d;
                    if(counter < 1) {
                        firstElement = d;
                        counter++;
                    }
                    else
                    {
                        cv::Point2i pt(firstElement,d);
                        imgPts.push_back(pt);
                        counter = 0;
                    }
                }
                return true;
            }
            lineNumber++;

        }

        imgMFile.close();
    }
    else
    {
    }

    return result;
}

static bool readImg3DMeasurementsFile(const std::string& file, std::vector<cv::Point3f>& pts, const unsigned long int frameCount)
{
    bool result = false;

    std::ifstream imgMFile;
    imgMFile.open(file, std::ios::in);
    if(imgMFile.is_open())
    {
        std::string line;
        unsigned long int lineNumber = 0;
        while(std::getline(imgMFile,line))
        {
            lineNumber++;
            if(lineNumber == frameCount)
            {

            }
        }

        imgMFile.close();
    }
    else
    {
    }

    return result;
}
static bool readCamOrientationsFile(const std::string& file, cv::Mat& R, const unsigned long int frameCount)
{
    bool result = false;
    const size_t nLines = 3;
    std::ifstream imgMFile;
    imgMFile.open(file, std::ios::in);
    if(imgMFile.is_open())
    {
        std::string line;
        unsigned long int lineNumber = 0;
        char rowCounter = 0;
        char colCounter = 0;
        R = cv::Mat(3,3,CV_32F);

        while(std::getline(imgMFile,line) && rowCounter < 3)
        {
            if(lineNumber == (frameCount*nLines)+rowCounter)
            {
                //read number by number, separated by delimiter
                std::istringstream s(line);
                std::string number;

                int firstElement = 0;
                while (std::getline(s, number, ','))
                {
                    std::istringstream n(number);
                    float d;
                    n >> d;
                    if(rowCounter <= 2)
                    {
                        if(colCounter <= 2)
                        {
                            R.at<float>(rowCounter, colCounter) = d;
                            colCounter++;
                            if(colCounter == 3)
                            {
                                colCounter = 0;
                                rowCounter ++;
                            }

                        }
                    }
                }
            }
            lineNumber++;
        }
        imgMFile.close();
        return true;
    }

    else
    {
    }

    return result;
}
static bool readCamTranslationsFile(const std::string& file, cv::Mat& t, const unsigned long int frameCount)
{
    bool result = false;

    std::ifstream imgMFile;
    imgMFile.open(file, std::ios::in);
    if(imgMFile.is_open())
    {
        std::string line;
        unsigned long int lineNumber = 0;
        while(std::getline(imgMFile,line))
        {
            if(lineNumber == frameCount)
            {
                //read number by number, separated by delimiter
                std::istringstream s(line);
                std::string number;
                char counter = 0;
                int firstElement = 0;
                t = cv::Mat(3,1,CV_32F);
                while (std::getline(s, number, ','))
                {
                    std::istringstream n(number);
                    float d;
                    n >> d;
                    if(counter < 3)
                    {
                        t.at<float>(counter) = d;
                        counter++;
                    }
                }
                return true;
            }
            lineNumber++;
        }
        imgMFile.close();

    }

    else
    {
    }

    return result;
}

static glm::vec3 readInVector(cv::FileStorage& fs, const std::string& parameter)
{
    std::vector<float> v;
    cv::FileNode fn = fs[parameter];

    if (fn.empty() || !fn.isSeq())
        std::cout << "Failed to read" + parameter + " from file." << std::endl;

    fn >> v;
    return glm::vec3(v[0],v[1],v[2]);
}

static bool readConfigFile(const std::string& configFile, SlamParams*& slamParams,AAssetManager *assetManager)
{
    std::cout << "Reading Config file:" + configFile << std::endl;
    std::string yamlFile = ReadTextFile(configFile, assetManager);
    if(yamlFile.empty())
    {
        std::cout << "Error reading Config file, YAML EMPTY!" << std::endl;
        return false;
    }
    cv::FileStorage fs(yamlFile, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if (!fs.isOpened())
    {
        //Logger::Log("Failed to open configuration file.", 3);
        return false;
    }

    //read in input source slamParams
    slamParams->inputParams.runMode	                = fs["InputSource.runMode"];
    slamParams->inputParams.type	                = fs["InputSource.type"];
    slamParams->inputParams.useCam	                = fs["InputSource.useCamDevice"];
    slamParams->inputParams.camDevice               = fs["InputSource.camDeviceNumber"];
    slamParams->inputParams.fileDir                 = static_cast<std::string>(fs["InputSource.sourceFileDir"]);
    slamParams->inputParams.sourceFileName          = static_cast<std::string>(fs["InputSource.sourceFileName"]);
    slamParams->inputParams.imageFilesSubdirectory  = static_cast<std::string>(fs["InputSource.imageFilesSubDir"]);
    slamParams->inputParams.camCaptureString        = static_cast<std::string>(fs["InputSource.captureString"]);

    //read in camera sensor slamParams
    slamParams->camParams.fx				= fs["Camera.fx"];
    slamParams->camParams.fy				= fs["Camera.fy"];
    slamParams->camParams.cx				= fs["Camera.cx"];
    slamParams->camParams.cy				= fs["Camera.cy"];
    slamParams->camParams.k1				= fs["Camera.k1"];
    slamParams->camParams.k2				= fs["Camera.k2"];
    slamParams->camParams.p1				= fs["Camera.p1"];
    slamParams->camParams.p2				= fs["Camera.p2"];
    slamParams->camParams.k3				= fs["Camera.k3"];
    slamParams->camParams.w                 = fs["Camera.w"];
    slamParams->camParams.h                 = fs["Camera.h"];

    //read in optimization slamParams
    slamParams->optimizationParams.BAIterations1        = fs["Optimization.BAIterations1"];
    slamParams->optimizationParams.BAIterations2        = fs["Optimization.BAIterations2"];
    slamParams->optimizationParams.GBAIterations        = fs["Optimization.GBAIterations"];
    slamParams->optimizationParams.PNPIterations        = fs["Optimization.PNPIterations"];
    slamParams->optimizationParams.SearchN              = fs["Optimization.SearchN"];
    slamParams->optimizationParams.minJumpFrames        = fs["Optimization.minJumpFrames"];
    slamParams->optimizationParams.maxJumpFrames        = fs["Optimization.maxJumpFrames"];
    slamParams->optimizationParams.GBAJumpFrames        = fs["Optimization.GBAJumpFrames"];
    slamParams->optimizationParams.minDisplacement      = fs["Optimization.minDisplacement"];

    //read in map points slamParams
    slamParams->mapPointParams.maxPointDistance         = fs["MapPoint.maxPointDistance"];
    slamParams->mapPointParams.maxNewPoints             = fs["MapPoint.maxNewPoints"];

    //read in feature slamParams
    slamParams->featureParams.maxFeatures               = fs["Feature.maxFeatures"];
    slamParams->featureParams.minFeatures	            = fs["Feature.minFeatures"];
    slamParams->featureParams.minMatches	            = fs["Feature.minMatches"];
    slamParams->featureParams.maxMatches	            = fs["Feature.maxMatches"];
    slamParams->featureParams.pruneDist	                = fs["Feature.pruneDist"];
    slamParams->featureParams.minRefFrameMatches	    = fs["Feature.minRefFrameMatches"];
    slamParams->featureParams.searchRadius              = fs["Feature.searchRadius"];
    slamParams->featureParams.minTracked                = fs["Feature.minTracked"];
    slamParams->featureParams.featuresMaxDepth          = fs["Feature.featuresMaxDepth"];

    //read in ransac slamParams
    slamParams->ransacParams.maxTrials		= fs["Ransac.maxTrials"];
    slamParams->ransacParams.threshold		= fs["Ransac.threshold"];
    slamParams->ransacParams.confidence	    = fs["Ransac.confidence"];
    slamParams->ransacParams.samples		= fs["Ransac.samples"];

    //read in error slamParams
    slamParams->errorParams.maxBadTrackCnt		= fs["Error.maxBadTrackCnt"];
    slamParams->errorParams.minInlierRatio		= fs["Error.minInlierRatio"];
    slamParams->errorParams.minParallax		    = fs["Error.minParallax"];
    slamParams->errorParams.reprojectThreshold  = fs["Error.reprojectThreshold"];
    slamParams->errorParams.minTriangulations	= fs["Error.minTriangulations"];
    slamParams->errorParams.maxSecondPassRatio	= fs["Error.maxSecondPassRatio"];
    slamParams->errorParams.maxBBRepDist	    = fs["Error.maxBBRepDist"];

    //read in viewer slamParams
    slamParams->viewerParams.runViewer          = fs["Viewer.runViewer"];
    slamParams->viewerParams.width              = fs["Viewer.width"];
    slamParams->viewerParams.height             = fs["Viewer.height"];
    slamParams->viewerParams.title              = static_cast<std::string>(fs["Viewer.title"]);
    slamParams->viewerParams.scaleFactor        = fs["Viewer.scaleFactor"];
    slamParams->viewerParams.camMoveFactor      = fs["Viewer.mouseMoveFactor"];
    slamParams->viewerParams.frameColor         = readInVector(fs, "Viewer.frameColor");
    slamParams->viewerParams.tframeColor        = readInVector(fs, "Viewer.tframeColor");
    slamParams->viewerParams.framePathColor     = readInVector(fs, "Viewer.framePathColor");
    slamParams->viewerParams.pointCloudColor    = readInVector(fs, "Viewer.pointCloudColor");
    slamParams->viewerParams.meshColor          = readInVector(fs, "Viewer.meshColor");
    slamParams->viewerParams.fov                = fs["Viewer.fov"];
    slamParams->viewerParams.far                = fs["Viewer.far"];
    slamParams->viewerParams.near               = fs["Viewer.near"];
    slamParams->viewerParams.forceOriginStart   = fs["Viewer.forceOriginStart"];

    return true;
}
static void trimString(std::string& str)
{
    const char *whiteSpace = " \t\n\r";
    size_t location;
    location = str.find_first_not_of(whiteSpace);
    str.erase(0,location);
    location = str.find_last_not_of(whiteSpace);
    str.erase(location+1);
}

//Used in simulation: 3d features used as image features idx and pos
// and cam position can be used to set scale factor during initialization
struct VertexFeatures
{
    VertexFeatures(): m_pts(std::vector<cv::Point3i>(0, cv::Point3i(0, 0, 0))), m_pose(glm::mat4(0)){}
    std::vector<cv::Point3i> m_pts;
    glm::mat4 m_pose;
};
#endif //CAPSLAM_UTILS_H
