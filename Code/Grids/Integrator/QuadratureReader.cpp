#include "QuadratureReader.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "../Logger.h"

CQuadratureReader::CQuadratureReader(const std::string& folderName, EIntegralDimension eD, int iMaxOrder)
{
    switch (eD)
    {
    case EIntegralDimension::EID_4D:
        {
            Read4D(folderName, iMaxOrder);
        }
        break;
    default:
        break;
    }

}

std::vector<double> CQuadratureReader::ParseLine(std::string& line)
{
    std::vector<double> ret;
    const char* toremove = " \t\v\n\r\f";
    std::string::iterator it;
    for (int i = 0; i < 5; i++) 
    {
        ReplaceAll(line, toremove[i], "");
    }

    if (',' != line.at(line.size() - 1))
    {
        line = line + ",";
    }

    size_t pos = line.find(',');
    while (pos != std::string::npos)
    {
        std::string temp = line.substr(0, pos);
        line = line.substr(pos + 1, line.size());
        char* pend;
        double oneNumber = std::strtod(temp.c_str(), &pend);
        if (pend != temp.c_str())
        {
            ret.push_back(oneNumber);
        }
        pos = line.find(',');
    }
    return ret;
}

void CQuadratureReader::PrintLine(const std::vector<double> numbers)
{
    printf("{ ");
    size_t numbersize = numbers.size();
    if (0 == numbersize)
    {
        printf("}\n");
        return;
    }
    for (size_t i = 1; i < numbersize; ++i)
    {
        printf("%.12f, ", numbers[i - 1]);
    }
    printf("%.12f }\n", numbers[numbersize - 1]);
}

void CQuadratureReader::Read4D(const std::string& folderName, int iMaxOrder)
{
    m_lstNewPoints.clear();
    m_lstWeights.clear();

    int iReadedPoints = 0;
    if (iMaxOrder < 0)
    {
        iMaxOrder = 100;
    }
    int iReadOrder = 0;
    while (iMaxOrder < 0 || iReadOrder < iMaxOrder)
    {
        std::string fileName = folderName + "/pt" + std::to_string(iReadOrder);
        std::ifstream file(fileName);
        if (file.bad() || file.fail())
        {
            break;
        }

        std::vector<std::vector<double>> points;
        std::vector<double> weights;
        //read points
        std::string line;
        while (std::getline(file, line))
        {
            std::vector<double> newpoint = ParseLine(line);
            if (4 == newpoint.size())
            {
                points.push_back(newpoint);
            }
            else
            {
                LogCrucial("loading 4D quadrature, but point length smaller than 4: \"%s\" \n", line.c_str());
            }
        }

        fileName = folderName + "/wt" + std::to_string(iReadOrder);
        std::ifstream filewt(fileName);
        //read weight
        while (std::getline(filewt, line))
        {
            char* pend;
            double oneNumber = std::strtod(line.c_str(), &pend);
            if (pend != line.c_str())
            {
                weights.push_back(oneNumber);
            }
        }

        iReadedPoints = iReadedPoints + static_cast<int>(points.size());
        if (weights.size() == iReadedPoints)
        {
            LogParanoiac("reading order %d: pts = %d, wts = %d\n", 
                iReadOrder, 
                static_cast<int>(points.size()), 
                static_cast<int>(weights.size())
            );
            m_lstNewPoints.push_back(points);
            m_lstWeights.push_back(weights);
        }
        else
        {
            LogCrucial("reading order %d: allpts = %d pts = %d, wts = %d\n", 
                iReadOrder,
                iReadedPoints, 
                static_cast<int>(points.size()), 
                static_cast<int>(weights.size())
            );
        }
        ++iReadOrder;
    }
}