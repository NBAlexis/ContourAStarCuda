#pragma once

#include <vector>
#include <string>

enum class EIntegralDimension
{
	EID_1D,
	EID_2D,
	EID_3D,
	EID_4D,
};

class CQuadratureReader
{
public:

	CQuadratureReader(const std::string& folderName, EIntegralDimension eD, int iMaxOrder = -1);

	static std::vector<double> ParseLine(std::string& line);
	static void PrintLine(const std::vector<double> numbers);

	static std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) 
	{
		size_t start_pos = 0;
		while ((start_pos = str.find(from, start_pos)) != std::string::npos) 
		{
			str.replace(start_pos, from.length(), to);
			start_pos += to.length();
		}
		return str;
	}

	static std::string ReplaceAll(std::string str, char from, const std::string& to)
	{
		size_t start_pos = 0;
		while ((start_pos = str.find(from, start_pos)) != std::string::npos)
		{
			str.replace(start_pos, 1, to);
			start_pos += to.length();
		}
		return str;
	}

	std::vector<std::vector<std::vector<double>>> m_lstNewPoints;
	std::vector<std::vector<double>> m_lstWeights;

protected:

	void Read4D(const std::string& folderName, int iMaxOrder);
};
