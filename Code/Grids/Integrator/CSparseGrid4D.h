#pragma once

#include <vector>
#include <string>
#include "cuComplex.h"
#include "../Logger.h"

typedef cuDoubleComplex (*integrand4d)(const cuDoubleComplex& x, const cuDoubleComplex& y, const cuDoubleComplex& z, const cuDoubleComplex& w);

class CSparseGrid4D : public CLogger
{
public:

	//CSparseGrid4D("../Data/SparseGrid4D/GaussPatterson/")
	CSparseGrid4D(const std::string& folderName, int iMaxOrder = -1, double epsilon = 1.0e-4);
	~CSparseGrid4D();

	bool Integrate(integrand4d integrand,
		cuDoubleComplex xfrom, cuDoubleComplex xto,
		cuDoubleComplex yfrom, cuDoubleComplex yto,
		cuDoubleComplex zfrom, cuDoubleComplex zto,
		cuDoubleComplex wfrom, cuDoubleComplex wto,
		cuDoubleComplex& res);


protected:

	int m_iOrder;
	double m_fEpsilon;
	std::vector<int> m_iBlockCountStep1;
	std::vector<int> m_iBlockCountStep2;
	std::vector<int> m_iPointStart;
	std::vector<int> m_iPointEnd;
	std::vector<int> m_iWeightCount;
	double** m_pWeightBuffer;
	double* m_pPointBuffer;
	cuDoubleComplex* m_pValueBuffer;
	cuDoubleComplex* m_pValueBufferMult;
};
