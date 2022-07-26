#pragma once

#include "Integrator/CSparseGrid4D.h"
#include "CGeneralGrids.h"
#include "CExtendedPathCommon.h"
#include "Logger.h"

#include <unordered_set>
#include <unordered_map>

const int _kDefaultListCapEP4D = 16384;

class CCheck4DW : public IConnectionChecker
{
public:
	CCheck4DW(class CExtendedPath4D* pOwner) : m_pOwner(pOwner) {}
	bool IsConnected(const COneGrid& grid1, const COneGrid& grid2, cuDoubleComplex& v) override;
	class CExtendedPath4D* m_pOwner;
};

class CCheck4DZ : public IConnectionChecker
{
public:
	CCheck4DZ(class CExtendedPath4D* pOwner) : m_pOwner(pOwner) {}
	bool IsConnected(const COneGrid& grid1, const COneGrid& grid2, cuDoubleComplex& v) override;
	class CExtendedPath4D* m_pOwner;
};

class CCheck4DY : public IConnectionChecker
{
public:
	CCheck4DY(class CExtendedPath4D* pOwner) : m_pOwner(pOwner) {}
	bool IsConnected(const COneGrid& grid1, const COneGrid& grid2, cuDoubleComplex& v) override;
	class CExtendedPath4D* m_pOwner;
};

class CCheck4DX : public IConnectionChecker
{
public:
	CCheck4DX(class CExtendedPath4D* pOwner) : m_pOwner(pOwner) {}
	bool IsConnected(const COneGrid& grid1, const COneGrid& grid2, cuDoubleComplex& v) override;
	class CExtendedPath4D* m_pOwner;
};

class CExtendedPath4D : public CLogger
{
public:

	CExtendedPath4D(int iWidth, int iHeight, int iEdge, const std::string& folderName, 
		int iMaxOrder = -1, double epsilon = 1.0e-4);
	SIntegrateRes Integrate(integrand4d integrand);

	friend class CCheck4DX; 
	friend class CCheck4DY; 
	friend class CCheck4DZ;
	friend class CCheck4DW;

	void SetIntegratorVerb(EVerboseLevel eLevel) { m_Integrator.SetVerb(eLevel); }
	void PrintResoult(const std::string& sFunc);

protected:

	bool IntegrateHyperCubic(
		const COneGrid& fromX, const COneGrid& toX,
		const COneGrid& fromY, const COneGrid& toY,
		const COneGrid& fromZ, const COneGrid& toZ,
		const COneGrid& fromW, const COneGrid& toW,
		cuDoubleComplex& res);

	void AddYZWPath(const CPath& YPath, const CPath& ZPath, const CPath& WPath, const CIntervalList& xInterval);
	void AddZWPath(const CPath& ZPath, const CPath& WPath, const CIntervalList& yInterval);
	void AddWPath(const CPath& WPath, const CIntervalList& yInterval);

	bool CalculateConnectionFindW(const COneGrid& from, const COneGrid& to, cuDoubleComplex& res);
	bool CalculateConnectionFindZ(const COneGrid& from, const COneGrid& to, cuDoubleComplex& res);
	bool CalculateConnectionFindY(const COneGrid& from, const COneGrid& to, cuDoubleComplex& res);
	bool CalculateConnectionFindX(const COneGrid& from, const COneGrid& to, cuDoubleComplex& res);

	EAStarResult ConsiderOneWPath();
	EAStarResult ConsiderOneZWPath();
	EAStarResult ConsiderOneYZWPath();

	void FindNewWPath();
	void FindNewZWPair();
	void FindNewYZWPair();

	CSparseGrid4D m_Integrator;
	CCheck4DX m_CheckerX;
	CCheck4DY m_CheckerY;
	CCheck4DZ m_CheckerZ;
	CCheck4DW m_CheckerW;
	CGeneralGrid m_XGrid;
	CGeneralGrid m_YGrid;
	CGeneralGrid m_ZGrid;
	CGeneralGrid m_WGrid;

	std::unordered_set<CIntervalList> m_XIntervalDic;
	std::unordered_set<CIntervalList> m_YIntervalDic;
	std::unordered_set<CIntervalList> m_ZIntervalDic;

	std::vector<SPathInterval> m_WPathList;
	std::vector<SPathPairInterval> m_ZWPathList;
	std::vector<SPathPairInterval> m_YZWPathList;

	std::unordered_set<CPath> m_WPathDic;
	std::unordered_set<CPathPair> m_ZWPathDic;
	std::unordered_set<CPathPair> m_YZWPathDic;

	int m_iWPathListIdx;
	int m_iZWPathListIdx;
	int m_iYZWPathListIdx;

	CIntervalList m_ConsideringXInterval;
	CIntervalList m_ConsideringYInterval;
	CIntervalList m_ConsideringZInterval;
	CPath m_ConsideringWPath;
	CPathPair m_ConsideringZWPath;
	CPathPair m_ConsideringYZWPath;

	std::unordered_map<int, std::unordered_map<int, SIntegrateRes>> m_integrateDic;

	//result
	CPath m_XPath;
	CPath m_YPath;
	CPath m_ZPath;
	CPath m_WPath;

	bool m_bDone;
	int m_iIntegrationDic;
	integrand4d m_func;
	long long m_iTotalPossible;
	cuDoubleComplex m_res;
};
