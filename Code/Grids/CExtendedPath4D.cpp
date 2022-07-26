#include "CExtendedPath4D.h"
#include "CudaHelper.h"

bool CCheck4DW::IsConnected(const COneGrid& grid1, const COneGrid& grid2, cuDoubleComplex& v)
{
	return m_pOwner->CalculateConnectionFindW(grid1, grid2, v);
}

bool CCheck4DZ::IsConnected(const COneGrid& grid1, const COneGrid& grid2, cuDoubleComplex& v)
{
	return m_pOwner->CalculateConnectionFindZ(grid1, grid2, v);
}

bool CCheck4DY::IsConnected(const COneGrid& grid1, const COneGrid& grid2, cuDoubleComplex& v)
{
	return m_pOwner->CalculateConnectionFindY(grid1, grid2, v);
}

bool CCheck4DX::IsConnected(const COneGrid& grid1, const COneGrid& grid2, cuDoubleComplex& v)
{
	return m_pOwner->CalculateConnectionFindX(grid1, grid2, v);
}

CExtendedPath4D::CExtendedPath4D(int iWidth, int iHeight, int iEdge, const std::string& folderName, int iMaxOrder, double epsilon)
	: CLogger()
	, m_Integrator(folderName, iMaxOrder, epsilon)
	, m_CheckerX(this)
	, m_CheckerY(this)
	, m_CheckerZ(this)
	, m_CheckerW(this)
	, m_XGrid(iWidth, iHeight, iEdge)
	, m_YGrid(iWidth, iHeight, iEdge)
	, m_ZGrid(iWidth, iHeight, iEdge)
	, m_WGrid(iWidth, iHeight, iEdge)
	, m_iWPathListIdx(0)
	, m_iZWPathListIdx(0)
	, m_iYZWPathListIdx(0)
	, m_ConsideringXInterval()
	, m_ConsideringYInterval()
	, m_ConsideringZInterval()
	, m_ConsideringWPath()
	, m_ConsideringZWPath()
	, m_ConsideringYZWPath()
	, m_bDone(false)
	, m_iIntegrationDic(0)
	, m_iTotalPossible(0)
{
	int gridSize = iWidth * iHeight;
	m_iTotalPossible = ((iWidth - 1) * iHeight + iWidth * (iHeight - 1));
	m_iTotalPossible = m_iTotalPossible * m_iTotalPossible * m_iTotalPossible * m_iTotalPossible;

	m_XGrid.SetHash(gridSize * gridSize * gridSize, gridSize * gridSize);
	m_YGrid.SetHash(gridSize, 1);
	m_ZGrid.SetHash(gridSize * gridSize * gridSize, gridSize * gridSize);
	m_WGrid.SetHash(gridSize, 1);

	m_WPathList.reserve(_kDefaultListCapEP4D);
	m_ZWPathList.reserve(_kDefaultListCapEP4D);
	m_YZWPathList.reserve(_kDefaultListCapEP4D);

	m_func = nullptr;
	m_res = make_cuDoubleComplex(0.0, 0.0);
}

SIntegrateRes CExtendedPath4D::Integrate(integrand4d integrand)
{
	//=========== initial ============
	m_func = integrand;

	m_iWPathListIdx = 0;
	m_iZWPathListIdx = 0;
	m_iYZWPathListIdx = 0;

	m_ConsideringXInterval.m_bActive = false;
	m_ConsideringYInterval.m_bActive = false;
	m_ConsideringZInterval.m_bActive = false;
	m_ConsideringWPath.m_bActive = false;
	m_ConsideringZWPath.m_bActive = false;
	m_ConsideringYZWPath.m_bActive = false;

	m_XIntervalDic.clear();
	m_YIntervalDic.clear();
	m_ZIntervalDic.clear();

	m_WPathList.clear();
	m_ZWPathList.clear();
	m_YZWPathList.clear();

	m_WPathDic.clear();
	m_ZWPathDic.clear();
	m_YZWPathDic.clear();

	m_integrateDic.clear();

	m_bDone = false;
	m_iIntegrationDic = 0;

	FindNewYZWPair();
	while (m_iYZWPathListIdx < m_YZWPathList.size())
	{
		EAStarResult eres = ConsiderOneYZWPath();
		if (EAStarResult::Finished == eres)
		{
			return SIntegrateRes(true, m_res);
		}
	}
	return SIntegrateRes(false, make_cuDoubleComplex(0.0, 0.0));
}

bool CExtendedPath4D::IntegrateHyperCubic(
	const COneGrid& nodeX1, const COneGrid& nodeX2,
	const COneGrid& nodeY1, const COneGrid& nodeY2,
	const COneGrid& nodeZ1, const COneGrid& nodeZ2,
	const COneGrid& nodeW1, const COneGrid& nodeW2,
	cuDoubleComplex& res)
{
	int key1;
	double finalProd;
	if (nodeX1.m_iIdx > nodeX2.m_iIdx)
	{
		if (nodeY1.m_iIdx > nodeY2.m_iIdx)
		{
			finalProd = 1.0;
			key1 = nodeX1.m_idx1 + nodeX2.m_idx2 + nodeY1.m_idx1 + nodeY2.m_idx2;
		}
		else
		{
			finalProd = -1.0;
			key1 = nodeX1.m_idx1 + nodeX2.m_idx2 + nodeY1.m_idx2 + nodeY2.m_idx1;
		}
	}
	else
	{
		if (nodeY1.m_iIdx > nodeY2.m_iIdx)
		{
			finalProd = -1.0;
			key1 = nodeX1.m_idx2 + nodeX2.m_idx1 + nodeY1.m_idx1 + nodeY2.m_idx2;
		}
		else
		{
			finalProd = 1.0;
			key1 = nodeX1.m_idx2 + nodeX2.m_idx1 + nodeY1.m_idx2 + nodeY2.m_idx1;
		}
	}
	int key2;
	if (nodeZ1.m_iIdx > nodeZ2.m_iIdx)
	{
		if (nodeW1.m_iIdx > nodeW2.m_iIdx)
		{
			key2 = nodeZ1.m_idx1 + nodeZ2.m_idx2 + nodeW1.m_idx1 + nodeW2.m_idx2;
		}
		else
		{
			finalProd = -finalProd;
			key2 = nodeZ1.m_idx1 + nodeZ2.m_idx2 + nodeW1.m_idx2 + nodeW2.m_idx1;
		}
	}
	else
	{
		if (nodeW1.m_iIdx > nodeW2.m_iIdx)
		{
			finalProd = -finalProd;
			key2 = nodeZ1.m_idx2 + nodeZ2.m_idx1 + nodeW1.m_idx1 + nodeW2.m_idx2;
		}
		else
		{
			key2 = nodeZ1.m_idx2 + nodeZ2.m_idx1 + nodeW1.m_idx2 + nodeW2.m_idx1;
		}
	}

	std::unordered_map<int, std::unordered_map<int, SIntegrateRes>>::iterator it = m_integrateDic.find(key1);

	if (it == m_integrateDic.end())
	{
		cuDoubleComplex intres;
		bool bDone = m_Integrator.Integrate(m_func, 
			nodeX1.m_v, nodeX2.m_v,
			nodeY1.m_v, nodeY2.m_v, 
			nodeZ1.m_v, nodeZ2.m_v, 
			nodeW1.m_v, nodeW2.m_v, 
			intres);

		std::unordered_map<int, SIntegrateRes> newDic;
		newDic.insert(
			std::unordered_map<int, SIntegrateRes>::
			value_type(key2, 
				SIntegrateRes(bDone, cuCmulcr(intres, finalProd))
			));
		m_integrateDic.insert(std::unordered_map<int, std::unordered_map<int, SIntegrateRes>>::
			value_type(key1, newDic));
		res = intres;
		++m_iIntegrationDic;
		MyLogParanoiac("--dic %d / %llu\n", m_iIntegrationDic, m_iTotalPossible);
		return bDone;
	}

	std::unordered_map<int, SIntegrateRes>::iterator it2 = it->second.find(key2);
	if (it2 == it->second.end())
	{
		cuDoubleComplex intres;
		bool bDone = m_Integrator.Integrate(m_func,
			nodeX1.m_v, nodeX2.m_v,
			nodeY1.m_v, nodeY2.m_v,
			nodeZ1.m_v, nodeZ2.m_v,
			nodeW1.m_v, nodeW2.m_v,
			intres);

		it->second.insert(
			std::unordered_map<int, SIntegrateRes>::
			value_type(key2,
				SIntegrateRes(bDone, cuCmulcr(intres, finalProd))
			));

		res = intres;
		++m_iIntegrationDic;
		MyLogParanoiac("--dic %d / %llu\n", m_iIntegrationDic, m_iTotalPossible);
		return bDone;
	}

	res = cuCmulcr(it2->second.m_v, finalProd);
	return it2->second.m_bDone;
}

void CExtendedPath4D::AddYZWPath(const CPath& YPath, const CPath& ZPath, const CPath& WPath, const CIntervalList& xInterval)
{
	const std::vector<CPath> pathlist = {YPath, ZPath, WPath};
	CPathPair pathpair(pathlist);
	if (m_YZWPathDic.find(pathpair) == m_YZWPathDic.end())
	{
		m_YZWPathDic.insert(pathpair);
		SPathPairInterval newpathpair(pathpair, xInterval);
		m_YZWPathList.push_back(newpathpair);
	}
}

void CExtendedPath4D::AddZWPath(const CPath& ZPath, const CPath& WPath, const CIntervalList& yInterval)
{
	const std::vector<CPath> pathlist = { ZPath, WPath };
	CPathPair pathpair(pathlist);
	if (m_ZWPathDic.find(pathpair) == m_ZWPathDic.end())
	{
		m_ZWPathDic.insert(pathpair);
		SPathPairInterval newpathpair(pathpair, yInterval);
		m_ZWPathList.push_back(newpathpair);
	}
}

void CExtendedPath4D::AddWPath(const CPath& WPath, const CIntervalList& zInterval)
{
	if (m_WPathDic.find(WPath) == m_WPathDic.end())
	{
		m_WPathDic.insert(WPath);
		SPathInterval newpathpair(WPath, zInterval);
		m_WPathList.push_back(newpathpair);
	}
}

bool CExtendedPath4D::CalculateConnectionFindW(const COneGrid& from, const COneGrid& to, cuDoubleComplex& res)
{
	res = make_cuDoubleComplex(0.0, 0.0);
	size_t consideringX = 0;
	if (m_ConsideringXInterval.m_bActive)
	{
		consideringX = m_ConsideringXInterval.m_intervals.size();
	}
	size_t consideringY = 0;
	if (m_ConsideringYInterval.m_bActive)
	{
		consideringY = m_ConsideringYInterval.m_intervals.size();
	}
	size_t consideringZ = 0;
	if (m_ConsideringZInterval.m_bActive)
	{
		consideringZ = m_ConsideringZInterval.m_intervals.size();
	}
	cuDoubleComplex v;
	for (size_t x = 0; x < consideringX; ++x)
	{
		const COneGrid* x1 = m_ConsideringXInterval.m_intervals[x].m_pNode1;
		const COneGrid* x2 = m_ConsideringXInterval.m_intervals[x].m_pNode2;
		for (size_t y = 0; y < consideringY; ++y)
		{
			const COneGrid* y1 = m_ConsideringYInterval.m_intervals[y].m_pNode1;
			const COneGrid* y2 = m_ConsideringYInterval.m_intervals[y].m_pNode2;
			for (size_t z = 0; z < consideringZ; ++z)
			{
				const COneGrid* z1 = m_ConsideringZInterval.m_intervals[z].m_pNode1;
				const COneGrid* z2 = m_ConsideringZInterval.m_intervals[z].m_pNode2;
				bool bHasV = IntegrateHyperCubic(*x1, *x2, *y1, *y2, *z1, *z2, from, to, v);
				if (!bHasV)
				{
					return false;
				}
			}
		}
	}
	return true;
}

bool CExtendedPath4D::CalculateConnectionFindZ(const COneGrid& from, const COneGrid& to, cuDoubleComplex& res)
{
	res = make_cuDoubleComplex(0.0, 0.0);
	size_t consideringX = 0;
	if (m_ConsideringXInterval.m_bActive)
	{
		consideringX = m_ConsideringXInterval.m_intervals.size();
	}
	size_t consideringY = 0;
	if (m_ConsideringYInterval.m_bActive)
	{
		consideringY = m_ConsideringYInterval.m_intervals.size();
	}
	const size_t consideringW = m_ConsideringWPath.m_nodes.size();
	cuDoubleComplex v;
	for (size_t x = 0; x < consideringX; ++x)
	{
		const COneGrid* x1 = m_ConsideringXInterval.m_intervals[x].m_pNode1;
		const COneGrid* x2 = m_ConsideringXInterval.m_intervals[x].m_pNode2;
		for (size_t y = 0; y < consideringY; ++y)
		{
			const COneGrid* y1 = m_ConsideringYInterval.m_intervals[y].m_pNode1;
			const COneGrid* y2 = m_ConsideringYInterval.m_intervals[y].m_pNode2;
			for (size_t w = 1; w < consideringW; ++w)
			{
				const COneGrid* w1 = m_ConsideringWPath.m_nodes[w - 1];
				const COneGrid* w2 = m_ConsideringWPath.m_nodes[w];
				bool bHasV = IntegrateHyperCubic(*x1, *x2, *y1, *y2, from, to, *w1, *w2, v);
				if (!bHasV)
				{
					return false;
				}
			}
		}
	}
	return true;
}

bool CExtendedPath4D::CalculateConnectionFindY(const COneGrid& from, const COneGrid& to, cuDoubleComplex& res)
{
	res = make_cuDoubleComplex(0.0, 0.0);
	size_t consideringX = 0;
	if (m_ConsideringXInterval.m_bActive)
	{
		consideringX = m_ConsideringXInterval.m_intervals.size();
	}
	const size_t consideringZ = m_ConsideringZWPath.m_Pathes[0].m_nodes.size();
	const size_t consideringW = m_ConsideringZWPath.m_Pathes[1].m_nodes.size();
	cuDoubleComplex v;
	for (size_t x = 0; x < consideringX; ++x)
	{
		const COneGrid* x1 = m_ConsideringXInterval.m_intervals[x].m_pNode1;
		const COneGrid* x2 = m_ConsideringXInterval.m_intervals[x].m_pNode2;
		for (size_t z = 1; z < consideringZ; ++z)
		{
			const COneGrid* z1 = m_ConsideringZWPath.m_Pathes[0].m_nodes[z -1];
			const COneGrid* z2 = m_ConsideringZWPath.m_Pathes[0].m_nodes[z];
			for (size_t w = 1; w < consideringW; ++w)
			{
				const COneGrid* w1 = m_ConsideringZWPath.m_Pathes[1].m_nodes[w - 1];
				const COneGrid* w2 = m_ConsideringZWPath.m_Pathes[1].m_nodes[w];
				bool bHasV = IntegrateHyperCubic(*x1, *x2, from, to, *z1, *z2, *w1, *w2, v);
				if (!bHasV)
				{
					return false;
				}
			}
		}
	}
	return true;
}

bool CExtendedPath4D::CalculateConnectionFindX(const COneGrid& from, const COneGrid& to, cuDoubleComplex& res)
{
	res = make_cuDoubleComplex(0.0, 0.0);

	const size_t consideringY = m_ConsideringYZWPath.m_Pathes[0].m_nodes.size();
	const size_t consideringZ = m_ConsideringYZWPath.m_Pathes[1].m_nodes.size();
	const size_t consideringW = m_ConsideringYZWPath.m_Pathes[2].m_nodes.size();
	cuDoubleComplex v;
	for (size_t y = 1; y < consideringY; ++y)
	{
		const COneGrid* y1 = m_ConsideringYZWPath.m_Pathes[0].m_nodes[y - 1];
		const COneGrid* y2 = m_ConsideringYZWPath.m_Pathes[0].m_nodes[y];
		for (size_t z = 1; z < consideringZ; ++z)
		{
			const COneGrid* z1 = m_ConsideringYZWPath.m_Pathes[1].m_nodes[z - 1];
			const COneGrid* z2 = m_ConsideringYZWPath.m_Pathes[1].m_nodes[z];
			for (size_t w = 1; w < consideringW; ++w)
			{
				const COneGrid* w1 = m_ConsideringYZWPath.m_Pathes[2].m_nodes[w - 1];
				const COneGrid* w2 = m_ConsideringYZWPath.m_Pathes[2].m_nodes[w];
				bool bHasV = IntegrateHyperCubic(from, to, *y1, *y2, *z1, *z2, *w1, *w2, v);
				if (!bHasV)
				{
					return false;
				}
				res = cuCadd(res, v);
			}
		}
	}
	return true;
}

EAStarResult CExtendedPath4D::ConsiderOneWPath()
{
	size_t wpathlen = m_WPathList.size();
	if (m_iWPathListIdx >= wpathlen)
	{
		return EAStarResult::Failed;
	}

	const SPathInterval considering = m_WPathList[m_iWPathListIdx];
	m_ConsideringWPath = m_WPathList[m_iWPathListIdx].m_p;
	++m_iWPathListIdx;

	MyLogDetailed("        W List: %d / %d\n", m_iWPathListIdx, wpathlen);
	m_ZGrid.ResetGrid();
	cuDoubleComplex v;
	std::vector<COneGrid*> path;
	std::vector<cuDoubleComplex> points;
	path.reserve(_kDefaultCapacity);
	points.reserve(_kDefaultCapacity);
	EAStarResult res = m_ZGrid.FindPath(&m_CheckerZ, path, points, v);
	if (res != EAStarResult::Finished)
	{
		std::vector<CInterval> intervals = m_ZGrid.GetAllDisconnectIntervals();
		size_t intervalsize = intervals.size();
		for (size_t i = 0; i < intervalsize; ++i)
		{
			const CIntervalList& nowZ = considering.m_inter;
			if (!nowZ.m_bActive)
			{
				std::vector<CInterval> newintervals({ intervals[i] });
				CIntervalList newintlist(newintervals);

				if (m_ZIntervalDic.find(newintlist) == m_ZIntervalDic.end())
				{
					m_ZIntervalDic.insert(newintlist);
					m_ConsideringZInterval = newintlist;
					FindNewWPath();
				}
			}
			else
			{
				CIntervalList newintlist(nowZ.m_intervals, intervals[i]);
				if (m_ZIntervalDic.find(newintlist) == m_ZIntervalDic.end())
				{
					m_ZIntervalDic.insert(newintlist);
					m_ConsideringZInterval = newintlist;
					FindNewWPath();
				}
			}
		}
	}
	else
	{
		AddZWPath(CPath(path, &points), m_ConsideringWPath, m_ConsideringYInterval);
	}
	return res;
}

EAStarResult CExtendedPath4D::ConsiderOneZWPath()
{
	size_t zwpathlen = m_ZWPathList.size();
	if (m_iZWPathListIdx >= zwpathlen)
	{
		return EAStarResult::Failed;
	}

	const SPathPairInterval considering = m_ZWPathList[m_iZWPathListIdx];
	m_ConsideringZWPath = m_ZWPathList[m_iZWPathListIdx].m_p;
	++m_iZWPathListIdx;

	MyLogDetailed("    ZW List: %d / %d\n", m_iZWPathListIdx, zwpathlen);
	m_YGrid.ResetGrid();
	cuDoubleComplex v;
	std::vector<COneGrid*> path;
	std::vector<cuDoubleComplex> points;
	path.reserve(_kDefaultCapacity);
	points.reserve(_kDefaultCapacity);
	EAStarResult res = m_YGrid.FindPath(&m_CheckerY, path, points, v);
	if (res != EAStarResult::Finished)
	{
		std::vector<CInterval> intervals = m_YGrid.GetAllDisconnectIntervals();
		size_t intervalsize = intervals.size();
		for (size_t i = 0; i < intervalsize; ++i)
		{
			const CIntervalList& nowY = considering.m_inter;
			if (!nowY.m_bActive)
			{
				std::vector<CInterval> newintervals({ intervals[i] });
				CIntervalList newintlist(newintervals);

				if (m_YIntervalDic.find(newintlist) == m_YIntervalDic.end())
				{
					m_YIntervalDic.insert(newintlist);
					m_ConsideringYInterval = newintlist;
					FindNewZWPair();
				}
			}
			else
			{
				CIntervalList newintlist(nowY.m_intervals, intervals[i]);
				if (m_YIntervalDic.find(newintlist) == m_YIntervalDic.end())
				{
					m_YIntervalDic.insert(newintlist);
					m_ConsideringYInterval = newintlist;
					FindNewZWPair();
				}
			}
		}
	}
	else
	{
		AddYZWPath(CPath(path, &points), m_ConsideringZWPath.m_Pathes[0], m_ConsideringZWPath.m_Pathes[1], m_ConsideringXInterval);
	}
	return res;
}

EAStarResult CExtendedPath4D::ConsiderOneYZWPath()
{
	size_t yzwpathlen = m_YZWPathList.size();
	if (m_iYZWPathListIdx >= yzwpathlen)
	{
		return EAStarResult::Failed;
	}

	const SPathPairInterval considering = m_YZWPathList[m_iYZWPathListIdx];
	m_ConsideringYZWPath = m_YZWPathList[m_iYZWPathListIdx].m_p;
	++m_iYZWPathListIdx;

	MyLogDetailed("YZW List: %d / %d\n", m_iYZWPathListIdx, yzwpathlen);
	m_XGrid.ResetGrid();
	cuDoubleComplex v;
	std::vector<COneGrid*> path;
	std::vector<cuDoubleComplex> points;
	path.reserve(_kDefaultCapacity);
	points.reserve(_kDefaultCapacity);
	EAStarResult res = m_XGrid.FindPath(&m_CheckerX, path, points, v);
	if (res != EAStarResult::Finished)
	{
		std::vector<CInterval> intervals = m_XGrid.GetAllDisconnectIntervals();
		size_t intervalsize = intervals.size();
		for (size_t i = 0; i < intervalsize; ++i)
		{
			const CIntervalList& nowX = considering.m_inter;
			if (!nowX.m_bActive)
			{
				std::vector<CInterval> newintervals({ intervals[i] });
				CIntervalList newintlist(newintervals);

				if (m_XIntervalDic.find(newintlist) == m_XIntervalDic.end())
				{
					m_XIntervalDic.insert(newintlist);
					m_ConsideringXInterval = newintlist;
					FindNewYZWPair();
				}
			}
			else
			{
				CIntervalList newintlist(nowX.m_intervals, intervals[i]);
				if (m_XIntervalDic.find(newintlist) == m_XIntervalDic.end())
				{
					m_XIntervalDic.insert(newintlist);
					m_ConsideringXInterval = newintlist;
					FindNewYZWPair();
				}
			}
		}
	}
	else
	{
		m_res = v;
		m_bDone = true;
		m_XPath = CPath(path, &points);
		m_YPath = m_ConsideringYZWPath.m_Pathes[0];
		m_ZPath = m_ConsideringYZWPath.m_Pathes[1];
		m_WPath = m_ConsideringYZWPath.m_Pathes[2];
	}
	return res;
}

void CExtendedPath4D::FindNewWPath()
{
	m_WGrid.ResetGrid();
	cuDoubleComplex v;
	std::vector<COneGrid*> path;
	std::vector<cuDoubleComplex> points;
	path.reserve(_kDefaultCapacity);
	points.reserve(_kDefaultCapacity);
	EAStarResult res = m_WGrid.FindPath(&m_CheckerW, path, points, v);
	if (EAStarResult::Finished == res)
	{
		AddWPath(CPath(path, &points), m_ConsideringZInterval);
	}
}

void CExtendedPath4D::FindNewZWPair()
{
	m_iWPathListIdx = 0;
	m_WPathList.clear();
	m_WPathDic.clear();
	m_ZIntervalDic.clear();
	m_ConsideringZInterval.m_bActive = false;
	FindNewWPath();
	while (m_iWPathListIdx < m_WPathList.size())
	{
		EAStarResult res = ConsiderOneWPath();
		if (EAStarResult::Finished == res)
		{
			return;
		}
	}
}

void CExtendedPath4D::FindNewYZWPair()
{
	m_iZWPathListIdx = 0;
	m_ZWPathList.clear();
	m_ZWPathDic.clear();
	m_YIntervalDic.clear();
	m_ConsideringYInterval.m_bActive = false;
	FindNewZWPair();
	while (m_iZWPathListIdx < m_ZWPathList.size())
	{
		EAStarResult res = ConsiderOneZWPath();
		if (EAStarResult::Finished == res)
		{
			return;
		}
	}
}

void CExtendedPath4D::PrintResoult(const std::string& sFunc)
{
	if (!m_bDone)
	{
		LogGeneral("integration failed!\n");
	}

	std::string res = "";
	std::string listPlotX = "wr={";
	std::string listPlotY = "wi={";
	for (int i = 0; i < m_WPath.m_points.size(); ++i)
	{
		listPlotX = listPlotX + 
			((0 != i) ? string_format(", %2.12f", m_WPath.m_points[i].x) : string_format("%2.12f", m_WPath.m_points[i].x));
		listPlotY = listPlotY + 
			((0 != i) ? string_format(", %2.12f", m_WPath.m_points[i].y) : string_format("%2.12f", m_WPath.m_points[i].y));
	}
	listPlotX = listPlotX + "};\n";
	listPlotY = listPlotY + "};\n";
	res = res + listPlotX + listPlotY;

	listPlotX = "zr={";
	listPlotY = "zi={";
	for (int i = 0; i < m_ZPath.m_points.size(); ++i)
	{
		listPlotX = listPlotX +
			((0 != i) ? string_format(", %2.12f", m_ZPath.m_points[i].x) : string_format("%2.12f", m_ZPath.m_points[i].x));
		listPlotY = listPlotY +
			((0 != i) ? string_format(", %2.12f", m_ZPath.m_points[i].y) : string_format("%2.12f", m_ZPath.m_points[i].y));
	}
	listPlotX = listPlotX + "};\n";
	listPlotY = listPlotY + "};\n";
	res = res + listPlotX + listPlotY;

	listPlotX = "yr={";
	listPlotY = "yi={";
	for (int i = 0; i < m_YPath.m_points.size(); ++i)
	{
		listPlotX = listPlotX +
			((0 != i) ? string_format(", %2.12f", m_YPath.m_points[i].x) : string_format("%2.12f", m_YPath.m_points[i].x));
		listPlotY = listPlotY +
			((0 != i) ? string_format(", %2.12f", m_YPath.m_points[i].y) : string_format("%2.12f", m_YPath.m_points[i].y));
	}
	listPlotX = listPlotX + "};\n";
	listPlotY = listPlotY + "};\n";
	res = res + listPlotX + listPlotY;

	listPlotX = "xr={";
	listPlotY = "xi={";
	for (int i = 0; i < m_XPath.m_points.size(); ++i)
	{
		listPlotX = listPlotX +
			((0 != i) ? string_format(", %2.12f", m_XPath.m_points[i].x) : string_format("%2.12f", m_XPath.m_points[i].x));
		listPlotY = listPlotY +
			((0 != i) ? string_format(", %2.12f", m_XPath.m_points[i].y) : string_format("%2.12f", m_XPath.m_points[i].y));
	}
	listPlotX = listPlotX + "};\n";
	listPlotY = listPlotY + "};\n";
	res = res + listPlotX + listPlotY;
	res = res + string_format("Print[\"expecting: %f %s %f I \"];\n", m_res.x, m_res.y > 0.0 ? "+" : "-", abs(m_res.y));

	res = res + "res = Sum[NIntegrate[f[x, y, z, w],\n";
	res = res + "{x, xr[[u]] + xi[[u]] I, xr[[u + 1]] + xi[[u + 1]] I},\n";
	res = res + "{y, yr[[v]] + yi[[v]] I, yr[[v + 1]] + yi[[v + 1]] I},\n";
	res = res + "{z, zr[[s]] + zi[[s]] I, zr[[s + 1]] + zi[[s + 1]] I},\n";
	res = res + "{w, wr[[t]] + wi[[t]] I, wr[[t + 1]] + wi[[t + 1]] I}],\n";
	res = res + "{u, 1, Length[xr] - 1 }, {v, 1, Length[yr] - 1}, {s, 1, Length[zr] - 1}, {t, 1, Length[wr] - 1}]\n";

	res = res + "ListLinePlot[Transpose[{xr, xi}], AxesLabel -> {\"Re[x]\", \"Im[x]\"}, PlotRange -> All]\n";
	res = res + "ListLinePlot[Transpose[{yr, yi}], AxesLabel -> {\"Re[y]\", \"Im[y]\"}, PlotRange -> All]\n";
	res = res + "ListLinePlot[Transpose[{zr, zi}], AxesLabel -> {\"Re[z]\", \"Im[z]\"}, PlotRange -> All]\n";
	res = res + "ListLinePlot[Transpose[{wr, wi}], AxesLabel -> {\"Re[w]\", \"Im[w]\"}, PlotRange -> All]\n";
	res = "\n(* =========== Copy these to Mathematica ========== *)\n\n" + std::string("f[x_,y_,z_,w_]:=") + sFunc + ";\n" + res;

	LogGeneral(res.c_str());
}
