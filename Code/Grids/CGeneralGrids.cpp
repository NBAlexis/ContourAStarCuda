#include "CGeneralGrids.h"

CGeneralGrid::CGeneralGrid(int width, int height, int edge)
	: m_width(width)
	, m_height(height)
	, m_edge(edge)
	, m_startX(0)
	, m_startY(0)
	, m_midY(0)
	, m_endX(0)
	, m_endY(0)
	, m_gridsize(0)
	, m_sep(0.0f)
	, m_eResult(EAStarResult::NotStarted)
{
	m_midY = height / 2;
	m_sep = 2.0f / static_cast<float>(width - 1 - 2 * edge);
	m_startX = edge;
	m_startY = m_midY;
	m_endX = width - 1 - edge;
	m_endY = m_midY;

	CreateGrid();

	m_gridsize = m_gridList.size();
}

void CGeneralGrid::CreateGrid()
{
	m_gridArray.clear();
	for (int i = 0; i < m_width; ++i)
	{
		std::vector<COneGrid> row;
		for (int j = 0; j < m_height; ++j)
		{
			row.push_back(COneGrid(this, i, j, i + j * m_width,
				make_cuDoubleComplex(-1.0 + (i - m_edge) * m_sep, (j - m_midY) * m_sep)
			));
		}
		m_gridArray.push_back(row);
	}

	static int xOffsets[4] = { 0, -1, 1, 0 };
	static int yOffsets[4] = { 1, 0, 0, -1 };
	for (int x = 0; x < m_width; ++x)
	{
		for (int y = 0; y < m_height; ++y)
		{
			m_gridArray[x][y].SetHn(m_endX, m_endY);
			m_gridList.push_back(&(m_gridArray[x][y]));
			for (int d = 0; d < (int)EGridDir::DMax; ++d)
			{
				int xoffset = x + xOffsets[d];
				int yoffset = y + yOffsets[d];
				if (xoffset >= 0 && xoffset < m_width && yoffset >= 0 && yoffset < m_height)
				{
					m_gridArray[x][y].SetNeighbour((EGridDir)d, &(m_gridArray[xoffset][yoffset]));
				}
				else
				{
					m_gridArray[x][y].SetNeighbour((EGridDir)d, nullptr);
				}
			}
		}
	}
	m_gridArray[m_startX][m_startY].m_bIsStart = true;
	m_gridArray[m_endX][m_endY].m_bIsTarget = true;
}

void CGeneralGrid::ResetGridToStart(bool bRecheckNeighbour)
{
	for (int x = 0; x < m_width; ++x)
	{
		for (int y = 0; y < m_height; ++y)
		{
			m_gridArray[x][y].SetHn(m_endX, m_endY);
			m_gridArray[x][y].m_pParent = nullptr;
			m_gridArray[x][y].m_eParentDir = EGridDir::DMax;
			m_gridArray[x][y].m_fn = 0.0f;
			m_gridArray[x][y].m_state = EGridState::NotDecide;
			if (bRecheckNeighbour)
			{
				for (int d = 0; d < (int)EGridDir::DMax; ++d)
				{
					if (nullptr == m_gridArray[x][y].m_neighbourNodes[d])
					{
						m_gridArray[x][y].m_neighbourState[d] = EGridNeighbour::NotConnected;
					}
					else
					{
						m_gridArray[x][y].m_neighbourState[d] = EGridNeighbour::Unknown;
					}
				}
			}
		}
	}
	m_gridArray[m_startX][m_startY].m_state = EGridState::OpenList;
	m_gridArray[m_startX][m_startY].m_bIsStart = true;
	m_gridArray[m_endX][m_endY].m_bIsTarget = true;
	m_eResult = EAStarResult::NotStarted;
}

void CGeneralGrid::CalculateConnectionStep(IConnectionChecker* pCheck, EGridDir d, int x, int y)
{
	if (EGridNeighbour::Connected == m_gridArray[x][y].m_neighbourState[(int)d])
	{
		return;
	}
	if (EGridNeighbour::NotConnected == m_gridArray[x][y].m_neighbourState[(int)d])
	{
		return;
	}
	if (nullptr == m_gridArray[x][y].m_neighbourNodes[(int)d])
	{
		m_gridArray[x][y].m_neighbourState[(int)d] = EGridNeighbour::NotConnected;
		return;
	}

	int opposite = (int)EGridDir::Down - (int)d;
	cuDoubleComplex outV;
	bool bHasValue = pCheck->IsConnected(m_gridArray[x][y], *m_gridArray[x][y].m_neighbourNodes[(int)d], outV);
	if (bHasValue)
	{
		m_gridArray[x][y].m_neighbourState[(int)d] = EGridNeighbour::Connected;
		m_gridArray[x][y].m_neighbourV[(int)d] = outV;
		m_gridArray[x][y].m_neighbourNodes[(int)d]->m_neighbourState[opposite] = EGridNeighbour::Connected;
		m_gridArray[x][y].m_neighbourNodes[(int)d]->m_neighbourV[opposite] = make_cuDoubleComplex(-outV.x, -outV.y);
	}
	else
	{
		m_gridArray[x][y].m_neighbourState[(int)d] = EGridNeighbour::NotConnected;
		m_gridArray[x][y].m_neighbourNodes[(int)d]->m_neighbourState[opposite] = EGridNeighbour::NotConnected;
	}
}

std::vector<CInterval> CGeneralGrid::GetAllDisconnectIntervals() const
{
	std::vector<CInterval> ret;
	ret.reserve(_kDefaultCapacity);
	for (int i = 0; i < m_gridsize; ++i)
	{
		if (EGridState::ClosedList == m_gridList[i]->m_state)
		{
			for (int d = 0; d < (int)EGridDir::DMax; ++d)
			{
				if (nullptr != m_gridList[i]->m_neighbourNodes[d]
					&& EGridNeighbour::NotConnected == m_gridList[i]->m_neighbourState[d])
				{
					CInterval toAdd(m_gridList[i], m_gridList[i]->m_neighbourNodes[d], static_cast<unsigned int>(m_gridsize));
					std::vector<CInterval>::iterator it = std::find(ret.begin(), ret.end(), toAdd);
					if (ret.end() == it)
					{
						ret.push_back(toAdd);
					}
				}
			}
		}
	}
	return ret;
}

EAStarResult CGeneralGrid::OneStep(IConnectionChecker* pCheck)
{
	float fn = 1000000000.0f;
	COneGrid* pSmallest = nullptr;
	for (int i = 0; i < m_gridsize; ++i)
	{
		if (EGridState::OpenList == m_gridList[i]->m_state && m_gridList[i]->m_fn < fn)
		{
			fn = m_gridList[i]->m_fn;
			pSmallest = m_gridList[i];
		}
	}
	if (nullptr == pSmallest)
	{
		return EAStarResult::Failed;
	}
	if (pSmallest->m_bIsTarget)
	{
		m_eResult = EAStarResult::Finished;
		return EAStarResult::Finished;
	}
	for (int d = 0; d < (int)EGridDir::DMax; ++d)
	{
		if (EGridNeighbour::Unknown == pSmallest->m_neighbourState[d])
		{
			CalculateConnectionStep(pCheck, (EGridDir)d, pSmallest->m_iX, pSmallest->m_iY);
		}
		if (EGridNeighbour::Connected == pSmallest->m_neighbourState[d])
		{
			COneGrid* pNeighbour = pSmallest->m_neighbourNodes[d];
			if (EGridState::ClosedList != pNeighbour->m_state)
			{
				pNeighbour->UpdateFn(pSmallest, (EGridDir)d);
			}
		}
	}
	pSmallest->m_state = EGridState::ClosedList;
	return EAStarResult::NotStarted;
}

EAStarResult CGeneralGrid::FindPath(IConnectionChecker* pCheck, std::vector<COneGrid*>& path, std::vector<cuDoubleComplex>& points, cuDoubleComplex& v)
{
	ResetGrid();
	EAStarResult res = OneStep(pCheck);
	while (EAStarResult::NotStarted == res)
	{
		res = OneStep(pCheck);
	}
	if (EAStarResult::Finished == res)
	{
		v = make_cuDoubleComplex(0.0, 0.0);
		COneGrid* pTarget = &(m_gridArray[m_endX][m_endY]);
		while (nullptr != pTarget && !pTarget->m_bIsStart)
		{
			v = cuCadd(v, pTarget->m_neighbourV[(int)pTarget->m_eParentDir]);
			pTarget = pTarget->m_pParent;
		}
		v.x = -v.x;
		v.y = -v.y;
		//gather path
		EGridDir eLastDir = EGridDir::DMax;
		pTarget = &(m_gridArray[m_endX][m_endY]);
		path.clear();
		points.clear();
		while (nullptr != pTarget && !pTarget->m_bIsStart)
		{
			if (eLastDir != pTarget->m_eParentDir)
			{
				points.insert(points.begin(), pTarget->m_v);
				eLastDir = pTarget->m_eParentDir;
			}
			path.insert(path.begin(), pTarget);
			pTarget = pTarget->m_pParent;
		}
		if (nullptr != pTarget)
		{
			path.insert(path.begin(), pTarget);
			points.insert(points.begin(), pTarget->m_v);
		}

		return EAStarResult::Finished;
	}
	v = make_cuDoubleComplex(0.0, 0.0);
	path.clear();
	points.clear();
	return EAStarResult::Failed;
}

