#pragma once

#include "cuComplex.h"
#include <vector>

enum class EGridState
{
	NotDecide,
	OpenList,
	ClosedList,
};

enum class EGridDir
{
	Up = 0,
	Left = 1,
	Right = 2,
	Down = 3,
	DMax = 4,
};

enum class EGridNeighbour
{
	NotConnected,
	Unknown,
	Connected,
};

enum class EAStarResult
{
	NotStarted,
	Finished,
	Failed,
};

const int _kDefaultCapacity = 128;

class COneGrid
{
public:

	COneGrid()
		: m_pOwner(nullptr)
		, m_iX(0)
		, m_iY(0)
		, m_iIdx(0)
		, m_state(EGridState::NotDecide)
		, m_fn(0.0f)
		, m_hn(0.0f)
		, m_gn(0.0f)
		, m_bIsStart(false)
		, m_bIsTarget(false)
		, m_idx1(0)
		, m_idx2(0)
		, m_pParent(nullptr)
		, m_eParentDir(EGridDir::DMax)
		, _unused1(0)
		, _unused2(0)
		, _unused3(0)
	{
		m_v = make_cuDoubleComplex(0.0, 0.0);

		m_neighbourV[0] = make_cuDoubleComplex(0, 0);
		m_neighbourV[1] = make_cuDoubleComplex(0, 0);
		m_neighbourV[2] = make_cuDoubleComplex(0, 0);
		m_neighbourV[3] = make_cuDoubleComplex(0, 0);
		m_neighbourNodes[0] = nullptr;
		m_neighbourNodes[1] = nullptr;
		m_neighbourNodes[2] = nullptr;
		m_neighbourNodes[3] = nullptr;
		m_neighbourState[0] = EGridNeighbour::Unknown;
		m_neighbourState[1] = EGridNeighbour::Unknown;
		m_neighbourState[2] = EGridNeighbour::Unknown;
		m_neighbourState[3] = EGridNeighbour::Unknown;
	}

	COneGrid(class CGeneralGrid* pOwner, int x, int y, unsigned int idx, const cuDoubleComplex& v)
		: m_pOwner(pOwner)
		, m_iX(x)
		, m_iY(y)
		, m_iIdx(idx)
		, m_state(EGridState::NotDecide)
		, m_fn(0.0f)
		, m_hn(0.0f)
		, m_gn(0.0f)
		, m_bIsStart(false)
		, m_bIsTarget(false)
		, m_idx1(idx)
		, m_idx2(idx)
		, m_pParent(nullptr)
		, m_eParentDir(EGridDir::DMax)
		, _unused1(0)
		, _unused2(0)
		, _unused3(0)
	{

		m_v = make_cuDoubleComplex(v.x, v.y);

		m_neighbourV[0] = make_cuDoubleComplex(0, 0);
		m_neighbourV[1] = make_cuDoubleComplex(0, 0); 
		m_neighbourV[2] = make_cuDoubleComplex(0, 0); 
		m_neighbourV[3] = make_cuDoubleComplex(0, 0); 
		m_neighbourNodes[0] = nullptr;
		m_neighbourNodes[1] = nullptr;
		m_neighbourNodes[2] = nullptr;
		m_neighbourNodes[3] = nullptr;
		m_neighbourState[0] = EGridNeighbour::Unknown;
		m_neighbourState[1] = EGridNeighbour::Unknown;
		m_neighbourState[2] = EGridNeighbour::Unknown;
		m_neighbourState[3] = EGridNeighbour::Unknown;
	}

	COneGrid(const COneGrid& other)
		: m_pOwner(other.m_pOwner)
		, m_iX(other.m_iX)
		, m_iY(other.m_iY)
		, m_iIdx(other.m_iIdx)
		, m_state(other.m_state)
		, m_fn(other.m_fn)
		, m_hn(other.m_hn)
		, m_gn(other.m_gn)
		, m_bIsStart(other.m_bIsStart)
		, m_bIsTarget(other.m_bIsTarget)
		, m_idx1(other.m_idx1)
		, m_idx2(other.m_idx2)
		, m_pParent(other.m_pParent)
		, m_eParentDir(other.m_eParentDir)
		, _unused1(0)
		, _unused2(0)
		, _unused3(0)
	{
		m_v = make_cuDoubleComplex(other.m_v.x, other.m_v.y);

		memcpy(m_neighbourV, other.m_neighbourV, sizeof(cuDoubleComplex) * 4);
		memcpy(m_neighbourNodes, other.m_neighbourNodes, sizeof(COneGrid*) * 4);
		memcpy(m_neighbourState, other.m_neighbourState, sizeof(EGridNeighbour) * 4);
	}

	inline bool operator==(const COneGrid& other) const
	{
		return m_iIdx == other.m_iIdx;
	}

	inline bool operator!=(const COneGrid& other) const
	{
		return m_iIdx != other.m_iIdx;
	}

	void SetNeighbour(EGridDir direction, COneGrid* neighbour)
	{
		if (nullptr == neighbour)
		{
			m_neighbourState[(int)direction] = EGridNeighbour::NotConnected;
			return;
		}
		m_neighbourNodes[(int)direction] = neighbour;
	}

	void SetHn(int targetX, int targetY)
	{
		m_hn = static_cast<float>(abs(targetX - m_iX) + abs(targetY - m_iY));
	}

	void UpdateFn(COneGrid* pParent, EGridDir direction)
	{
		EGridDir opppsite = (EGridDir)((int)EGridDir::Down - (int)direction);
		if (EGridState::NotDecide == m_state)
		{
			m_pParent = pParent;
			m_eParentDir = opppsite;
			m_gn = 1.0f + pParent->m_gn;
			m_fn = m_hn + m_gn;
			m_state = EGridState::OpenList;
			return;
		}

		float newgn = 1.0f + pParent->m_gn;
		if (newgn < m_gn)
		{
			m_gn = newgn;
			m_pParent = pParent;
			m_eParentDir = opppsite;
			m_fn = m_hn + m_gn;
		}
	}

	class CGeneralGrid* m_pOwner;
	int m_iX;
	int m_iY;
	unsigned int m_iIdx;
	COneGrid* m_neighbourNodes[4];
	EGridNeighbour m_neighbourState[4];
	EGridState m_state;

	float m_fn;
	float m_hn;
	float m_gn;
	bool m_bIsStart;
	bool m_bIsTarget;
	int m_idx1;
	int m_idx2;

	COneGrid* m_pParent;
	EGridDir m_eParentDir;

	int _unused1;
	int _unused2; 
	int _unused3;

	cuDoubleComplex m_v;
	cuDoubleComplex m_neighbourV[4];
};

class CPath
{
public:

	CPath()
		: m_bActive(false)
		, m_iHash(0)
	{
		m_nodes.reserve(_kDefaultCapacity);
		m_points.reserve(_kDefaultCapacity);
	}

	CPath(const std::vector<COneGrid*>& nodes, const std::vector<cuDoubleComplex>* points = nullptr)
		: m_bActive(true)
		, m_iHash(0)
	{
		size_t iHash = 0;
		size_t nodeSize = nodes.size();
		for (int i = 0; i < nodeSize; ++i)
		{
			iHash += nodes[i]->m_iIdx * 1000000;
		}
		iHash += 1000000 * nodeSize;
		m_nodes = nodes;
		m_iHash = iHash;
		if (nullptr != points)
		{
			m_points = *points;
		}
	}

	inline bool operator==(const CPath& other) const
	{
		if (m_iHash != other.m_iHash)
		{
			return false;
		}
		size_t nodeSize = m_nodes.size();
		if (nodeSize != other.m_nodes.size())
		{
			return false;
		}
		for (int i = 0; i < nodeSize; ++i)
		{
			if (m_nodes[i]->m_iIdx != other.m_nodes[i]->m_iIdx)
			{
				return false;
			}
		}

		return true;
	}

	inline bool operator!=(const CPath& other) const
	{
		return !(*this == other);
	}

	std::vector<COneGrid*> m_nodes;
	std::vector<cuDoubleComplex> m_points;
	bool m_bActive;
	size_t m_iHash;
};

template<> struct std::hash<CPath>
{
	std::size_t operator()(const CPath& path) const noexcept
	{
		return path.m_iHash;
	}
};

class CPathPair
{
public:
	CPathPair()
		: m_bActive(false)
		, m_iHash(0)
	{
		m_Pathes.reserve(_kDefaultCapacity);
	}

	CPathPair(const std::vector<CPath>& pathes)
		: m_bActive(true)
		, m_iHash(0)
	{
		m_iHash = 0;
		m_Pathes = pathes;
		size_t nodeSize = pathes.size();
		for (size_t i = 0; i < nodeSize; ++i)
		{
			m_iHash += pathes[i].m_iHash;
		}
	}

	inline bool operator==(const CPathPair& other) const
	{
		if (m_iHash != other.m_iHash)
		{
			return false;
		}
		size_t nodeSize = m_Pathes.size();
		if (nodeSize != other.m_Pathes.size())
		{
			return false;
		}
		for (int i = 0; i < nodeSize; ++i)
		{
			if (m_Pathes[i] != other.m_Pathes[i])
			{
				return false;
			}
		}

		return true;
	}

	inline bool operator!=(const CPathPair& other) const
	{
		return !(*this == other);
	}

	std::vector<CPath> m_Pathes;
	bool m_bActive;
	size_t m_iHash;
};

template<> struct std::hash<CPathPair>
{
	std::size_t operator()(const CPathPair& path) const noexcept
	{
		return path.m_iHash;
	}
};

struct CInterval
{
	CInterval()
		: m_pNode1(nullptr)
		, m_pNode2(nullptr)
		, m_iIdx(0)
	{

	}

	CInterval(COneGrid* pNode1, COneGrid* pNode2, unsigned int iNodeCount)
		: m_pNode1(pNode1)
		, m_pNode2(pNode2)
	{
		unsigned int nodeidx1 = pNode1->m_iIdx;
		unsigned int nodeidx2 = pNode2->m_iIdx;
		m_iIdx = nodeidx1 > nodeidx2 ? (nodeidx2 * iNodeCount + nodeidx1) : (nodeidx1 * iNodeCount + nodeidx2);
	}

	inline CInterval& operator=(const CInterval& other)
	{ 
		m_pNode1 = other.m_pNode1;
		m_pNode2 = other.m_pNode2;
		m_iIdx = other.m_iIdx;
		return *this; 
	}

	inline bool operator==(const CInterval& other) const
	{
		return m_iIdx == other.m_iIdx;
	}

	inline bool operator!=(const CInterval& other) const
	{
		return m_iIdx != other.m_iIdx;
	}

	COneGrid* m_pNode1;
	COneGrid* m_pNode2;
	unsigned int m_iIdx;
};

class CIntervalList
{
public:

	CIntervalList()
		: m_bActive(false)
		, m_iHash(0)
	{
		m_intervals.reserve(_kDefaultCapacity);
	}

	CIntervalList(const std::vector<CInterval>& intervals)
		: m_bActive(true)
		, m_iHash(0)
	{
		m_intervals = intervals;
		int imuti = 1;
		size_t intervalSize = m_intervals.size();
		for (size_t i = 0; i < intervalSize; ++i)
		{
			for (size_t j = i + 1; j < intervalSize; ++j)
			{
				if (m_intervals[i].m_iIdx > m_intervals[j].m_iIdx)
				{
					CInterval tmp = m_intervals[i];
					m_intervals[i] = m_intervals[j];
					m_intervals[j] = tmp;
				}
			}
			m_iHash += imuti * m_intervals[i].m_iIdx;
			imuti *= 7;
		}
		m_iHash += 1000000 * static_cast<int>(intervalSize);
	}

	CIntervalList(const std::vector<CInterval>& intervals, const CInterval& add)
		: m_bActive(true)
		, m_iHash(0)
	{
		m_intervals = intervals;
		size_t intervalSize = m_intervals.size();
		for (int i = 0; i < intervalSize; ++i)
		{
			if (add.m_iIdx < m_intervals[i].m_iIdx)
			{
				m_intervals.insert(m_intervals.begin() + i, add);
				break;
			}

			if (add.m_iIdx == m_intervals[i].m_iIdx)
			{
				m_intervals.clear();
				return;
			}
		}

		int imuti = 1;
		for (int i = 0; i < intervalSize; ++i)
		{
			m_iHash += imuti * m_intervals[i].m_iIdx;
			imuti *= 7;
		}
		m_iHash += 1000000 * intervalSize;
	}

	inline bool operator==(const CIntervalList& other) const
	{
		if (m_iHash != other.m_iHash)
		{
			return false;
		}
		size_t nodeSize = m_intervals.size();
		if (nodeSize != other.m_intervals.size())
		{
			return false;
		}
		for (int i = 0; i < nodeSize; ++i)
		{
			if (m_intervals[i].m_iIdx != other.m_intervals[i].m_iIdx)
			{
				return false;
			}
		}

		return true;
	}

	inline bool operator!=(const CIntervalList& other) const
	{
		return !(*this == other);
	}

	std::vector<CInterval> m_intervals;
	bool m_bActive;
	size_t m_iHash;
};

template<> struct std::hash<CIntervalList>
{
	std::size_t operator()(const CIntervalList& inteveral) const noexcept
	{
		return inteveral.m_iHash;
	}
};

class IConnectionChecker
{
public:
	virtual ~IConnectionChecker()  { }
	virtual bool IsConnected(const COneGrid& grid1, const COneGrid& grid2, cuDoubleComplex& v) = 0;
};

class CGeneralGrid
{
public:

	CGeneralGrid(int width, int height, int edge);

	int m_midY;
	int m_width;
	int m_height;
	int m_edge;
	int m_startX;
	int m_startY;
	int m_endX;
	int m_endY;
	size_t m_gridsize;
	float m_sep;
	EAStarResult m_eResult;
	std::vector<std::vector<COneGrid>> m_gridArray;
	std::vector<COneGrid*> m_gridList;

	// totally reset to a new grid
	void ResetGrid()
	{
		ResetGridToStart(true);
	}

	// call before a new path finding
	void RestartGrid()
	{
		ResetGridToStart(false);
	}

	std::vector<CInterval> GetAllDisconnectIntervals() const;

	EAStarResult FindPath(IConnectionChecker* pCheck, std::vector<COneGrid*>& path, std::vector<cuDoubleComplex>& points, cuDoubleComplex& v);
	
	void SetHash(int iHash1, int iHash2)
	{
		size_t nodeCount = m_gridList.size();
		for (size_t i = 0; i < nodeCount; ++i)
		{
			m_gridList[i]->m_idx1 = m_gridList[i]->m_iIdx * iHash1;
			m_gridList[i]->m_idx2 = m_gridList[i]->m_iIdx * iHash2;
		}
	}

protected:

	void ResetGridToStart(bool bRecheckNeighbour);
	void CreateGrid();
	void CalculateConnectionStep(IConnectionChecker* pCheck, EGridDir d, int x, int y);
	EAStarResult OneStep(IConnectionChecker* pCheck);
};