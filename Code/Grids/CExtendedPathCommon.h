#pragma once

#include "CGeneralGrids.h"

struct SPathInterval
{
	SPathInterval(const CPath& p, const CIntervalList& l)
	{
		m_p = p;
		m_inter = l;
	}
	CPath m_p;
	CIntervalList m_inter;
};

struct SPathPairInterval
{
	SPathPairInterval(const CPathPair& p, const CIntervalList& l)
	{
		m_p = p;
		m_inter = l;
	}
	CPathPair m_p;
	CIntervalList m_inter;
};

struct SIntegrateRes
{
	SIntegrateRes(bool bDone, const cuDoubleComplex& v)
		: m_bDone(bDone)
		, _unused1(0)
	{
		m_v = v;
	}

	cuDoubleComplex m_v;
	bool m_bDone;
	int _unused1;
};