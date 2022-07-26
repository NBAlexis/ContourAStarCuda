#include "CSparseGrid4D.h"
#include "QuadratureReader.h"
#include "../CudaHelper.h"
#include "../Logger.h"

#pragma region cuda

__global__ void
LAUNCH_BOUND
_kernalCalcV(
	integrand4d integrand,
	cuDoubleComplex strideX, cuDoubleComplex strideY, cuDoubleComplex strideZ, cuDoubleComplex strideW,
	cuDoubleComplex fromX, cuDoubleComplex fromY, cuDoubleComplex fromZ, cuDoubleComplex fromW, 
	cuDoubleComplex* values, 
	const double* __restrict__ points, 
	unsigned int pointStart, unsigned int iMaxP)
{
	const unsigned int uiId = (threadIdx.x + blockIdx.x * blockDim.x);

	if (uiId < iMaxP)
	{
		const unsigned int realId = uiId + pointStart;
		const cuDoubleComplex xV = cuCadd(cuCmulcr(strideX, 1.0 + points[4U * realId]), fromX);
		const cuDoubleComplex yV = cuCadd(cuCmulcr(strideY, 1.0 + points[4U * realId + 1U]), fromY);
		const cuDoubleComplex zV = cuCadd(cuCmulcr(strideZ, 1.0 + points[4U * realId + 2U]), fromZ);
		const cuDoubleComplex wV = cuCadd(cuCmulcr(strideW, 1.0 + points[4U * realId + 3U]), fromW);

		values[realId] = (*integrand)(xV, yV, zV, wV);
	}
}

__global__ void
LAUNCH_BOUND
_kernalMultWeight(
	cuDoubleComplex* res,
	const cuDoubleComplex* __restrict__ values, 
	const double* __restrict__ weights,
	unsigned int iMaxP)
{
	unsigned int uiId = (threadIdx.x + blockIdx.x * blockDim.x);

	if (uiId < iMaxP)
	{
		res[uiId] = cuCmulcr(values[uiId], weights[uiId]);
	}
}

#pragma endregion

CSparseGrid4D::CSparseGrid4D(const std::string& folderName, int iMaxOrder, double epsilon)
	: CLogger()
	, m_iOrder(0)
	, m_pWeightBuffer(nullptr)
	, m_pPointBuffer(nullptr)
	, m_pValueBuffer(nullptr)
	, m_pValueBufferMult(nullptr)
	, m_fEpsilon(epsilon)
{
	CQuadratureReader reader(folderName, EIntegralDimension::EID_4D, iMaxOrder);
	m_iOrder = static_cast<int>(reader.m_lstWeights.size());
	int maxPoint = static_cast<int>(reader.m_lstWeights[m_iOrder - 1].size());
	int lastPointCount = static_cast<int>(reader.m_lstNewPoints[m_iOrder - 1].size());
	double* points = (double*)malloc(sizeof(double) * maxPoint * 4);
	double* pointdata = (double*)malloc(sizeof(double) * lastPointCount * 4);
	double* weightdata = (double*)malloc(sizeof(double) * maxPoint);

	double** pWeightBuffer = (double**)malloc(sizeof(double*) * m_iOrder);

	int iStart = 0;
	int iOrderStart = 0;
	int iOrderIdx = 0;
	//skip the small orders
	for (int i = 0; i < m_iOrder; ++i)
	{
		int iNewPointCount = static_cast<int>(reader.m_lstNewPoints[i].size());
		for (int j = 0; j < iNewPointCount; ++j)
		{
			for (int k = 0; k < 4; ++k)
			{
				pointdata[k + j * 4] = reader.m_lstNewPoints[i][j][k];
			}
		}
		memcpy(points + iStart * 4, pointdata, sizeof(double) * 4 * iNewPointCount);
		iStart = iStart + iNewPointCount;
		
		if (iStart > _kMaxCudaThread)
		{
			m_iPointStart.push_back(0);
			m_iPointEnd.push_back(iStart);
			m_iBlockCountStep1.push_back((iStart / _kMaxCudaThread) + 1);

			std::copy(reader.m_lstWeights[i].begin(), reader.m_lstWeights[i].end(), weightdata);
			double* deviceWeightBuffer = nullptr;
			size_t weightCount = reader.m_lstWeights[i].size();
			m_iWeightCount.push_back(static_cast<int>(weightCount));
			checkCudaErrors(cudaMalloc((void**)&deviceWeightBuffer, sizeof(double) * weightCount));
			checkCudaErrors(cudaMemcpy(deviceWeightBuffer, weightdata, sizeof(double) * weightCount, cudaMemcpyHostToDevice));
			pWeightBuffer[iOrderIdx] = deviceWeightBuffer;
			m_iBlockCountStep2.push_back((static_cast<int>(weightCount) / _kMaxCudaThread) + 1);

			++iOrderIdx;
			iOrderStart = i + 1;
			break;
		}
	}

	for (int i = iOrderStart; i < m_iOrder; ++i)
	{
		int iNewPointCount = static_cast<int>(reader.m_lstNewPoints[i].size());
		for (int j = 0; j < iNewPointCount; ++j)
		{
			for (int k = 0; k < 4; ++k)
			{
				pointdata[k + j * 4] = reader.m_lstNewPoints[i][j][k];
			}
		}
		memcpy(points + iStart * 4, pointdata, sizeof(double) * 4 * iNewPointCount);

		m_iPointStart.push_back(iStart);
		iStart = iStart + iNewPointCount;
		m_iPointEnd.push_back(iStart);
		m_iBlockCountStep1.push_back((iNewPointCount / _kMaxCudaThread) + 1);

		std::copy(reader.m_lstWeights[i].begin(), reader.m_lstWeights[i].end(), weightdata);
		double* deviceWeightBuffer = nullptr;
		size_t weightCount = reader.m_lstWeights[i].size();
		m_iWeightCount.push_back(static_cast<int>(weightCount));
		checkCudaErrors(cudaMalloc((void**)&deviceWeightBuffer, sizeof(double) * weightCount));
		checkCudaErrors(cudaMemcpy(deviceWeightBuffer, weightdata, sizeof(double) * weightCount, cudaMemcpyHostToDevice));
		pWeightBuffer[iOrderIdx] = deviceWeightBuffer;
		m_iBlockCountStep2.push_back((static_cast<int>(weightCount) / _kMaxCudaThread) + 1);
		++iOrderIdx;
	}

	m_iOrder = iOrderIdx;
	LogGeneral("Loaded integrator data with order:%d\n", m_iOrder);
	m_pWeightBuffer = (double**)malloc(sizeof(double*) * m_iOrder);
	memcpy(m_pWeightBuffer, pWeightBuffer, sizeof(double*) * m_iOrder);
	checkCudaErrors(cudaMalloc((void**)&m_pPointBuffer, sizeof(double) * maxPoint * 4));
	checkCudaErrors(cudaMemcpy(m_pPointBuffer, points, sizeof(double) * maxPoint * 4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&m_pValueBuffer, sizeof(cuDoubleComplex) * lastPointCount));
	checkCudaErrors(cudaMalloc((void**)&m_pValueBufferMult, sizeof(cuDoubleComplex) * lastPointCount));

	free(points);
	free(pointdata);
	free(weightdata);
	free(pWeightBuffer);
	//checkCudaErrors(cudaThreadSynchronize());
}

CSparseGrid4D::~CSparseGrid4D()
{
	for (int i = 0; i < m_iOrder; ++i)
	{
		checkCudaErrors(cudaFree(m_pWeightBuffer[i]));
	}
	checkCudaErrors(cudaFree(m_pPointBuffer));
	checkCudaErrors(cudaFree(m_pValueBuffer));
	checkCudaErrors(cudaFree(m_pValueBufferMult));

	free(m_pWeightBuffer);
}

bool CSparseGrid4D::Integrate(integrand4d integrand,
	cuDoubleComplex fromX, cuDoubleComplex toX,
	cuDoubleComplex fromY, cuDoubleComplex toY,
	cuDoubleComplex fromZ, cuDoubleComplex toZ,
	cuDoubleComplex fromW, cuDoubleComplex toW,
	cuDoubleComplex& res)
{
	cuDoubleComplex resOld = make_cuDoubleComplex(0.0, 0.0);
	const cuDoubleComplex strideX = cuCmulcr(cuCsub(toX, fromX), 0.5);
	const cuDoubleComplex strideY = cuCmulcr(cuCsub(toY, fromY), 0.5);
	const cuDoubleComplex strideZ = cuCmulcr(cuCsub(toZ, fromZ), 0.5);
	const cuDoubleComplex strideW = cuCmulcr(cuCsub(toW, fromW), 0.5);
	double delta = 0;

	for (int i = 0; i < m_iOrder; ++i)
	{
		const unsigned int uiBlockCount1 = static_cast<unsigned int>(m_iBlockCountStep1[i]);
		_kernalCalcV <<<uiBlockCount1, _kMaxCudaThread >>> (
			integrand, 
			strideX, strideY, strideZ, strideW,
			fromX, fromY, fromZ, fromW,
			m_pValueBuffer, 
			m_pPointBuffer,
			static_cast<unsigned int>(m_iPointStart[i]),
			static_cast<unsigned int>(m_iPointEnd[i] - m_iPointStart[i])
			);

		//checkCudaErrors(cudaThreadSynchronize());

		const unsigned int uiBlockCount2 = static_cast<unsigned int>(m_iBlockCountStep2[i]);
		_kernalMultWeight <<<uiBlockCount2, _kMaxCudaThread >>> (
			m_pValueBufferMult,
			m_pValueBuffer,
			m_pWeightBuffer[i],
			static_cast<unsigned int>(m_iWeightCount[i])
			);

		//checkCudaErrors(cudaThreadSynchronize());

		cuDoubleComplex resNew = ReduceComplex(m_pValueBufferMult, m_iWeightCount[i]);

		//checkCudaErrors(cudaThreadSynchronize());

		if (resNew.x != resNew.x || resNew.y != resNew.y)
		{
			//nan encountered
			res = resNew;
			return false;
		}

		if (i > 0)
		{
			double checkDelta = cuCabs(resNew);
			checkDelta = checkDelta < 1.0e-6 ? 1.0e-6 : checkDelta;
			//checkDelta = checkDelta > 1.0e6 ? 1.0e6 : checkDelta;
			delta = cuCabs(cuCsub(resNew, resOld)) / checkDelta;
			MyLogParanoiac("delta = %2.12f\n", delta);

			if (delta < m_fEpsilon)
			{
				res = cuCmul(resNew, 
					cuCmul(cuCmul(strideX, strideY), 
						   cuCmul(strideZ, strideW)));
				MyLogDetailed("{x, %f+%f I, %f+%f I},{y, %f+%f I, %f+%f I},\n{z, %f+%f I, %f+%f I},{w, %f+%f I, %f+%f I}:\n res = %f + %f I\n",
					fromX.x, fromX.y, toX.x, toX.y,
					fromY.x, fromY.y, toY.x, toY.y,
					fromZ.x, fromZ.y, toZ.x, toZ.y,
					fromW.x, fromW.y, toW.x, toW.y,
					res.x, res.y);
				return true;
			}
		}
		resOld = resNew;
	}
	
	res = cuCmul(resOld,
		cuCmul(cuCmul(strideX, strideY),
			   cuCmul(strideZ, strideW)));
	MyLogDetailed("{x, %f+%f I, %f+%f I},{y, %f+%f I, %f+%f I},\n{z, %f+%f I, %f+%f I},{w, %f+%f I, %f+%f I}:\n integralte failed last delta = %2.12f, res = %f + %f I\n", 
		fromX.x, fromX.y, toX.x, toX.y,
		fromY.x, fromY.y, toY.x, toY.y,
		fromZ.x, fromZ.y, toZ.x, toZ.y,
		fromW.x, fromW.y, toW.x, toW.y,
		delta, res.x, res.y);
	return false;
}
