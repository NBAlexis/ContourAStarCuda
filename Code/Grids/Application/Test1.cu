#include <cuda_runtime.h>
#include "../CGeneralGrids.h"
#include "../Integrator/CSparseGrid4D.h"
#include "../Logger.h"
#include "../CExtendedPath4D.h"
#include "Test1.h"

__device__ cuDoubleComplex test1_func1(const cuDoubleComplex& x, const cuDoubleComplex& y, const cuDoubleComplex& z, const cuDoubleComplex& w)
{
	cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);

	return cuCdiv(one,
		cuCadd(cuCadd(x, cuCmul(y, y)),
			cuCadd(z, cuCmul(w, cuCmul(w, w)))));
}

__device__ cuDoubleComplex test1_func2(const cuDoubleComplex& x, const cuDoubleComplex& y, const cuDoubleComplex& z, const cuDoubleComplex& w)
{
	cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);

	return cuCdiv(one,
		cuCadd(cuCadd(x, y),
			   cuCadd(z, w)));
}

__device__ integrand4d test1_fp1 = test1_func1;
__device__ integrand4d test1_fp2 = test1_func2;

void Test1()
{
	//CQuadratureReader reader("../Data/SparseGrid4D/GaussPatterson/", EIntegralDimension::EID_4D);
	CExtendedPath4D testgrid(5, 5, 1, "../Data/SparseGrid4D/GaussPatterson/");
	testgrid.SetIntegratorVerb(EVerboseLevel::GENERAL);
	integrand4d hfp1;
	cudaMemcpyFromSymbol(&hfp1, test1_fp1, sizeof(integrand4d));
	SIntegrateRes res = testgrid.Integrate(hfp1);
	LogGeneral("%d, res = %f + %f I", res.m_bDone, res.m_v.x, res.m_v.y);
	testgrid.PrintResoult("1/(x+y*y+z+w*w*w)");

	integrand4d hfp2;
	cudaMemcpyFromSymbol(&hfp2, test1_fp2, sizeof(integrand4d));
	res = testgrid.Integrate(hfp2);
	LogGeneral("%d, res = %f + %f I", res.m_bDone, res.m_v.x, res.m_v.y);
	testgrid.PrintResoult("1/(x+y+z+w)");
}


