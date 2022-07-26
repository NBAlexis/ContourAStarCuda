#include "../Logger.h"
#include "../CExtendedPath4D.h"
#include "../CudaHelper.h"
#include "Test1.h"

__device__ cuDoubleComplex test2_func1(const cuDoubleComplex& x, const cuDoubleComplex& y, const cuDoubleComplex& z, const cuDoubleComplex& w)
{
	const cuDoubleComplex xm1 = make_cuDoubleComplex(x.x - 1.0, x.y);
	const cuDoubleComplex ym1 = make_cuDoubleComplex(y.x - 1.0, y.y);
	const cuDoubleComplex zm1 = make_cuDoubleComplex(z.x - 1.0, z.y);
	const cuDoubleComplex zp1 = make_cuDoubleComplex(z.x + 1.0, z.y);
	const cuDoubleComplex xy = cuCmul(x, y);
	const cuDoubleComplex z2 = cuCmul(z, z);
	const cuDoubleComplex y2 = cuCmul(y, y);
	const cuDoubleComplex z2p3 = make_cuDoubleComplex(z2.x + 3.0, z2.y);

	//(128*(-1 + x)*(-1 + z))
	const cuDoubleComplex num = cuCmulcr(cuCmul(xm1, zm1), 128.0);

	//d1 = (13 + 3*y + (-1 + y)*z^2 - x*(-1 + y)*(3 + z^2))
	cuDoubleComplex d1 = cuCadd(cuCmulcr(y, 3.0), cuCmul(ym1, z2));
	d1 = cuCsub(d1, cuCmul(cuCmul(x, ym1), z2p3));
	d1.x = d1.x + 13.0;

	//d2 = 83 + 69*x + 50*y + 54*x*y - 5*y^2 + 5*x*y^2 
	cuDoubleComplex d2 = cuCadd(cuCadd(cuCmulcr(x, 69.0), cuCmulcr(y, 50.0)), cuCmulcr(xy, 54.0));
	d2 = cuCadd(cuCsub(d2, cuCmulcr(y2, 5.0)), cuCmulcr(cuCmul(x, y2), 5.0));
	d2.x = d2.x + 83.0;
	//d3 = w^2 (-1 + z)^2 (7 + y + x (-1 + y) (-1 + z) + z - y z)
	cuDoubleComplex d3 = cuCsub(cuCadd(cuCadd(cuCmul(ym1, zm1), y), z), cuCmul(y, z));
	d3.x = d3.x + 7.0;
	d3 = cuCmul(cuCmul(cuCmul(cuCmul(d3, w), w), zm1), zm1);
	//d4 = 4 w (1 - y + x (3 + y)) (-1 + z^2)
	cuDoubleComplex d4 = cuCsub(cuCmul(make_cuDoubleComplex(y.x + 3.0, y.y), x), y);
	d4.x = d4.x + 1.0;
	d4 = cuCmulcr(cuCmul(cuCmul(d4, make_cuDoubleComplex(z2.x - 1.0, z2.y)), w), 4.0);
	//d5 = z (13 + 3 y + (7 + y) z + (-1 + y) z^2 - x (-1 + y) (3 + z + z^2))
	cuDoubleComplex d5 = cuCadd(cuCadd(cuCmul(make_cuDoubleComplex(y.x + 7.0, y.y), z), cuCmul(ym1, z2)), cuCmulcr(y, 3.0));
	d5 = cuCsub(d5, cuCmul(cuCmul(cuCadd(z, z2p3), ym1), x));
	d5.x = d5.x + 13.0;
	d5 = cuCmul(d5, z);
	
	//d6 = (-7 + z + x*(-1 + y)*(1 + z) - y*(1 + z))
	cuDoubleComplex d6 = cuCadd(cuCsub(cuCmul(cuCmul(ym1, zp1), x), cuCmul(y, zp1)), z);
	d6.x = d6.x - 7.0;
	//d7 = (-13 + z^2 + x*(-1 + y)*(3 + z^2) - y*(3 + z^2)))
	cuDoubleComplex d7 = cuCadd(cuCsub(cuCmul(cuCmul(ym1, z2p3), x), cuCmul(z2p3, y)), z2);
	d7.x = d7.x - 13.0;
	
	// d = d1 * (  s(x-1)(d2 + (y-1)(d3-d4+d5) )  +   4*mlsq*d6*d7   )
	cuDoubleComplex d = cuCmul(cuCadd(cuCsub(d3, d4), d5), ym1);
	d = cuCmul(cuCadd(d, d2), xm1);
	d = cuCmul(cuCadd(cuCmulcr(cuCmul(d6, d7), 4.0 * 0.142857), d), d1);

	return cuCdiv(num, d);
}

__device__ integrand4d test2_fp1 = test2_func1;

void Test2()
{
	CExtendedPath4D testgrid(9, 9, 0, "../Data/SparseGrid4D/GaussPatterson/", -1, 0.5);
	testgrid.SetIntegratorVerb(EVerboseLevel::GENERAL);
	integrand4d hfp1;
	cudaMemcpyFromSymbol(&hfp1, test2_fp1, sizeof(integrand4d));
	SIntegrateRes res = testgrid.Integrate(hfp1);
	
	LogGeneral("%d, res = %f + %f I", res.m_bDone, res.m_v.x, res.m_v.y);

	testgrid.PrintResoult("(128*(-1 + x)*(-1 + z))/((13 + 3*y + (-1 + y)*z^2 - x*(-1 + y)*(3 + z^2))*((-1 + x)*(83 + 69*x + 50*y + 54*x*y - 5*y^2 + 5*x*y^2 - (-1 + y)*(-13 - 3*x + 3*(-1 + x)*y)*z - (-1 + y)*(-7 - x + (-1 + x)*y)*z^2 - (-1 + x)*(-1 + y)^2*z^3 + w ^ 2 * (-1 + y) * (-1 + z) ^ 2 * (7 + x + y - x * y + (-1 + x) * (-1 + y) * z) - 4 * w * (-1 + y) * (1 + 3 * x + (-1 + x) * y) * (-1 + z ^ 2)) + 4 * 0.142857 * (-7 + z + x * (-1 + y) * (1 + z) - y * (1 + z)) * (-13 + z ^ 2 + x * (-1 + y) * (3 + z ^ 2) - y * (3 + z ^ 2))))");
}


