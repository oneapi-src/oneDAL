/* file: cholesky_kernel.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

//++
//  Declaration of template function that calculate choleskys.
//--


#ifndef __CHOLESKY_KERNEL_H__
#define __CHOLESKY_KERNEL_H__

#include "cholesky.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_lapack.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace cholesky
{
namespace internal
{
/**
 *  \brief Kernel for cholesky calculation
 *  in case floating point type of intermediate calculations
 *  and method of calculations are different
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class CholeskyKernel : public Kernel
{
public:
    services::Status compute(NumericTable *a, NumericTable *r, const daal::algorithms::Parameter *par);

private:
    services::Status copyMatrix(NumericTableIface::StorageLayout iLayout, const algorithmFPType *pA,
                    NumericTableIface::StorageLayout rLayout, algorithmFPType *pL, DAAL_INT dim) const;
    services::Status performCholesky(NumericTableIface::StorageLayout rLayout, algorithmFPType *pL, DAAL_INT dim);
    bool copyToFullMatrix(NumericTableIface::StorageLayout iLayout, const algorithmFPType *pA, algorithmFPType *pL,
                          DAAL_INT dim) const;
    bool copyToLowerTrianglePacked(NumericTableIface::StorageLayout iLayout, const algorithmFPType *pA, algorithmFPType *pL,
                                   DAAL_INT dim) const;
};

} // namespace daal::internal
}
}
} // namespace daal

#endif
