/* file: cholesky_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
    void compute(NumericTable *a, NumericTable *r, const daal::algorithms::Parameter *par);
private:
    void copyMatrix(NumericTableIface::StorageLayout iLayout, algorithmFPType **pA,
                    NumericTableIface::StorageLayout rLayout, algorithmFPType **pL, MKL_INT dim);
    void performCholesky(NumericTableIface::StorageLayout rLayout, algorithmFPType **pL, MKL_INT dim);
    void copyToFullMatrix(NumericTableIface::StorageLayout iLayout, algorithmFPType **pA, algorithmFPType **pL,
                          MKL_INT dim);
    void copyToLowerTrianglePacked(NumericTableIface::StorageLayout iLayout, algorithmFPType **pA, algorithmFPType **pL,
                                   MKL_INT dim);
};

} // namespace daal::internal
}
}
} // namespace daal

#endif
