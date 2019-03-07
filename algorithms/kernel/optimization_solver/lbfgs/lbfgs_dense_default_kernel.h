/* file: lbfgs_dense_default_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Declaration of template function that computes LBFGS.
//--


#ifndef __LBFGS_DENSE_DEFAULT_KERNEL_H__
#define __LBFGS_DENSE_DEFAULT_KERNEL_H__

#include "lbfgs_base.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace internal
{

/**
 *  \brief Kernel for LBFGS computation
 */
template<typename algorithmFPType, CpuType cpu>
class LBFGSKernel<algorithmFPType, defaultDense, cpu> : public Kernel
{
public:
    services::Status compute(HostAppIface* pHost, NumericTable* correctionPairsInput, NumericTable* correctionIndicesInput,
                 NumericTable *inputArgument, NumericTable* averageArgLIterInput, OptionalArgument *optionalArgumentInput, NumericTable *correctionPairsResult,
                 NumericTable *correctionIndicesResult, NumericTable *minimum, NumericTable *nIterationsNT, NumericTable *averageArgLIterResult,
                 OptionalArgument *optionalArgumentResult, Parameter *parameter, engines::BatchBase &engine);
};

} // namespace daal::internal

} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
