/* file: kernel_function_dense_base.h */
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

/*
//++
//  Declaration of template structs that calculate SVM Kernel functions.
//--
*/

#ifndef __KERNEL_FUNCTION_DENSE_BASE_H__
#define __KERNEL_FUNCTION_DENSE_BASE_H__

#include "numeric_table.h"
#include "kernel_function_types_linear.h"
#include "kernel_function_types_rbf.h"
#include "kernel_function_linear.h"
#include "kernel_function_rbf.h"
#include "service_micro_table.h"
#include "kernel.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
struct KernelImplBase : public Kernel
{

    virtual services::Status computeInternalVectorVector(const NumericTable *a1, const NumericTable *a2, NumericTable *r, const ParameterBase *par) = 0;
    virtual services::Status computeInternalMatrixVector(const NumericTable *a1, const NumericTable *a2, NumericTable *r, const ParameterBase *par) = 0;
    virtual services::Status computeInternalMatrixMatrix(const NumericTable *a1, const NumericTable *a2, NumericTable *r, const ParameterBase *par) = 0;

    services::Status compute(ComputationMode computationMode, const NumericTable *a1, const NumericTable *a2, NumericTable *r,
                 const daal::algorithms::Parameter *par)
    {
        const ParameterBase *svmPar = static_cast<const ParameterBase *>(par);

        switch(computationMode)
        {
        case vectorVector:
            return computeInternalVectorVector(a1, a2, r, svmPar);
            break;
        case matrixVector:
            return computeInternalMatrixVector(a1, a2, r, svmPar);
            break;
        case matrixMatrix:
            return computeInternalMatrixMatrix(a1, a2, r, svmPar);
            break;
        }

        DAAL_ASSERT(false); //should never come here
        return services::Status();
    }
};

} // namespace internal

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
