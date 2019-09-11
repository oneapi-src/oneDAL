/* file: linear_model_predict_kernel.h */
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

/*
//++
//  Declaration of template function that computes linear regression
//  prediction results.
//--
*/

#ifndef __LINEAR_MODEL_PREDICT_KERNEL_H__
#define __LINEAR_MODEL_PREDICT_KERNEL_H__

#include "linear_model_predict.h"
#include "service_memory.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_blas.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{
namespace internal
{

template <typename algorithmFpType, prediction::Method method, CpuType cpu>
class PredictKernel : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Compute linear regression prediction results.
     *
     *  \param a[in]    Matrix of input variables X
     *  \param m[in]    Linear regression model obtained on training stage
     *  \param r[out]   Prediction results
     */
    services::Status compute(const NumericTable *a, const linear_model::Model *m, NumericTable *r);
};

template <typename algorithmFpType, CpuType cpu>
class PredictKernel<algorithmFpType, defaultDense, cpu> : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable *a, const linear_model::Model *m, NumericTable *r);

protected:
    void computeBlockOfResponses(DAAL_INT *numFeatures, DAAL_INT *numRows, const algorithmFpType *dataBlock,
                                 DAAL_INT *numBetas, const algorithmFpType *beta,
                                 DAAL_INT *numResponses, algorithmFpType *responseBlock, bool findBeta0);

    static const size_t _numRowsInBlock = 256;
};

} // namespace internal
}
}
}
} // namespace daal

#endif
