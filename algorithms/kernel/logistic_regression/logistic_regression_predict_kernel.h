/* file: logistic_regression_predict_kernel.h */
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
//  Declaration of template function that computes logistic regression
//  prediction results.
//--
*/

#ifndef __LOGISTIC_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __LOGISTIC_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "algorithms/logistic_regression/logistic_regression_predict.h"
#include "service_memory.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{
namespace internal
{

template <typename algorithmFpType, logistic_regression::prediction::Method method, CpuType cpu>
class PredictKernel : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Compute logistic regression prediction results.
     *
     *  \param a[in]   Matrix of input variables X
     *  \param m[in]   Logistic regression model obtained on training stage
     *  \param nClasses[in] Number of classes in logistic regression algorithm parameter
     *  \param pRes[out] Prediction results
     *  \param pProbab[out] Probability prediction results
     *  \param pLogProbab[out] Log of probability prediction results
     */
    services::Status compute(services::HostAppIface* pHostApp, const NumericTable *a, const logistic_regression::Model *m, size_t nClasses,
        NumericTable* pRes, NumericTable* pProbab, NumericTable* pLogProbab);
};

} // namespace internal
}
}
}
} // namespace daal

#endif
