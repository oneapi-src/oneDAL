/* file: multiclassclassifier_predict_kernel.h */
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
//  Declaration of template function that computes prediction results using
//  Multi-class classifier model.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_PREDICT_FPK_H__
#define __MULTICLASSCLASSIFIER_PREDICT_FPK_H__

#include "numeric_table.h"
#include "model.h"
#include "algorithm.h"
#include "multi_class_classifier_predict_types.h"
#include "service_defines.h"
#include "service_arrays.h"

using namespace daal::data_management;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace prediction
{
namespace internal
{
template<prediction::Method pmethod, training::Method tmethod, typename algorithmFPType, CpuType cpu>
struct MultiClassClassifierPredictKernel : public Kernel
{
    services::Status compute(const NumericTable *a, const daal::algorithms::Model *m, NumericTable *r,
                             const daal::algorithms::Parameter *par);
};

template<typename algorithmFPType, CpuType cpu>
size_t getMultiClassClassifierPredictBlockSize()
{
    return 128;
}

template<typename algorithmFPType, CpuType cpu>
services::Status getNonEmptyClassMap(size_t &nClasses, const Model *model, size_t *nonEmptyClassMap)
{
    TArray<bool, cpu> nonEmptyClassBuffer(nClasses);
    DAAL_CHECK_MALLOC(nonEmptyClassBuffer.get());
    bool *nonEmptyClass = (bool *)nonEmptyClassBuffer.get();
    for (size_t i = 0; i < nClasses; i++)
        nonEmptyClass[i] = false;

    for (size_t i = 1, imodel = 0; i < nClasses; i++)
    {
        for (size_t j = 0; j < i; j++, imodel++)
        {
            const bool ijModelNotEmpty(model->getTwoClassClassifierModel(imodel));
            nonEmptyClass[i] |= ijModelNotEmpty;
            nonEmptyClass[j] |= ijModelNotEmpty;
        }
    }

    size_t nNonEmptyClasses = 0;
    for (size_t i = 0; i < nClasses; i++)
    {
        if (nonEmptyClass[i])
            nonEmptyClassMap[nNonEmptyClasses++] = i;
    }
    nClasses = nNonEmptyClasses;
    return services::Status();
}

} // namespace internal
} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
