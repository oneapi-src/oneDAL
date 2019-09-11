/* file: stump_train_kernel.h */
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
//  Declaration of template function that trains Decision Stump.
//--
*/

#ifndef __STUMP_TRAIN_KERNEL_H__
#define __STUMP_TRAIN_KERNEL_H__

#include "stump_training_types.h"
#include "stump_model.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace training
{
namespace internal
{

template <Method method, typename algorithmFPtype , CpuType cpu>
class StumpTrainKernel : public Kernel
{
public:
    services::Status compute(size_t n, const NumericTable *const *a, Model *r, const Parameter *par);

private:
    void StumpQSort( size_t n, algorithmFPtype *x, algorithmFPtype *w, algorithmFPtype *z );

    services::Status stumpRegressionOrdered(size_t nVectors,
                                const algorithmFPtype *x, const algorithmFPtype *w, const algorithmFPtype *z,
                                algorithmFPtype sumW, algorithmFPtype sumM, algorithmFPtype sumS,
                                algorithmFPtype &minS, algorithmFPtype& splitPoint,
                                algorithmFPtype& lMean, algorithmFPtype& rMean);

    services::Status stumpRegressionCategorical(size_t n, size_t nCategories,
                                    const int *x, const algorithmFPtype *w, const algorithmFPtype *z,
                                    algorithmFPtype sumW, algorithmFPtype sumM, algorithmFPtype sumS,
                                    algorithmFPtype &minS, algorithmFPtype& splitPoint,
                                    algorithmFPtype& lMean, algorithmFPtype& rMean);

    void computeSums(size_t n, const algorithmFPtype *w, const algorithmFPtype *z, algorithmFPtype& sumW, algorithmFPtype& sumM,
                     algorithmFPtype& sumS);

    services::Status doStumpRegression(size_t n, size_t dim, const NumericTable *x, const algorithmFPtype *w,
        const algorithmFPtype *z, size_t& splitFeature, algorithmFPtype& splitPoint,
        algorithmFPtype& leftValue, algorithmFPtype& rightValue);
};

} // namespace daal::algorithms::stump::training::internal
}
}
}
} // namespace daal

#endif
