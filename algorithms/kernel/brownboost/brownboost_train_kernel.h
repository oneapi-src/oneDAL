/* file: brownboost_train_kernel.h */
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
//  Declaration of template function that trains Brown Boost model.
//--
*/

#ifndef __BROWN_BOOST_TRAIN_KERNEL_H__
#define __BROWN_BOOST_TRAIN_KERNEL_H__

#include "brownboost_model.h"
#include "brownboost_training_types.h"
#include "kernel.h"
#include "service_numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace training
{
namespace internal
{

template <Method method, typename algorithmFPType, CpuType cpu>
class BrownBoostTrainKernel : public Kernel
{
public:
    services::Status compute(size_t n, NumericTablePtr *a, Model *r, const Parameter *par);

private:
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;

    void updateWeights(size_t nVectors, algorithmFPType s, algorithmFPType c, algorithmFPType invSqrtC,
                       const algorithmFPType *r, algorithmFPType *nra, algorithmFPType *nre2, algorithmFPType *w);

    algorithmFPType *reallocateAlpha(size_t oldAlphaSize, size_t alphaSize, algorithmFPType *oldAlpha);

    services::Status brownBoostFreundKernel(size_t nVectors,
                                NumericTablePtr weakLearnerInputTables[],
                                const HomogenNTPtr& hTable, const algorithmFPType *y,
                                Model *boostModel, Parameter *parameter, size_t& nWeakLearners,
                                algorithmFPType *&alpha);
};

template <Method method, typename algorithmFPType, CpuType cpu>
struct NewtonRaphsonKernel
{
    NewtonRaphsonKernel(size_t nVectors, Parameter *parameter);
    bool isValid() const
    {
        return (aNrd.get() && aNrw.get() && aNra.get() && aNrb.get() && aNre1.get() && aNre2.get());
    }

    void compute(algorithmFPType gamma, algorithmFPType s, const algorithmFPType *h, const algorithmFPType *y);

    size_t nVectors;
    algorithmFPType nrT;
    algorithmFPType nrAlpha;

    algorithmFPType c;
    daal::internal::TArray<algorithmFPType, cpu> aNrd;
    daal::internal::TArray<algorithmFPType, cpu> aNrw;
    daal::internal::TArray<algorithmFPType, cpu> aNra;
    daal::internal::TArray<algorithmFPType, cpu> aNrb;
    daal::internal::TArray<algorithmFPType, cpu> aNre1;
    daal::internal::TArray<algorithmFPType, cpu> aNre2;

    const size_t nrMaxIter;
    const algorithmFPType error;
    const algorithmFPType nrAccuracy;
    algorithmFPType nu;
    algorithmFPType invC;
    algorithmFPType sqrtC;
    algorithmFPType invSqrtC;
    algorithmFPType sqrtPiC;
};

} // namespace daal::algorithms::brownboost::training::internal
}
}
}
} // namespace daal

#endif
