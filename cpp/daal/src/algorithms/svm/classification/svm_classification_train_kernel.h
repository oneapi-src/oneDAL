/* file: svm_train_kernel.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

/*
//++
//  Declaration of template structs that calculate SVM Training functions.
//--
*/

#ifndef __SVM_TRAIN_KERNEL_H__
#define __SVM_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/model.h"
#include "services/daal_defines.h"
#include "algorithms/svm/svm_train_types.h"
#include "src/algorithms/kernel.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/svm/svm_train_common.h"

#include "src/algorithms/svm/svm_train_boser_cache.i"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, CpuType cpu>
struct SVMTrainTask
{
    static const size_t kernelFunctionBlockSize = 1024; /* Size of the block of kernel function elements */

    SVMTrainTask(size_t nVectors) : _cache(nullptr), _nVectors(nVectors) {}

    Status init(algorithmFPType C, const NumericTablePtr & wTable, NumericTable & yTable);

    Status setup(const Parameter & svmPar, const NumericTablePtr & xTable);

    /* Perform Sequential Minimum Optimization (SMO) algorithm to find optimal coefficients alpha */
    Status compute(const Parameter & svmPar);

    /* Write support vectors and classification coefficients into model */
    Status setResultsToModel(const NumericTable & xTable, Model & model) const;

    ~SVMTrainTask();

protected:
    inline void updateI(size_t index);

    bool findMaximumViolatingPair(size_t nActiveVectors, algorithmFPType tau, int & Bi, int & Bj, algorithmFPType & delta, algorithmFPType & ma,
                                  algorithmFPType & Ma, algorithmFPType & curEps, Status & s) const;

    services::Status WSSj(size_t nActiveVectors, algorithmFPType tau, int Bi, algorithmFPType GMin, int & Bj, algorithmFPType & delta,
                          algorithmFPType & res) const;

    Status reconstructGradient(size_t & nActiveVectors);

    Status update(size_t nActiveVectors, int Bi, int Bj, algorithmFPType delta);

    size_t updateShrinkingFlags(size_t nActiveVectors, algorithmFPType ma, algorithmFPType Ma);

    /*** Methods used in shrinking ***/
    size_t doShrink(size_t nActiveVectors);

    inline void updateAlpha(int Bi, int Bj, algorithmFPType delta, algorithmFPType & newDeltai, algorithmFPType & newDeltaj);

protected:
    const size_t _nVectors;                              //Number of observations in the input data set
    TArray<algorithmFPType, cpu> _y;                     //Array of class labels
    TArray<algorithmFPType, cpu> _alpha;                 //Array of classification coefficients
    TArray<algorithmFPType, cpu> _grad;                  //Objective function gradient
    TArray<algorithmFPType, cpu> _cw;                    //C[i] = C * weight[i]
    TArray<algorithmFPType, cpu> _kernelDiag;            //diagonal elements of the matrix Q (kernel(x[i], x[i]))
    TArray<char, cpu> _I;                                //array of flags I_LOW and I_UP
    SVMCacheIface<boser, algorithmFPType, cpu> * _cache; //caches matrix Q (kernel(x[i], x[j])) values
};

template <Method method, typename algorithmFPType, CpuType cpu>
struct SVMTrainImpl : public Kernel
{
    services::Status compute(const NumericTablePtr & xTable, const NumericTablePtr & wTable, NumericTable & yTable, daal::algorithms::Model * r,
                             const Parameter * par)
    {
        return services::ErrorMethodNotImplemented;
    }
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
