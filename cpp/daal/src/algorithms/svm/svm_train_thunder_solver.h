/* file: svm_train_thunder_solver.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  SVM training algorithm implementation using Thunder method
//--
*/
/*
//  DESCRIPTION
//
//  Definition of the functions for training with SVM 2-class classifier.
//
//  REFERENCES
//
//  1. Zeyi Wen, Jiashuai Shi, Bingsheng He
//     ThunderSVM: A Fast SVM Library on GPUs and CPUs,
//     Journal of Machine Learning Research, 19, 1-5 (2018)
//  2. Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin,
//     Working Set Selection Using Second Order Information
//     for Training Support Vector Machines,
//     Journal of Machine Learning Research 6 (2005), pp. 1889___1918
//  3. Bernard E. boser, Isabelle M. Guyon, Vladimir N. Vapnik,
//     A Training Algorithm for Optimal Margin Classifiers.
//  4. Thorsten Joachims, Making Large-Scale SVM Learning Practical,
//     Advances in Kernel Methods - Support Vector Learning
*/

#ifndef __SVM_TRAIN_THUNDER_I__
#define __SVM_TRAIN_THUNDER_I__

#include "src/externals/service_memory.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_utils.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_ittnotify.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_math.h"

#include "src/algorithms/svm/svm_train_common.h"
#include "src/algorithms/svm/svm_train_thunder_workset.h"
#include "src/algorithms/svm/svm_train_thunder_cache.h"
#include "src/algorithms/svm/svm_train_result.h"

#include "src/algorithms/svm/svm_train_common_impl.i"

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
template <typename algorithmFPType, CpuType cpu>
class Solver<thunder, algorithmFPType, cpu>
{
public:
    template <typename algorithmFPType, CpuType cpu>
    services::Status Solver<thunder, algorithmFPType, cpu>::compute(const NumericTablePtr & xTable, const NumericTablePtr & wTable,
                                                                    NumericTable & yTable, daal::algorithms::Model * r, const Parameter * svmPar);

protected:
    services::Status SMOBlockSolver(const uint32_t * wsIndices, const NumericTablePtr & kernelWS);

    services::Status updateGrad(const NumericTablePtr & kernelWS, const size_t nWS);
    services::Status checkStopCondition(const algorithmFPType diff, const algorithmFPType diffPrev)

        private : size_t _blockSizeWS;

    // One of the conditions for stopping is diff stays unchanged. nNoChanges - number of repetitions
    static const size_t nNoChanges = 5;
    // The maximum numbers of iteration of the subtask is number of observation in WS x cInnerIterations.
    // It's enough to find minimum for subtask.
    static const size_t cInnerIterations = 100;
    // The maximum block size for blocked SMO solver.
    // Need of (maxBlockSize*6 + maxBlockSize*maxBlockSize)*sizeof(algorithmFPType) internal memory.
    // It should fit into the cache L2 including the use of hardware prefetch.
    static const size_t maxBlockSize = 2048;

    enum MemSmoId
    {
        alphaBuffID    = 0,
        yBuffID        = 1,
        gradBuffID     = 2,
        kdBuffID       = 3,
        oldAlphaBuffID = 4,
        cwBuffID       = 5,
        latest         = cwBuffID + 1,
    };

    const algorithmFPType * y;
    const algorithmFPType * grad;
    const algorithmFPType * cw;
    algorithmFPType * buffer;
    const size_t nVectors;
    const double accuracyThreshold;
    const double tau;
    const double I;
    algorithms * alpha;
    algorithms * deltaAlpha;
    algorithms localDiff;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
