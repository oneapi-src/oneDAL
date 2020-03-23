/* file: svm_train_boser_impl.i */
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
//  SVM training algorithm implementation
//--
*/
/*
//  DESCRIPTION
//
//  Definition of the functions for training with SVM 2-class classifier.
//
//  REFERENCES
//
//  1. Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin,
//     Working Set Selection Using Second Order Information
//     for Training Support Vector Machines,
//     Journal of Machine Learning Research 6 (2005), pp. 1889___1918
//  2. Bernard E. boser, Isabelle M. Guyon, Vladimir N. Vapnik,
//     A Training Algorithm for Optimal Margin Classifiers.
//  3. Thorsten Joachims, Making Large-Scale SVM Learning Practical,
//     Advances in Kernel Methods - Support Vector Learning
*/

#ifndef __SVM_TRAIN_GPU_IMPL_I__
#define __SVM_TRAIN_GPU_IMPL_I__

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_micro_table.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/service_utils.h"
#include "service/kernel/service_data_utils.h"

#include "algorithms/kernel/svm/oneapi/oneapi/cl_kernel/svm_train_oneapi.cl"

#include <cstdlib>

using namespace daal::internal;
using namespace daal::services::internal;

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
template <typename algorithmFPType, typename ParameterType>
services::Status SVMTrainOneAPI<algorithmFPType, boser>::initGrad(const services::Buffer<algorithmFPType> & y, services::Buffer<algorithmFPType> & f,
                                                                  const size_t n)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(initGrad);

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    services::Status status = buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("initGradient");

    KernelArguments args(2);
    args.set(0, y, AccessModeIds::read);
    args.set(1, f, AccessModeIds::write);

    KernelRange range(n);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

// void SelectWorkingSet(VectorView<Int> ws,
//                       const VectorView<bool> &ws_indicator,
//                       const Vector<Int> &f_argsort,
//                       const Vector<Int> &yy,
//                       const Vector<Float> &aa,
//                       Float C) {

//     const Size ws_size = ws.size();
//     const Size n_samples = yy.size();

//     Size n_selected = 0;
//     Size p_left = 0;
//     long long p_right = n_samples - 1;

//     const Int *yy_data = yy.data();
//     const Float *aa_data = aa.data();
//     const Int *f_argsort_data = f_argsort.data();

//     Int *ws_data = ws.data();
//     bool *ws_indicator_data = ws_indicator.data();

//     while (n_selected < ws_size) {
//         if (p_left < n_samples) {
//             Int i = f_argsort_data[p_left];
//             while ( ws_indicator_data[i] || !is_I_up(aa_data[i], yy_data[i], C) ) {
//                 p_left++;
//                 if (p_left == n_samples) {
//                     break;
//                 }
//                 i = f_argsort_data[p_left];
//             }
//             if (p_left < n_samples) {
//                 ws_data[n_selected] = i;
//                 ws_indicator_data[i] = 1;
//                 n_selected++;
//             }
//         }

//         if (p_right >= 0) {
//             Int i = f_argsort_data[p_right];
//             while ( ws_indicator_data[i] || !is_I_low(aa_data[i], yy_data[i], C) ) {
//                 p_right--;
//                 if (p_right == -1) {
//                     break;
//                 }
//                 i = f_argsort_data[p_right];
//             }
//             if (p_right >= 0) {
//                 ws_data[n_selected] = i;
//                 ws_indicator_data[i] = 1;
//                 n_selected++;
//             }
//         }
//     }

//     // ws.Sort();
// }

template <typename algorithmFPType, typename ParameterType>
services::Status SVMTrainOneAPI<algorithmFPType, boser>::compute(const NumericTablePtr & xTable, NumericTable & yTable, daal::algorithms::Model * r,
                                                                 const ParameterType * svmPar)
{
    services::Status status;

    if (const char * env_p = std::getenv("SVM_VERBOSE"))
    {
        printf(">> VERBOSE MODE\n");
        verbose = true;
    }

    const algorithmFPType C(svmPar.C);
    const algorithmFPType eps(svmPar.accuracyThreshold);
    const algorithmFPType tau(svmPar.tau);
    const size_t maxIterations(svmPar.maxIterations);
    // TODO
    const size_t innerMaxIterations(100);

    size_t nVectors = xTable->getNumberOfRows();

    // ai = 0
    UniversalBuffer alpha = ctx.allocate(idType, nVectors, &status);
    ctx.fill(alpha, 0.0, &status);
    DAAL_CHECK_STATUS_VAR(status);

    // fi = -yi
    UniversalBuffer f = ctx.allocate(idType, nVectors, &status);
    DAAL_CHECK_STATUS_VAR(status);
    DAAL_CHECK_STATUS(status, initGrad(y, f, nVectors));

    UniversalBuffer alpha = ctx.allocate(idType, nVectors, &status);
    DAAL_CHECK_STATUS_VAR(status);

    const size_t nWS = SelectWorkingSetSize(nVectors);

    if (verbose)
    {
        printf(">> LINE: %lu: nWS %lu\n", __LINE__, nWS);
    }

    // TODO transfer on GPU

    // for (size_t iter = 0; iter < maxIterations; i++)
    {
        if (verbose)
        {
            const auto t_0 = high_resolution_clock::now();
        }

        SelectWS();

        if (verbose)
        {
            const auto t_1           = high_resolution_clock::now();
            const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
            printf(">> SelectWS.compute time(ms) = %f\n", duration_sec);
        }
    }

    // return s.ok() ? task.setResultsToModel(*xTable, *static_cast<Model *>(r), svmPar->C) : s;
}

// inline Size MaxPow2(Size n) {
//     if (!(n & (n - 1))) {
//         return n;
//     }

//     Size count = 0;
//     while (n > 1) {
//         n >>= 1;
//         count++;
//     }
//     return 1 << count;
// }

template <typename algorithmFPType, typename ParameterType>
size_t SVMTrainOneAPI<boser, algorithmFPType, cpu>::SelectWorkingSetSize(const size_t n)
{
    // Depends on cache size
    // constexpr Size max_ws = 512;
    // constexpr Size max_ws = 1024;
    constexpr size_t max_ws = 256;
    // constexpr Size max_ws = 4096;
    return Min(max_ws, n);
    // return Min(MaxPow2(n_samples), max_ws);
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
