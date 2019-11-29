/* file: kdtree_knn_classification_predict_dense_default_batch.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Declaration of template function that computes K-Nearest Neighbors prediction results.
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __KDTREE_KNN_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "kdtree_knn_classification_predict.h"
#include "kdtree_knn_classification_model_impl.h"
#include "service_memory.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_blas.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace internal
{
template <typename T, CpuType cpu>
class Stack;
} // namespace internal

namespace prediction
{
namespace internal
{
using namespace daal::data_management;

template <typename algorithmFpType, CpuType cpu>
struct GlobalNeighbors;
template <typename T, CpuType cpu>
class Heap;
template <typename algorithmFpType>
struct SearchNode;

template <typename algorithmFpType, prediction::Method method, CpuType cpu>
class KNNClassificationPredictKernel : public daal::algorithms::Kernel
{};

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationPredictKernel<algorithmFpType, defaultDense, cpu> : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable * x, const classifier::Model * m, NumericTable * y, const daal::algorithms::Parameter * par);

protected:
    void findNearestNeighbors(const algorithmFpType * query, Heap<GlobalNeighbors<algorithmFpType, cpu>, cpu> & heap,
                              kdtree_knn_classification::internal::Stack<SearchNode<algorithmFpType>, cpu> & stack, size_t k, algorithmFpType radius,
                              const KDTreeTable & kdTreeTable, size_t rootTreeNodeIndex, const NumericTable & data);

    services::Status predict(algorithmFpType & predictedClass, const Heap<GlobalNeighbors<algorithmFpType, cpu>, cpu> & heap,
                             const NumericTable & labels, size_t k);
};

} // namespace internal
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
