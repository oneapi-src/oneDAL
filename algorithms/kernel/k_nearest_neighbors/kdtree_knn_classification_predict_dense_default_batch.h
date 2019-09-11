/* file: kdtree_knn_classification_predict_dense_default_batch.h */
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
template <typename T, CpuType cpu> class Stack;
} // namespace internal

namespace prediction
{
namespace internal
{

using namespace daal::data_management;

template <typename algorithmFpType, CpuType cpu> struct GlobalNeighbors;
template <typename T, CpuType cpu> class Heap;
template <typename algorithmFpType> struct SearchNode;

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

    void predict(algorithmFpType & predictedClass, const Heap<GlobalNeighbors<algorithmFpType, cpu>, cpu> & heap, const NumericTable & labels,
                 size_t k);
};

} // namespace internal
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
