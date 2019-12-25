/* file: kdtree_knn_classification_predict_dense_default_batch_impl.i */
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
//  Common functions for K-Nearest Neighbors predictions calculation
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __KDTREE_KNN_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "threading.h"
#include "daal_defines.h"
#include "algorithm.h"
#include "daal_atomic_int.h"
#include "service_memory.h"
#include "service_data_utils.h"
#include "service_math.h"
#include "service_rng.h"
#include "service_sort.h"
#include "numeric_table.h"
#include "kdtree_knn_classification_predict_dense_default_batch.h"
#include "kdtree_knn_classification_model_impl.h"
#include "kdtree_knn_impl.i"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace prediction
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::internal;
using namespace kdtree_knn_classification::internal;

template <CpuType cpu, typename T>
DAAL_FORCEINLINE T heapLeftChildIndex(T index)
{
    return 2 * index + 1;
}
template <CpuType cpu, typename T>
DAAL_FORCEINLINE T heapRightChildIndex(T index)
{
    return 2 * index + 2;
}
template <CpuType cpu, typename T>
DAAL_FORCEINLINE T heapParentIndex(T index)
{
    return (index - 1) / 2;
}

template <CpuType cpu, typename RandomAccessIterator>
void pushMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    if (first != last)
    {
        --last;
        auto i = last - first;
        if (i > 0)
        {
            const auto newItem = *last; // It can be moved instead.
            auto prev          = i;
            for (i = heapParentIndex<cpu>(i); i && (*(first + i) < newItem); i = heapParentIndex<cpu>(i))
            {
                *(first + prev) = *(first + i); // It can be moved instead.
                prev            = i;
            }
            *(first + i) = newItem; // It can be moved instead.
        }
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Diff>
DAAL_FORCEINLINE void internalAdjustMaxHeap(RandomAccessIterator first, RandomAccessIterator /*last*/, Diff count, Diff i)
{
    for (auto largest = i;; i = largest)
    {
        const auto l = heapLeftChildIndex<cpu>(i);
        if ((l < count) && (*(first + largest) < *(first + l)))
        {
            largest = l;
        }
        const auto r = heapRightChildIndex<cpu>(i);
        if ((r < count) && (*(first + largest) < *(first + r)))
        {
            largest = r;
        }

        if (largest == i)
        {
            break;
        }
        auto temp          = *(first + i);
        *(first + i)       = *(first + largest);
        *(first + largest) = temp; // Moving can be used instead.
    }
}

template <CpuType cpu, typename RandomAccessIterator>
void popMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    if (1 < last - first)
    {
        --last;
        auto temp = *first;
        *first    = *last;
        *last     = temp; // Moving can be used instead.
        internalAdjustMaxHeap<cpu>(first, last, last - first, first - first);
    }
}

template <CpuType cpu, typename RandomAccessIterator>
void makeMaxHeap(RandomAccessIterator first, RandomAccessIterator last)
{
    const auto count = last - first;
    auto i           = count / 2;
    while (0 < i)
    {
        internalAdjustMaxHeap<cpu>(first, last, count, --i);
    }
}

template <typename T, CpuType cpu>
class Heap
{
public:
    Heap() : _elements(nullptr), _count(0) {}

    ~Heap()
    {
        services::daal_free(_elements);
        _elements = nullptr;
    }

    bool init(size_t size)
    {
        _count    = 0;
        _elements = static_cast<T *>(daal::services::internal::service_malloc<T, cpu>(size * sizeof(T)));
        return _elements;
    }

    void clear()
    {
        if (_elements)
        {
            services::daal_free(_elements);
            _elements = nullptr;
        }
    }

    void reset() { _count = 0; }

    void push(const T & e, size_t k)
    {
        _elements[_count++] = e;
        if (_count == k)
        {
            makeMaxHeap<cpu>(_elements, _elements + _count);
        }
    }

    void replaceMax(const T & e)
    {
        popMaxHeap<cpu>(_elements, _elements + _count);
        _elements[_count - 1] = e;
        pushMaxHeap<cpu>(_elements, _elements + _count);
    }

    size_t size() const { return _count; }

    T * getMax() { return _elements; }

    const T & operator[](size_t index) const { return *(_elements + index); }

private:
    T * _elements;
    size_t _count;
};

template <typename algorithmFpType, CpuType cpu>
struct GlobalNeighbors
{
    algorithmFpType distance;
    size_t index;

    inline bool operator<(const GlobalNeighbors & rhs) const { return (distance < rhs.distance); }
};

template <typename algorithmFpType>
struct SearchNode
{
    size_t nodeIndex;
    algorithmFpType minDistance;
};

template <typename algorithmFpType, CpuType cpu>
Status KNNClassificationPredictKernel<algorithmFpType, defaultDense, cpu>::compute(const NumericTable * x, const classifier::Model * m,
                                                                                   NumericTable * y, const daal::algorithms::Parameter * par)
{
    Status status;

    typedef GlobalNeighbors<algorithmFpType, cpu> Neighbors;
    typedef Heap<Neighbors, cpu> MaxHeap;
    typedef kdtree_knn_classification::internal::Stack<SearchNode<algorithmFpType>, cpu> SearchStack;
    typedef daal::services::internal::MaxVal<algorithmFpType> MaxVal;
    typedef daal::internal::Math<algorithmFpType, cpu> Math;

    size_t k;
    {
        auto par1 = dynamic_cast<const kdtree_knn_classification::interface1::Parameter *>(par);
        if (par1) k = par1->k;

        auto par2 = dynamic_cast<const kdtree_knn_classification::interface2::Parameter *>(par);
        if (par2) k = par2->k;

        if (par1 == NULL && par2 == NULL) return Status(ErrorNullParameterNotSupported);
    }

    const Model * const model    = static_cast<const Model *>(m);
    const auto & kdTreeTable     = *(model->impl()->getKDTreeTable());
    const auto rootTreeNodeIndex = model->impl()->getRootNodeIndex();
    const NumericTable & data    = *(model->impl()->getData());
    const NumericTable & labels  = *(model->impl()->getLabels());

    size_t iSize = 1;
    while (iSize < k)
    {
        iSize *= 2;
    }
    const size_t heapSize = (iSize / 16 + 1) * 16;

    const size_t xRowCount        = x->getNumberOfRows();
    const algorithmFpType base    = 2.0;
    const size_t expectedMaxDepth = (Math::sLog(xRowCount) / Math::sLog(base) + 1) * __KDTREE_DEPTH_MULTIPLICATION_FACTOR;
    const size_t stackSize        = Math::sPowx(base, Math::sCeil(Math::sLog(expectedMaxDepth) / Math::sLog(base)));
    struct Local
    {
        MaxHeap heap;
        SearchStack stack;
    };
    daal::tls<Local *> localTLS([=, &status]() -> Local * {
        Local * const ptr = service_scalable_calloc<Local, cpu>(1);
        if (ptr)
        {
            if (!ptr->heap.init(heapSize))
            {
                status.add(services::ErrorMemoryAllocationFailed);
                service_scalable_free<Local, cpu>(ptr);
                return nullptr;
            }
            if (!ptr->stack.init(stackSize))
            {
                status.add(services::ErrorMemoryAllocationFailed);
                ptr->heap.clear();
                service_scalable_free<Local, cpu>(ptr);
                return nullptr;
            }
        }
        else
        {
            status.add(services::ErrorMemoryAllocationFailed);
        }
        return ptr;
    });

    DAAL_CHECK_STATUS_OK((status.ok()), status);

    const auto maxThreads     = threader_get_threads_number();
    const size_t xColumnCount = x->getNumberOfColumns();
    const size_t yColumnCount = y->getNumberOfColumns();
    const auto rowsPerBlock   = (xRowCount + maxThreads - 1) / maxThreads;
    const auto blockCount     = (xRowCount + rowsPerBlock - 1) / rowsPerBlock;
    SafeStatus safeStat;
    daal::threader_for(blockCount, blockCount, [=, &localTLS, &kdTreeTable, &data, &labels, &rowsPerBlock, &k, &safeStat](int iBlock) {
        Local * const local = localTLS.local();
        if (local)
        {
            const size_t first = iBlock * rowsPerBlock;
            const size_t last  = min<cpu>(static_cast<decltype(xRowCount)>(first + rowsPerBlock), xRowCount);

            const algorithmFpType radius = MaxVal::get();
            data_management::BlockDescriptor<algorithmFpType> xBD;
            const_cast<NumericTable &>(*x).getBlockOfRows(first, last - first, readOnly, xBD);
            const algorithmFpType * const dx = xBD.getBlockPtr();
            data_management::BlockDescriptor<algorithmFpType> yBD;
            y->getBlockOfRows(first, last - first, writeOnly, yBD);
            auto * const dy = yBD.getBlockPtr();
            for (size_t i = 0; i < last - first; ++i)
            {
                findNearestNeighbors(&dx[i * xColumnCount], local->heap, local->stack, k, radius, kdTreeTable, rootTreeNodeIndex, data);
                services::Status s = predict(dy[i * yColumnCount], local->heap, labels, k);
                DAAL_CHECK_STATUS_THR(s)
            }
            y->releaseBlockOfRows(yBD);
            const_cast<NumericTable &>(*x).releaseBlockOfRows(xBD);
        }
    });

    DAAL_CHECK_SAFE_STATUS()

    localTLS.reduce([=](Local * ptr) -> void {
        if (ptr)
        {
            ptr->stack.clear();
            ptr->heap.clear();
            service_scalable_free<Local, cpu>(ptr);
        }
    });
    return status;
}

template <typename algorithmFpType, CpuType cpu>
void KNNClassificationPredictKernel<algorithmFpType, defaultDense, cpu>::findNearestNeighbors(
    const algorithmFpType * query, Heap<GlobalNeighbors<algorithmFpType, cpu>, cpu> & heap,
    kdtree_knn_classification::internal::Stack<SearchNode<algorithmFpType>, cpu> & stack, size_t k, algorithmFpType radius,
    const KDTreeTable & kdTreeTable, size_t rootTreeNodeIndex, const NumericTable & data)
{
    heap.reset();
    stack.reset();
    GlobalNeighbors<algorithmFpType, cpu> curNeighbor;
    size_t i, j;
    SearchNode<algorithmFpType> cur, toPush;
    const KDTreeNode * node;
    cur.nodeIndex   = rootTreeNodeIndex;
    cur.minDistance = 0;

    const size_t xColumnCount = data.getNumberOfColumns();

    DAAL_ALIGNAS(256) algorithmFpType distance[__KDTREE_LEAF_BUCKET_SIZE + 1];
    size_t start, end;
    algorithmFpType dist, diff, val;

    data_management::BlockDescriptor<algorithmFpType> xBD[2];
    size_t curBDIdx, nextBDIdx;
    for (;;)
    {
        node = static_cast<const KDTreeNode *>(kdTreeTable.getArray()) + cur.nodeIndex;
        if (node->dimension == __KDTREE_NULLDIMENSION)
        {
            start = node->leftIndex;
            end   = node->rightIndex;

            for (i = start; i < end; ++i)
            {
                distance[i - start] = 0;
            }

            curBDIdx  = 0;
            nextBDIdx = 1;
            const_cast<NumericTable &>(data).getBlockOfColumnValues(0, start, end - start, readOnly, xBD[curBDIdx]);
            for (j = 1; j < xColumnCount; ++j)
            {
                const algorithmFpType * const dx = xBD[curBDIdx].getBlockPtr();

                const_cast<NumericTable &>(data).getBlockOfColumnValues(j, start, end - start, readOnly, xBD[nextBDIdx]);
                const algorithmFpType * const nx = xBD[nextBDIdx].getBlockPtr();
                DAAL_PREFETCH_READ_T0(nx);
                DAAL_PREFETCH_READ_T0(nx + 16);

                for (i = 0; i < end - start; ++i)
                {
                    distance[i] += (query[j - 1] - dx[i]) * (query[j - 1] - dx[i]);
                }

                const_cast<NumericTable &>(data).releaseBlockOfColumnValues(xBD[curBDIdx]);

                const auto tempBDIdx = curBDIdx;
                curBDIdx             = nextBDIdx;
                nextBDIdx            = tempBDIdx;
            }
            {
                const algorithmFpType * const dx = xBD[curBDIdx].getBlockPtr();
                for (i = 0; i < end - start; ++i)
                {
                    distance[i] += (query[j - 1] - dx[i]) * (query[j - 1] - dx[i]);
                }
                const_cast<NumericTable &>(data).releaseBlockOfColumnValues(xBD[curBDIdx]);
            }

            for (i = start; i < end; ++i)
            {
                if (distance[i - start] <= radius)
                {
                    curNeighbor.distance = distance[i - start];
                    curNeighbor.index    = i;
                    if (heap.size() < k)
                    {
                        heap.push(curNeighbor, k);

                        if (heap.size() == k)
                        {
                            radius = heap.getMax()->distance;
                        }
                    }
                    else
                    {
                        if (heap.getMax()->distance > curNeighbor.distance)
                        {
                            heap.replaceMax(curNeighbor);
                            radius = heap.getMax()->distance;
                        }
                    }
                }
            }

            if (!stack.empty())
            {
                cur = stack.pop();
                DAAL_PREFETCH_READ_T0(static_cast<const KDTreeNode *>(kdTreeTable.getArray()) + cur.nodeIndex);
            }
            else
            {
                break;
            }
        }
        else
        {
            val  = query[node->dimension];
            diff = val - node->cutPoint;

            if (cur.minDistance <= radius)
            {
                cur.nodeIndex    = (diff < 0) ? node->leftIndex : node->rightIndex;
                toPush.nodeIndex = (diff < 0) ? node->rightIndex : node->leftIndex;
                val -= node->cutPoint;
                toPush.minDistance = cur.minDistance + val * val;
                stack.push(toPush);
            }
            else if (!stack.empty())
            {
                cur = stack.pop();
                DAAL_PREFETCH_READ_T0(static_cast<const KDTreeNode *>(kdTreeTable.getArray()) + cur.nodeIndex);
            }
            else
            {
                break;
            }
        }
    }
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationPredictKernel<algorithmFpType, defaultDense, cpu>::predict(
    algorithmFpType & predictedClass, const Heap<GlobalNeighbors<algorithmFpType, cpu>, cpu> & heap, const NumericTable & labels, size_t k)
{
    const size_t heapSize = heap.size();
    if (heapSize < 1) return services::Status();

    struct Voting
    {
        algorithmFpType predictedClass;
        size_t weight;
    };

    data_management::BlockDescriptor<algorithmFpType> labelBD;
    algorithmFpType * classes =
        static_cast<algorithmFpType *>(daal::services::internal::service_malloc<algorithmFpType, cpu>(heapSize * sizeof(*classes)));
    DAAL_CHECK_MALLOC(classes)
    for (size_t i = 0; i < heapSize; ++i)
    {
        const_cast<NumericTable &>(labels).getBlockOfColumnValues(0, heap[i].index, 1, readOnly, labelBD);
        classes[i] = *(labelBD.getBlockPtr());
        const_cast<NumericTable &>(labels).releaseBlockOfColumnValues(labelBD);
    }
    daal::algorithms::internal::qSort<algorithmFpType, cpu>(heapSize, classes);
    algorithmFpType currentClass = classes[0];
    algorithmFpType winnerClass  = currentClass;
    size_t currentWeight         = 1;
    size_t winnerWeight          = currentWeight;
    for (size_t i = 1; i < heapSize; ++i)
    {
        if (classes[i] == currentClass)
        {
            if ((++currentWeight) > winnerWeight)
            {
                winnerWeight = currentWeight;
                winnerClass  = currentClass;
            }
        }
        else
        {
            currentWeight = 1;
            currentClass  = classes[i];
        }
    }
    predictedClass = winnerClass;
    daal_free(classes);
    classes = nullptr;
    return services::Status();
}

} // namespace internal
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
