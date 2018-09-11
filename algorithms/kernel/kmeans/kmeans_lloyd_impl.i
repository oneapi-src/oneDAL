/* file: kmeans_lloyd_impl.i */
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
//  Implementation of auxiliary functions used in Lloyd method
//  of K-means algorithm.
//--
*/

#include "service_memory.h"
#include "service_numeric_table.h"
#include "service_defines.h"
#include "service_error_handling.h"

#include "threading.h"
#include "service_blas.h"
#include "service_spblas.h"

// CPU intrinsics for Intel Compiler only
#if defined (__INTEL_COMPILER) && defined(__linux__) && defined(__x86_64__)
    #include <immintrin.h>
#endif


namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

template<typename algorithmFPType, CpuType cpu>
struct tls_task_t
{
    DAAL_NEW_DELETE();

    tls_task_t(int dim, int clNum, int max_block_size)
    {
        mkl_buff = service_calloc<algorithmFPType, cpu>(max_block_size * clNum);
        cS1      = service_calloc<algorithmFPType, cpu>(clNum * dim);
        cS0      = service_calloc<int,cpu>(clNum);
        cValues  = service_calloc<algorithmFPType, cpu>(clNum);
        cIndices = service_calloc<size_t,cpu>(clNum);
    }

    ~tls_task_t()
    {
        if (mkl_buff)
        {
            service_free<algorithmFPType, cpu>(mkl_buff);
        }
        if (cS1)
        {
            service_free<algorithmFPType, cpu>(cS1);
        }
        if (cS0)
        {
            service_free<int, cpu>(cS0);
        }
        if (cValues)
        {
            service_free<algorithmFPType, cpu>(cValues);
        }
        if (cIndices)
        {
            service_free<size_t, cpu>(cIndices);
        }
    }

    static tls_task_t<algorithmFPType, cpu>* create(int dim, int clNum, int max_block_size)
    {
        tls_task_t<algorithmFPType, cpu> *result = new tls_task_t<algorithmFPType, cpu>(dim, clNum, max_block_size);
        if (!result)
        {
            return nullptr;
        }
        if (!result->mkl_buff || !result->cS1 || !result->cS0)
        {
            delete result;
            return nullptr;
        }
        return result;
    }

    algorithmFPType *mkl_buff = nullptr;
    algorithmFPType *cS1 = nullptr;
    int    *cS0 = nullptr;
    algorithmFPType goalFunc = 0.0;
    size_t cNum = 0;
    algorithmFPType *cValues = nullptr;
    size_t *cIndices = nullptr;
};

template<typename algorithmFPType>
struct Fp2IntSize {};
template<> struct Fp2IntSize<float>  { typedef int IntT;     };
template<> struct Fp2IntSize<double> { typedef __int64 IntT; };

template<typename algorithmFPType, CpuType cpu>
struct task_t
{
    DAAL_NEW_DELETE();

    task_t(int _dim, int _clNum, algorithmFPType *_centroids)
    {
        dim       = _dim;
        clNum     = _clNum;
        cCenters  = _centroids;
        max_block_size = 512;

        /* Allocate memory for all arrays inside TLS */
        tls_task = new daal::tls<tls_task_t<algorithmFPType, cpu>*>([=]()-> tls_task_t<algorithmFPType, cpu> *
        {
            return tls_task_t<algorithmFPType, cpu>::create(dim, clNum, max_block_size);
        } ); /* Allocate memory for all arrays inside TLS: end */

        clSq = service_calloc<algorithmFPType, cpu>(clNum);
        if(clSq)
        {
            for(size_t k=0;k<clNum;k++)
            {
                for(size_t j=0;j<dim;j++)
                {
                    clSq[k] += cCenters[k*dim + j]*cCenters[k*dim + j] * 0.5;
                }
            }
        }
    }

    ~task_t()
    {
        if (tls_task)
        {
            tls_task->reduce( [ = ](tls_task_t<algorithmFPType, cpu> *tt)-> void
            {
                delete tt;
            } );
            delete tls_task;
        }
        if (clSq)
        {
            service_free<algorithmFPType, cpu>( clSq );
        }
    }

    static SharedPtr<task_t<algorithmFPType, cpu> > create(int dim, int clNum, algorithmFPType *centroids)
    {
        SharedPtr<task_t<algorithmFPType, cpu> > result(new task_t<algorithmFPType, cpu>(dim, clNum, centroids));
        if (result.get() && (!result->tls_task || !result->clSq))
        {
            result.reset();
        }
        return result;
    }

    Status addNTToTaskThreadedDense(const NumericTable *ntData, algorithmFPType *catCoef, NumericTable *ntAssign = 0);

    Status addNTToTaskThreadedCSR(const NumericTable *ntDataGen, algorithmFPType *catCoef, NumericTable *ntAssign = 0);

    template<Method method>
    Status getNTAssignmentsThreaded(const NumericTable *ntData, const NumericTable *ntAssign, algorithmFPType *catCoef);

    template<Method method>
    Status addNTToTaskThreaded(const NumericTable *ntData, algorithmFPType *catCoef, NumericTable *ntAssign = 0);

    template<typename centroidsFPType>
    int kmeansUpdateCluster(int jidx, centroidsFPType *s1);

    template<Method method>
    void kmeansComputeCentroids(int *clusterS0, algorithmFPType *clusterS1, double *auxData);

    void kmeansInsertCandidate(tls_task_t<algorithmFPType, cpu> *tt, algorithmFPType value, size_t index);

    Status kmeansComputeCentroidsCandidates(algorithmFPType *cValues, size_t *cIndices, size_t &cNum);

    void kmeansClearClusters(algorithmFPType *goalFunc);

    daal::tls<tls_task_t<algorithmFPType, cpu>*> *tls_task;
    algorithmFPType *clSq;
    algorithmFPType *cCenters;

    int      dim;
    int      clNum;
    int      max_block_size;

    typedef typename Fp2IntSize<algorithmFPType>::IntT algIntType;
};

template<typename algorithmFPType, CpuType cpu>
Status task_t<algorithmFPType, cpu>::addNTToTaskThreadedDense(const NumericTable *ntData, algorithmFPType *catCoef, NumericTable *ntAssign)
{
    const size_t n = ntData->getNumberOfRows();

    const size_t blockSizeDeafult = max_block_size;

    size_t nBlocks = n / blockSizeDeafult;
    nBlocks += (nBlocks*blockSizeDeafult != n);

    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [=, &safeStat](int k)
    {
        struct tls_task_t<algorithmFPType, cpu> *tt = tls_task->local();
        DAAL_CHECK_MALLOC_THR(tt);
        size_t blockSize = blockSizeDeafult;
        if( k == nBlocks-1 )
        {
            blockSize = n - k*blockSizeDeafult;
        }

        ReadRows<algorithmFPType, cpu> mtData(*const_cast<NumericTable *>(ntData), k*blockSizeDeafult, blockSize);
        DAAL_CHECK_BLOCK_STATUS_THR(mtData);
        const algorithmFPType *data = mtData.get();

        size_t p           = dim;
        size_t nClusters   = clNum;
        algorithmFPType *inClusters = cCenters;
        algorithmFPType *clustersSq = clSq;
        int    *cS0        = tt->cS0;
        algorithmFPType *cS1        = tt->cS1;
        algorithmFPType *trg        = &(tt->goalFunc);
        algorithmFPType *x_clusters = tt->mkl_buff;

        WriteOnlyRows<int, cpu> assignBlock(ntAssign ? const_cast<NumericTable *>(ntAssign) : nullptr, k*blockSizeDeafult, blockSize);
        int* assignments = nullptr;
        if(ntAssign)
        {
            DAAL_CHECK_BLOCK_STATUS_THR(assignBlock);
            assignments = assignBlock.get();
        }

        char transa = 't';
        char transb = 'n';
        DAAL_INT _m = blockSize;
        DAAL_INT _n = nClusters;
        DAAL_INT _k = p;
        algorithmFPType alpha = -1.0;
        DAAL_INT lda = p;
        DAAL_INT ldy = p;
        algorithmFPType beta = 1.0;
        DAAL_INT ldaty = blockSize;

      PRAGMA_IVDEP
        for (size_t j = 0; j < nClusters; j++)
        {
            for (size_t i = 0; i < blockSize; i++)
            {
                x_clusters[i + j*blockSize] = clustersSq[j];
            }
        }

        Blas<algorithmFPType, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, data,
                                           &lda, inClusters, &ldy, &beta, x_clusters, &ldaty);

      PRAGMA_ICC_OMP(simd simdlen(16))
        for (algIntType i = 0; i < (algIntType)blockSize; i++)
        {
            algorithmFPType minGoalVal = x_clusters[i];
            algIntType minIdx = 0;

            for (algIntType j = 0; j < (algIntType)nClusters; j++)
            {
                algorithmFPType localGoalVal = x_clusters[i + j*blockSize];
                if( minGoalVal > localGoalVal )
                {
                    minGoalVal = localGoalVal;
                    minIdx = j;
                }
            }

            minGoalVal *= 2.0;

            *((algIntType*)&(x_clusters[i])) = minIdx;
            x_clusters[i+blockSize] = minGoalVal;
        }

        algorithmFPType goal = (algorithmFPType)0;
        for (size_t i = 0; i < blockSize; i++)
        {
            size_t minIdx = *((algIntType*)&(x_clusters[i]));
            algorithmFPType minGoalVal = x_clusters[i+blockSize];

          PRAGMA_ICC_NO16(omp simd reduction(+:minGoalVal))
            for (size_t j = 0; j < p; j++)
            {
                cS1[minIdx * p + j] += data[i*p + j];
                minGoalVal += data[ i*p + j ] * data[ i*p + j ];
            }

            kmeansInsertCandidate(tt, minGoalVal, k * blockSizeDeafult + i);

            cS0[minIdx]++;

            goal += minGoalVal;

            if(ntAssign)
            {
                assignments[i] = (int)minIdx;
            }
        } /* for (size_t i = 0; i < blockSize; i++) */

        *trg  += goal;
    } ); /* daal::threader_for( nBlocks, nBlocks, [=](int k) */
    return safeStat.detach();
}

template<typename algorithmFPType, CpuType cpu>
Status task_t<algorithmFPType, cpu>::addNTToTaskThreadedCSR(const NumericTable *ntDataGen, algorithmFPType *catCoef, NumericTable *ntAssign)
{
    CSRNumericTableIface *ntData  = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(ntDataGen));

    size_t n = ntDataGen->getNumberOfRows();

    size_t blockSizeDeafult = max_block_size;

    size_t nBlocks = n / blockSizeDeafult;
    nBlocks += (nBlocks*blockSizeDeafult != n);

    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [=, &safeStat](int k)
    {
        struct tls_task_t<algorithmFPType, cpu> *tt = tls_task->local();
        DAAL_CHECK_MALLOC_THR(tt);
        size_t blockSize = blockSizeDeafult;
        if( k == nBlocks-1 )
        {
            blockSize = n - k*blockSizeDeafult;
        }

        ReadRowsCSR<algorithmFPType, cpu> dataBlock(ntData, k*blockSizeDeafult, blockSize);
        DAAL_CHECK_BLOCK_STATUS_THR(dataBlock);

        const algorithmFPType *data  = dataBlock.values();
        const size_t *colIdx = dataBlock.cols();
        const size_t *rowIdx = dataBlock.rows();

        size_t p           = dim;
        size_t nClusters   = clNum;
        algorithmFPType *inClusters = cCenters;
        algorithmFPType *clustersSq = clSq;
        int    *cS0        = tt->cS0;
        algorithmFPType *cS1        = tt->cS1;
        algorithmFPType *trg        = &(tt->goalFunc);
        algorithmFPType *x_clusters = tt->mkl_buff;

        WriteOnlyRows<int, cpu> assignBlock(ntAssign, k*blockSizeDeafult, blockSize);
        int* assignments = nullptr;
        if(ntAssign)
        {
            DAAL_CHECK_BLOCK_STATUS_THR(assignBlock);
            assignments = assignBlock.get();
        }
        {
            char transa = 'n';
            DAAL_INT _n = blockSize;
            DAAL_INT _p = p;
            DAAL_INT _c = nClusters;
            algorithmFPType alpha = 1.0;
            algorithmFPType beta  = 0.0;
            DAAL_INT ldaty = blockSize;
            char matdescra[6] = {'G',0,0,'F',0,0};

            SpBlas<algorithmFPType, cpu>::xxcsrmm(&transa, &_n, &_c, &_p, &alpha, matdescra,
                                                  data, (DAAL_INT *)colIdx, (DAAL_INT *)rowIdx,
                                                  inClusters, &_p, &beta, x_clusters, &_n);
        }

        size_t csrCursor=0;
        for (size_t i = 0; i < blockSize; i++)
        {
            algorithmFPType minGoalVal = clustersSq[0] - x_clusters[i];
            size_t minIdx = 0;

            for (size_t j = 0; j < nClusters; j++)
            {
                if( minGoalVal > clustersSq[j] - x_clusters[i + j*blockSize] )
                {
                    minGoalVal = clustersSq[j] - x_clusters[i + j*blockSize];
                    minIdx = j;
                }
            }

            minGoalVal *= 2.0;

            size_t valuesNum = rowIdx[i+1]-rowIdx[i];
            for (size_t j = 0; j < valuesNum; j++)
            {
                cS1[minIdx * p + colIdx[csrCursor]-1] += data[csrCursor];
                minGoalVal += data[csrCursor]*data[csrCursor];
                csrCursor++;
            }

            kmeansInsertCandidate(tt, minGoalVal, k * blockSizeDeafult + i);

            *trg += minGoalVal;

            cS0[minIdx]++;

            if(ntAssign)
            {
                assignments[i] = (int)minIdx;
            }
        }

    } );
    return safeStat.detach();
}

template<typename algorithmFPType, CpuType cpu>
template<Method method>
Status task_t<algorithmFPType, cpu>::addNTToTaskThreaded(const NumericTable *ntData, algorithmFPType *catCoef, NumericTable *ntAssign)
{
    if(method == lloydDense)
    {
        return addNTToTaskThreadedDense( ntData, catCoef, ntAssign );
    }
    else if(method == lloydCSR)
    {
        return addNTToTaskThreadedCSR( ntData, catCoef, ntAssign );
    }
    DAAL_ASSERT(false);
    return Status();
}

template<typename algorithmFPType, CpuType cpu>
template<Method method>
Status task_t<algorithmFPType, cpu>::getNTAssignmentsThreaded(const NumericTable *ntData, const NumericTable *ntAssign, algorithmFPType *catCoef)
{
    const size_t n = ntData->getNumberOfRows();

    size_t blockSizeDeafult = max_block_size;

    size_t nBlocks = n / blockSizeDeafult;
    nBlocks += (nBlocks*blockSizeDeafult != n);

    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [=, &safeStat](int k)
    {
        struct tls_task_t<algorithmFPType, cpu> *tt = tls_task->local();
        DAAL_CHECK_MALLOC_THR(tt);
        size_t blockSize = blockSizeDeafult;
        if( k == nBlocks-1 )
        {
            blockSize = n - k*blockSizeDeafult;
        }

        ReadRows<algorithmFPType, cpu> mtData(*const_cast<NumericTable*>(ntData), k*blockSizeDeafult, blockSize);
        DAAL_CHECK_BLOCK_STATUS_THR(mtData);
        WriteOnlyRows<int, cpu> mtAssign(*const_cast<NumericTable*>(ntAssign), k*blockSizeDeafult, blockSize);
        DAAL_CHECK_BLOCK_STATUS_THR(mtAssign);
        const algorithmFPType *data = mtData.get();
        int *assign = mtAssign.get();

        size_t p           = dim;
        size_t nClusters   = clNum;
        algorithmFPType *inClusters = cCenters;
        algorithmFPType *clustersSq = clSq;
        algorithmFPType *x_clusters = tt->mkl_buff;

        char transa = 't';
        char transb = 'n';
        DAAL_INT _m = nClusters;
        DAAL_INT _n = blockSize;
        DAAL_INT _k = p;
        algorithmFPType alpha = 1.0;
        DAAL_INT lda = p;
        DAAL_INT ldy = p;
        algorithmFPType beta = 0.0;
        DAAL_INT ldaty = nClusters;

        Blas<algorithmFPType, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, inClusters,
                                           &lda, data, &ldy, &beta, x_clusters, &ldaty);

        for (size_t i = 0; i < blockSize; i++)
        {
            algorithmFPType minGoalVal = clustersSq[0] - x_clusters[i * nClusters];
            size_t minIdx = 0;

            for (size_t j = 0; j < nClusters; j++)
            {
                if( minGoalVal > clustersSq[j] - x_clusters[i*nClusters + j] )
                {
                    minGoalVal = clustersSq[j] - x_clusters[i*nClusters + j];
                    minIdx = j;
                }
            }

            assign[i] = minIdx;
        }

    } );
    return safeStat.detach();
}

template<typename algorithmFPType, CpuType cpu>
template<typename centroidsFPType>
int task_t<algorithmFPType, cpu>::kmeansUpdateCluster(int jidx, centroidsFPType *s1)
{
    int i, j;

    int idx   = (int)jidx;

    int s0=0;

    tls_task->reduce( [&](tls_task_t<algorithmFPType, cpu> *tt)-> void
    {
        s0 += tt->cS0[idx];
    } );

    tls_task->reduce( [ = ](tls_task_t<algorithmFPType, cpu> *tt)-> void
    {
        int j;
      PRAGMA_IVDEP
        for(j=0;j<dim;j++)
        {
            s1[j] += tt->cS1[idx*dim + j];
        }
    } );

    return s0;
}

template<typename algorithmFPType, CpuType cpu>
template<Method method>
void task_t<algorithmFPType, cpu>::kmeansComputeCentroids(int *clusterS0, algorithmFPType *clusterS1, double *auxData)
{
    if (method == defaultDense && auxData)
    {
        for (size_t i = 0; i < clNum; i++)
        {
            for (size_t j = 0; j < dim; j++)
            {
                auxData[j] = 0.0;
            }

            clusterS0[i] = kmeansUpdateCluster<double>( i, auxData );

            for (size_t j = 0; j < dim; j++)
            {
                clusterS1[i * dim + j] = auxData[j];
            }
        }
    }
    else
    {
        for (size_t i = 0; i < clNum; i++)
        {
            for (size_t j = 0; j < dim; j++)
            {
                clusterS1[i * dim + j] = 0.0;
            }

            clusterS0[i] = kmeansUpdateCluster<algorithmFPType>( i, &clusterS1[i * dim] );
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
void task_t<algorithmFPType, cpu>::kmeansInsertCandidate(tls_task_t<algorithmFPType, cpu> *tt, algorithmFPType value, size_t index)
{
    size_t cPos = tt->cNum;
    while (cPos > 0 && tt->cValues[cPos - 1] < value)
    {
        if (cPos < clNum)
        {
            tt->cValues[cPos] = tt->cValues[cPos - 1];
            tt->cIndices[cPos] = tt->cIndices[cPos - 1];
        }
        cPos--;
    }

    if (cPos < clNum)
    {
        tt->cValues[cPos] = value;
        tt->cIndices[cPos] = index;
        if (tt->cNum < clNum)
        {
            tt->cNum++;
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
Status task_t<algorithmFPType, cpu>::kmeansComputeCentroidsCandidates(algorithmFPType *cValues, size_t *cIndices, size_t &cNum)
{
    cNum = 0;

    TArray<algorithmFPType, cpu> tmpValues(clNum);
    TArray<size_t, cpu> tmpIndices(clNum);
    DAAL_CHECK_MALLOC(tmpValues.get() && tmpIndices.get());

    algorithmFPType *tmpValuesPtr = tmpValues.get();
    size_t *tmpIndicesPtr = tmpIndices.get();

    tls_task->reduce( [&](tls_task_t<algorithmFPType, cpu> *tt)-> void
    {
        size_t lcNum = tt->cNum;
        algorithmFPType *lcValues = tt->cValues;
        size_t *lcIndices = tt->cIndices;

        size_t cPos = 0;
        size_t lcPos = 0;

        while (cPos + lcPos < clNum && (cPos < cNum || lcPos < lcNum))
        {
            if (cPos < cNum && (lcPos == lcNum || cValues[cPos] > lcValues[lcPos]))
            {
                tmpValuesPtr[cPos + lcPos] = cValues[cPos];
                tmpIndicesPtr[cPos + lcPos] = cIndices[cPos];
                cPos++;
            }
            else
            {
                tmpValuesPtr[cPos + lcPos] = lcValues[lcPos];
                tmpIndicesPtr[cPos + lcPos] = lcIndices[lcPos];
                lcPos++;
            }
        }
        cNum = cPos + lcPos;
        daal::services::daal_memcpy_s(cValues, cNum * sizeof(algorithmFPType), tmpValuesPtr, cNum * sizeof(algorithmFPType));
        daal::services::daal_memcpy_s(cIndices, cNum * sizeof(size_t), tmpIndicesPtr, cNum * sizeof(size_t));
    } );

    return services::Status();
}

template<typename algorithmFPType, CpuType cpu>
void task_t<algorithmFPType, cpu>::kmeansClearClusters(algorithmFPType *goalFunc)
{
    int i, j;

    if( clNum != 0)
    {
        clNum = 0;

        if( goalFunc != 0 )
        {
            *goalFunc = (algorithmFPType)(0.0);

            tls_task->reduce( [ = ](tls_task_t<algorithmFPType, cpu> *tt)-> void
            {
                (*goalFunc) += tt->goalFunc;
            } );
        }
    }
}

} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
