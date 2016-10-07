/* file: kmeans_lloyd_impl.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of auxiliary functions used in Lloyd method
//  of K-means algorithm.
//--
*/

#include "service_memory.h"
#include "service_micro_table.h"

#include "threading.h"
#include "service_blas.h"
#include "service_spblas.h"
#include "service_defines.h"


// CPU intrinsics for Intel Compiler only
#if defined (__INTEL_COMPILER) && defined(__linux__) && defined(__x86_64__)
    #include <immintrin.h>
#endif


using namespace daal::services::internal;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{


template<typename algorithmFPType, CpuType cpu>
struct tls_task_t
{
    algorithmFPType *mkl_buff;
    algorithmFPType *cS1;
    int    *cS0;
    algorithmFPType goalFunc;
};

template<typename algorithmFPType, CpuType cpu>
struct task_t
{
    daal::tls<tls_task_t<algorithmFPType, cpu>*> *tls_task;
    algorithmFPType *clSq;
    algorithmFPType *cCenters;

    int      dim;
    int      clNum;
    int      max_block_size;

};

template<typename algorithmFPType, CpuType cpu>
void *kmeansInitTask(int dim, int clNum, algorithmFPType *centroids,
                     services::SharedPtr<services::KernelErrorCollection> &_errors)
{
    struct task_t<algorithmFPType, cpu> *t;
    t = (task_t<algorithmFPType, cpu> *)daal::services::daal_malloc(sizeof(struct task_t<algorithmFPType, cpu>));
    if(!t)
    {
        _errors->add(services::ErrorMemoryAllocationFailed);
        return 0;
    }

    t->dim       = dim;
    t->clNum     = clNum;
    t->cCenters  = centroids;
    t->max_block_size = 512;

    /* Allocate memory for all arrays inside TLS */
    t->tls_task = new daal::tls<tls_task_t<algorithmFPType, cpu>*>( [ = ]()-> tls_task_t<algorithmFPType, cpu> *
    {
        tls_task_t<algorithmFPType, cpu> *tt = new tls_task_t<algorithmFPType, cpu>;
        if(!tt)
        {
            _errors->add(services::ErrorMemoryAllocationFailed);
            return 0;
        }

        tt->mkl_buff = service_calloc<algorithmFPType, cpu>(t->max_block_size * t->clNum);
        if(!tt->mkl_buff)
        {
            _errors->add(services::ErrorMemoryAllocationFailed);
            delete tt;
            return 0;
        }

        tt->cS1      = service_calloc<algorithmFPType, cpu>(t->clNum * t->dim);
        if(!tt->cS1)
        {
            _errors->add(services::ErrorMemoryAllocationFailed);
            service_free<algorithmFPType, cpu>(tt->mkl_buff);
            delete tt;
            return 0;
        }

        tt->cS0      = service_calloc<int,cpu>(t->clNum);
        if(!tt->cS0)
        {
            service_free<algorithmFPType, cpu>(tt->mkl_buff);
            service_free<algorithmFPType, cpu>(tt->cS1);
            _errors->add(services::ErrorMemoryAllocationFailed);
            delete tt;
            return 0;
        }

        tt->goalFunc = (algorithmFPType)(0.0);

        return tt;
    } ); /* Allocate memory for all arrays inside TLS: end */

    if(!t->tls_task)
    {
        _errors->add(services::ErrorMemoryAllocationFailed);
        daal::services::daal_free(t);
        return 0;
    }

    t->clSq      = service_calloc<algorithmFPType, cpu>(clNum);
    if(!t->clSq)
    {
        _errors->add(services::ErrorMemoryAllocationFailed);
        daal::services::daal_free(t);
        return 0;
    }

    for(size_t k=0;k<clNum;k++)
    {
        for(size_t j=0;j<dim;j++)
        {
            t->clSq[k] += centroids[k*dim + j]*centroids[k*dim + j] * 0.5;
        }
    }

    void * task_id;
    *(size_t*)(&task_id) = (size_t)t;

    return task_id;
}

template<typename algorithmFPType, CpuType cpu, int assignFlag>
void addNTToTaskThreadedDense(void *task_id, const NumericTable *ntData, algorithmFPType *catCoef, NumericTable *ntAssign = 0 )
{
    struct task_t<algorithmFPType, cpu> *t  = static_cast<task_t<algorithmFPType, cpu> *>(task_id);

    size_t n = ntData->getNumberOfRows();

    size_t blockSizeDeafult = t->max_block_size;

    size_t nBlocks = n / blockSizeDeafult;
    nBlocks += (nBlocks*blockSizeDeafult != n);

    daal::threader_for( nBlocks, nBlocks, [=](int k)
    {
        struct tls_task_t<algorithmFPType, cpu> *tt = t->tls_task->local();
        size_t blockSize = blockSizeDeafult;
        if( k == nBlocks-1 )
        {
            blockSize = n - k*blockSizeDeafult;
        }

        BlockDescriptor<int> assignBlock;

        BlockMicroTable<algorithmFPType, readOnly,  cpu> mtData( ntData );
        algorithmFPType *data;

        size_t p           = t->dim;
        size_t nClusters   = t->clNum;
        algorithmFPType *inClusters = t->cCenters;
        algorithmFPType *clustersSq = t->clSq;
        int    *cS0        = tt->cS0;
        algorithmFPType *cS1        = tt->cS1;
        algorithmFPType *trg        = &(tt->goalFunc);
        algorithmFPType *x_clusters = tt->mkl_buff;

        mtData.getBlockOfRows( k*blockSizeDeafult, blockSize, &data );

        int* assignments = 0;

        if(assignFlag)
        {
            ntAssign->getBlockOfRows( k*blockSizeDeafult, blockSize, writeOnly, assignBlock );
            assignments = assignBlock.getBlockPtr();
        }

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

            minGoalVal *= 2.0;

            for (size_t j = 0; j < p; j++)
            {
                cS1[minIdx * p + j] += data[i*p + j];
                minGoalVal += data[ i*p + j ] * data[ i*p + j ];
            }

            *trg += minGoalVal;

            cS0[minIdx]++;

            if(assignFlag)
            {
                assignments[i] = (int)minIdx;
            }
        } /* for (size_t i = 0; i < blockSize; i++) */

        if(assignFlag)
        {
            ntAssign->releaseBlockOfRows( assignBlock );
        }

        mtData.release();

    } ); /* daal::threader_for( nBlocks, nBlocks, [=](int k) */
}

template<typename algorithmFPType, CpuType cpu, int assignFlag>
void addNTToTaskThreadedCSR(void *task_id, const NumericTable *ntDataGen, algorithmFPType *catCoef, NumericTable *ntAssign = 0 )
{
    CSRNumericTableIface *ntData  = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(ntDataGen));

    struct task_t<algorithmFPType, cpu> *t  = static_cast<task_t<algorithmFPType, cpu> *>(task_id);

    size_t n = ntDataGen->getNumberOfRows();

    size_t blockSizeDeafult = t->max_block_size;

    size_t nBlocks = n / blockSizeDeafult;
    nBlocks += (nBlocks*blockSizeDeafult != n);

    daal::threader_for( nBlocks, nBlocks, [=](int k)
    {
        struct tls_task_t<algorithmFPType, cpu> *tt = t->tls_task->local();
        size_t blockSize = blockSizeDeafult;
        if( k == nBlocks-1 )
        {
            blockSize = n - k*blockSizeDeafult;
        }

        BlockDescriptor<int> assignBlock;
        CSRBlockDescriptor<algorithmFPType> dataBlock;

        ntData->getSparseBlock( k*blockSizeDeafult, blockSize, readOnly, dataBlock );

        algorithmFPType *data        = dataBlock.getBlockValuesPtr();
        size_t *colIdx      = dataBlock.getBlockColumnIndicesPtr();
        size_t *rowIdx      = dataBlock.getBlockRowIndicesPtr();

        size_t p           = t->dim;
        size_t nClusters   = t->clNum;
        algorithmFPType *inClusters = t->cCenters;
        algorithmFPType *clustersSq = t->clSq;
        int    *cS0        = tt->cS0;
        algorithmFPType *cS1        = tt->cS1;
        algorithmFPType *trg        = &(tt->goalFunc);
        algorithmFPType *x_clusters = tt->mkl_buff;

        int* assignments = 0;

        if(assignFlag)
        {
            ntAssign->getBlockOfRows( k*blockSizeDeafult, blockSize, writeOnly, assignBlock );
            assignments = assignBlock.getBlockPtr();
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

            *trg += minGoalVal;

            cS0[minIdx]++;

            if(assignFlag)
            {
                assignments[i] = (int)minIdx;
            }
        }

        if(assignFlag)
        {
            ntAssign->releaseBlockOfRows( assignBlock );
        }

        ntData->releaseSparseBlock(dataBlock);
    } );
}

template<Method method, typename algorithmFPType, CpuType cpu, int assignFlag>
void addNTToTaskThreaded(void *task_id, const NumericTable *ntData, algorithmFPType *catCoef, NumericTable *ntAssign = 0 )
{
    if(method == lloydDense)
    {
        addNTToTaskThreadedDense<algorithmFPType, cpu, assignFlag>( task_id, ntData, catCoef, ntAssign );
    }
    else if(method == lloydCSR)
    {
        addNTToTaskThreadedCSR<algorithmFPType, cpu, assignFlag>( task_id, ntData, catCoef, ntAssign );
    }
}

template<Method method, typename algorithmFPType, CpuType cpu>
void getNTAssignmentsThreaded(void *task_id, const NumericTable *ntData, const NumericTable *ntAssign, algorithmFPType *catCoef )
{
    struct task_t<algorithmFPType, cpu> *t  = static_cast<task_t<algorithmFPType, cpu> *>(task_id);

    size_t n = ntData->getNumberOfRows();

    size_t blockSizeDeafult = t->max_block_size;

    size_t nBlocks = n / blockSizeDeafult;
    nBlocks += (nBlocks*blockSizeDeafult != n);

    daal::threader_for( nBlocks, nBlocks, [=](int k)
    {
        struct tls_task_t<algorithmFPType, cpu> *tt = t->tls_task->local();
        size_t blockSize = blockSizeDeafult;
        if( k == nBlocks-1 )
        {
            blockSize = n - k*blockSizeDeafult;
        }

        BlockMicroTable<algorithmFPType, readOnly,  cpu> mtData( ntData );
        BlockMicroTable<int   , writeOnly, cpu> mtAssign( ntAssign );
        algorithmFPType *data;
        int    *assign;

        mtData  .getBlockOfRows( k*blockSizeDeafult, blockSize, &data   );
        mtAssign.getBlockOfRows( k*blockSizeDeafult, blockSize, &assign );

        size_t p           = t->dim;
        size_t nClusters   = t->clNum;
        algorithmFPType *inClusters = t->cCenters;
        algorithmFPType *clustersSq = t->clSq;
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

        mtAssign.release();
        mtData.release();
    } );
}

template<typename algorithmFPType, CpuType cpu>
int kmeansUpdateCluster(void *task_id, int jidx, algorithmFPType *s1)
{
    int i, j;
    struct task_t<algorithmFPType, cpu> *t = static_cast<task_t<algorithmFPType, cpu> *>(task_id);

    int idx   = (int)jidx;
    int dim   = t->dim;
    int clNum = t->clNum;

    int s0=0;

    t->tls_task->reduce( [&](tls_task_t<algorithmFPType, cpu> *tt)-> void
    {
        s0 += tt->cS0[idx];
    } );

    t->tls_task->reduce( [ = ](tls_task_t<algorithmFPType, cpu> *tt)-> void
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
void kmeansClearClusters(void *task_id, algorithmFPType *goalFunc)
{
    int i, j;
    struct task_t<algorithmFPType, cpu> *t = static_cast<task_t<algorithmFPType, cpu> *>(task_id);

    if( t->clNum != 0)
    {
        t->clNum = 0;

        if( goalFunc!= 0 )
        {
            *goalFunc = (algorithmFPType)(0.0);

            t->tls_task->reduce( [ = ](tls_task_t<algorithmFPType, cpu> *tt)-> void
            {
                (*goalFunc) += tt->goalFunc;
            } );
        }

        t->tls_task->reduce( [ = ](tls_task_t<algorithmFPType, cpu> *tt)-> void
        {
            service_free<int, cpu>( tt->cS0 );
            service_free<algorithmFPType, cpu>( tt->cS1 );
            service_free<algorithmFPType, cpu>( tt->mkl_buff );
            delete tt;
        } );
        delete t->tls_task;

        service_free<algorithmFPType, cpu>( t->clSq );

    }

    daal::services::daal_free(t);
}

// AVX512-MIC optimization via template specialization (Intel compiler only)
#if defined (__INTEL_COMPILER) && defined(__linux__) && defined(__x86_64__) && ( __CPUID__(DAAL_CPU) == __avx512_mic__ )
    #include "kmeans_lloyd_impl_avx512_mic.i"
#endif

} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
