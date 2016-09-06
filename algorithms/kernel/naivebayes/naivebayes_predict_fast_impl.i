/* file: naivebayes_predict_fast_impl.i */
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
//  Implementation of Fast method for assign K-means algorithm.
//
//  Based on paper: Tackling the Poor Assumptions of Naive Bayes Text Classifiers
//  Url: http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
//--
*/

#include "algorithm.h"
#include "numeric_table.h"
#include "threading.h"
#include "daal_defines.h"
#include "service_memory.h"
#include "csr_numeric_table.h"
#include "service_data_utils.h"
#include "service_blas.h"
#include "service_spblas.h"

#if( __CPUID__(DAAL_CPU) == __avx512_mic__ )

    #if( __FPTYPE__(DAAL_FPTYPE) == __double__ )

        #define _BLOCKSIZE_   (((p <= 100) && (p < c))?((p<=20)?32:64):256)

    #else /* float */

        #define _BLOCKSIZE_   ((c > 100)?128:256)

    #endif

    #define _CALLOC_      service_scalable_calloc
    #define _FREE_        service_scalable_free

#else

    #define _BLOCKSIZE_   (128)
    #define _CALLOC_      service_calloc
    #define _FREE_        service_free

#endif


using namespace daal::services::internal;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace prediction
{
namespace internal
{

template<Method method, typename interm, CpuType cpu>
struct methodSpecific {};

template<typename interm, CpuType cpu>
struct methodSpecific<defaultDense, interm, cpu>
{
    static void getPredictionData(
        interm* aux_table, NumericTable * ntData,
        size_t n0, size_t n, size_t p, size_t c, int* classes, interm* buff );
};

template<typename interm, CpuType cpu>
struct methodSpecific<fastCSR, interm, cpu>
{
    static void getPredictionData(
        interm* aux_table, NumericTable * ntData,
        size_t n0, size_t n, size_t p, size_t c, int* classes, interm* buff );
};

template<typename interm, Method method, CpuType cpu>
void NaiveBayesPredictKernel<interm, method, cpu>::compute(const NumericTable *a, const daal::algorithms::Model *m,
                                                           size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    int p, n, c;

    NumericTable *ntClass = r[0];
    Model *mdl = static_cast<Model *>(const_cast<daal::algorithms::Model *>(m));

    NumericTable *ntData = const_cast<NumericTable *>( a );

    const Parameter *nbPar = static_cast<const Parameter *>(par);

    p = ntData->getNumberOfColumns();
    n = ntData->getNumberOfRows();
    c = nbPar->nClasses;

    NumericTable *ntAuxTable = mdl->getAuxTable().get();

    BlockDescriptor<interm> auxTableBlock;
    ntAuxTable->getBlockOfRows( 0, c, readOnly, auxTableBlock );
    interm *aux_table = auxTableBlock.getBlockPtr();

    size_t blockSizeDeafult = _BLOCKSIZE_;

    size_t nBlocks = n / blockSizeDeafult;
    nBlocks += (nBlocks * blockSizeDeafult != n);

    daal::tls<interm *> mkl_buff( [ = ]()-> interm* { return _CALLOC_<interm, cpu>(blockSizeDeafult * c); } );

    daal::threader_for( nBlocks, nBlocks, [ =, &mkl_buff ](int k)
    {
        interm *buff =  mkl_buff.local();

        size_t jn = blockSizeDeafult;
        if( k == nBlocks - 1 )
        {
            jn = n - k * blockSizeDeafult;
        }
        size_t j0 = k * blockSizeDeafult;

        BlockDescriptor<int>    classesBlock;
        ntClass->getBlockOfRows( j0, jn, writeOnly, classesBlock );
        int *classes = classesBlock.getBlockPtr();

        methodSpecific<method, interm, cpu>::getPredictionData( aux_table, ntData, j0, jn, p, c, classes, buff );

        ntClass->releaseBlockOfRows( classesBlock );
    } );

    mkl_buff.reduce( [ = ](interm * v)-> void { _FREE_<interm, cpu>( v ); } );

    ntAuxTable->releaseBlockOfRows( auxTableBlock );

    return;
}

template<typename interm, CpuType cpu>
void methodSpecific<defaultDense, interm, cpu>::getPredictionData( interm* aux_table, NumericTable * ntData, size_t n0, size_t n,
                                                                   size_t p, size_t c, int* classes, interm* buff )
{
     BlockDescriptor<interm> dataBlock;
     ntData->getBlockOfRows( n0, n, readOnly, dataBlock );
     interm *data = dataBlock.getBlockPtr();

    {
        char transa = 't';
        char transb = 'n';
        MKL_INT _m = c;
        MKL_INT _n = n;
        MKL_INT _k = p;
        interm alpha = 1.0;
        MKL_INT lda = p;
        MKL_INT ldy = p;
        interm beta = 0.0;
        MKL_INT ldaty = c;

        Blas<interm, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, aux_table,
            &lda, data, &ldy, &beta, buff, &ldaty);
    }

    for( size_t j = 0; j < n; j++ )
    {
        int max_c = 0;
        interm max_c_val = -(data_feature_utils::internal::MaxVal<interm,cpu>::get());

      PRAGMA_IVDEP
        for( size_t cl=0 ; cl<c ; cl++ )
        {
            interm val = buff[j*c + cl];

            if( val > max_c_val )
            {
                max_c_val = val;
                max_c     = cl;
            }
        }

        classes[ j ] = max_c;
    }

    ntData->releaseBlockOfRows( dataBlock );
}

template<typename interm, CpuType cpu>
void methodSpecific<fastCSR, interm, cpu>::getPredictionData( interm* aux_table, NumericTable * ntData, size_t n0, size_t n,
                                                              size_t p, size_t c, int* classes, interm* buff )
{
    CSRNumericTableIface *ntCSRData = dynamic_cast<CSRNumericTableIface*>(ntData);

    CSRBlockDescriptor<interm> dataBlock;
    ntCSRData->getSparseBlock( n0, n, readOnly, dataBlock );

    interm *values = dataBlock.getBlockValuesPtr();
    size_t *colIdx = dataBlock.getBlockColumnIndicesPtr();
    size_t *rowIdx = dataBlock.getBlockRowIndicesPtr();

    {
        char transa = 'n';
        MKL_INT _n = n;
        MKL_INT _p = p;
        MKL_INT _c = c;
        interm alpha = 1.0;
        interm beta  = 0.0;
        MKL_INT ldaty = n;
        char matdescra[6] = {'G',0,0,'F',0,0};

        SpBlas<interm, cpu>::xxcsrmm(&transa, &_n, &_c, &_p, &alpha,
                     matdescra,
                     values, (MKL_INT*)colIdx, (MKL_INT*)rowIdx,
                     aux_table, &_p, &beta, buff, &_n);
    }

    for( size_t j = 0; j < n; j++ )
    {
        int max_c = 0;
        interm max_c_val = -(data_feature_utils::internal::MaxVal<interm,cpu>::get());

      PRAGMA_IVDEP
        for( size_t cl=0 ; cl<c ; cl++ )
        {
            interm val = buff[j + cl*n];

            if( val > max_c_val )
            {
                max_c_val = val;
                max_c     = cl;
            }
        }

        classes[ j ] = max_c;
    }

    ntCSRData->releaseSparseBlock( dataBlock );
}

} // namespace internal
} // namespace prediction
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
