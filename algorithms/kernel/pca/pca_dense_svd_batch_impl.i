/* file: pca_dense_svd_batch_impl.i */
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
//  Functuons that are used in PCA algorithm
//--
*/

#ifndef __PCA_DENSE_SVD_BATCH_IMPL_I__
#define __PCA_DENSE_SVD_BATCH_IMPL_I__

#include "service_math.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "threading.h"

using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{

template <typename interm, CpuType cpu>
void PCASVDBatchKernel<interm, cpu>::compute(const data_management::NumericTablePtr &data,
                                             data_management::NumericTablePtr &eigenvalues,
                                             data_management::NumericTablePtr &eigenvectors)
{
    _data = data;
    _eigenvalues = eigenvalues;
    _eigenvectors = eigenvectors;

    _nObservations = _data->getNumberOfRows();
    _nFeatures = _data->getNumberOfColumns();

    normalizeDataset();

    if (this->_errors->size() != 0) { return; }

    decompose();

    if (this->_errors->size() != 0) { return; }

    this->scaleSingularValues(_eigenvalues.get(), _nObservations);
}


/********************* tls_data_t class *******************************************************/
template<typename interm, CpuType cpu> struct tls_data_t
{
    interm* mean;
    interm* variance;
    interm  nvectors;
    int malloc_errors;

    tls_data_t(size_t nFeatures)
    {
        malloc_errors = 0;

        nvectors = 0;

        mean  = service_scalable_calloc<interm, cpu>(nFeatures);
        if(!mean) { malloc_errors++; }

        variance  = service_scalable_calloc<interm, cpu>(nFeatures);
        if(!variance) { malloc_errors++; }

    }

    ~tls_data_t()
    {
        if(mean)  { service_scalable_free<interm,cpu>( mean );  mean = 0; }
        if(variance) { service_scalable_free<interm,cpu>( variance ); variance = 0; }
    }
};


template <typename interm, CpuType cpu>
void PCASVDBatchKernel<interm, cpu>::normalizeDataset()
{
    using data_management::NumericTable;
    using data_management::HomogenNumericTable;
    using daal::internal::HomogenNumericTableCPU;

    if(this->_type == normalizedDataset)
    {
        _normalizedData = _data;
        return;
    }

    BlockDescriptor<interm> block;
    _data->getBlockOfRows(0, _nObservations, data_management::readOnly, block);
    interm *dataArray = block.getBlockPtr();

    HomogenNumericTableCPU<interm, cpu> *normalized = new HomogenNumericTableCPU<interm, cpu>(_nFeatures, _nObservations);
    normalized->assign(0);

    _normalizedData = services::SharedPtr<HomogenNumericTable<interm> >(normalized);

    BlockDescriptor<interm> normalizedBlock;
    _normalizedData->getBlockOfRows(0, _nObservations, data_management::readOnly, normalizedBlock);
    interm *normalizedDataArray = normalizedBlock.getBlockPtr();

    interm* mean_total      = service_calloc<interm, cpu>(_nFeatures);
    interm* inv_sigma_total = service_calloc<interm, cpu>(_nFeatures);

    if(!(mean_total) || !(inv_sigma_total))
    {
        this->_errors->add(daal::services::ErrorMemoryAllocationFailed);

        if(mean_total)      service_free<interm,cpu>( mean_total );
        if(inv_sigma_total) service_free<interm,cpu>( inv_sigma_total );

        return;
    }

#define _BLOCK_SIZE_ 256
    /* Split rows by blocks, block size cannot be less than _nObservations */
    size_t numRowsInBlock = (_nObservations > _BLOCK_SIZE_)?_BLOCK_SIZE_:_nObservations;
    /* Number of blocks */
    size_t numBlocks   = _nObservations / numRowsInBlock;
    /* Last block can be bigger than others */
    size_t numRowsInLastBlock = numRowsInBlock + ( _nObservations - numBlocks * numRowsInBlock);

    /* TLS data initialization */
    daal::tls<tls_data_t<interm, cpu> *> tls_data([ = ]()
    {
        return new tls_data_t<interm, cpu>( _nFeatures );
    });

    /* Compute partial means and variances for each block */
    daal::threader_for( numBlocks, numBlocks, [ & ](int iBlock)
    {
        struct tls_data_t<interm,cpu> * tls_data_local = tls_data.local();
        if(tls_data_local->malloc_errors) return;

        size_t nVectors_local = (iBlock < (numBlocks-1))?numRowsInBlock:numRowsInLastBlock;
        size_t startRow       = iBlock * numRowsInBlock;

        interm* dataArray_local = dataArray + startRow * _nFeatures;
        interm* mean_local      = tls_data_local->mean;
        interm* variance_local  = tls_data_local->variance;

        for(int i = 0; i < nVectors_local; i++)
        {
            interm _invN = interm(1.0) / interm(tls_data_local->nvectors+1);

           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for(int j = 0; j < _nFeatures; j++)
            {
                interm delta_local  = dataArray_local[i*_nFeatures + j] - mean_local[j];

                mean_local[j]      += delta_local * _invN;
                variance_local[j]  += delta_local *
                                      ( dataArray_local[i*_nFeatures + j] - mean_local[j] );
            }

            tls_data_local->nvectors++;
        }

    } );

    interm n_current = 0;

    /* Merge mean and variance arrays by blocks */
    tls_data.reduce( [ =, &inv_sigma_total, &mean_total, &n_current ]( tls_data_t<interm,cpu>* tls_data_local )
    {
        if(tls_data_local->malloc_errors)
        {
            this->_errors->add(daal::services::ErrorMemoryAllocationFailed);
            delete tls_data_local;
            return;
        }

        /* loop invariants */
        interm n_local           = tls_data_local->nvectors;
        interm n1_p_n2           = n_current + n_local;
        interm n1_m_n2           = n_current * n_local;
        interm n1_m_n2_o_n1_p_n2 = n1_m_n2 / n1_p_n2;
        interm inv_n1_p_n2       = interm(1.0) / (n1_p_n2);
        interm inv_n1_p_n2_m1    = interm(1.0) / (n1_p_n2 - interm(1.0));

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(int j = 0; j < _nFeatures; j++)
        {
            interm _delta       = tls_data_local->mean[j] - mean_total[j];

            interm _vl          = tls_data_local->variance[j];        /* local variances are not scaled yet */
            interm _vt          = inv_sigma_total[j] * (n_current-1); /* merged variances are already scaled */

            /* merging variances */
            inv_sigma_total[j]  = ( _vt + _vl +
                                    _delta*_delta * n1_m_n2_o_n1_p_n2 )
                                    * inv_n1_p_n2_m1;
            /* merging means */
            mean_total[j]       = ( mean_total[j] * n_current +
                                    tls_data_local->mean[j] * n_local )
                                    * inv_n1_p_n2;
        }

        /* Increase number of already merged values */
        n_current += n_local;

        delete tls_data_local;
    } );


    /* Convert array of variances to inverse sigma's */
   PRAGMA_IVDEP
   PRAGMA_VECTOR_ALWAYS
    for(int j = 0; j < _nFeatures; j++)
    {
        inv_sigma_total[j] = interm(1.0) / daal::internal::Math<interm, cpu>::sSqrt(inv_sigma_total[j]);
    }

    /* Final normalization threaded loop */
    daal::threader_for( numBlocks, numBlocks, [ & ](int iBlock)
    {
        size_t nVectors_local = (iBlock < (numBlocks-1))?numRowsInBlock:numRowsInLastBlock;
        size_t startRow       = iBlock * numRowsInBlock;

        interm* dataArray_local     = dataArray           + startRow * _nFeatures;
        interm* normDataArray_local = normalizedDataArray + startRow * _nFeatures;

        for(int i = 0; i < nVectors_local; i++)
        {
           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for(int j = 0; j < _nFeatures; j++)
            {
                normDataArray_local[i * _nFeatures + j] = (dataArray_local[i * _nFeatures + j] - mean_total[j]) * inv_sigma_total[j];
            }
        }
    } );

    service_free<interm,cpu>( mean_total );
    service_free<interm,cpu>( inv_sigma_total );

    _data->releaseBlockOfRows(block);
    _normalizedData->releaseBlockOfRows(normalizedBlock);
}

template <typename interm, CpuType cpu>
void PCASVDBatchKernel<interm, cpu>::decompose()
{
    const NumericTable *normalizedDataTable = _normalizedData.get();
    const NumericTable *const *svdInputs = &normalizedDataTable;

    NumericTable *svdResults[3];
    svdResults[0] = _eigenvalues.get();
    svdResults[1] = 0;
    svdResults[2] = _eigenvectors.get();

    svd::Parameter params;
    params.leftSingularMatrix = svd::notRequired;

    daal::algorithms::svd::internal::SVDBatchKernel<interm, svd::defaultDense, cpu> svdKernel;
    svdKernel.compute(1, svdInputs, 3, svdResults, &params);

    if(svdKernel.getErrorCollection()->size() > 0) { this->_errors->add(svdKernel.getErrorCollection()); }
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
