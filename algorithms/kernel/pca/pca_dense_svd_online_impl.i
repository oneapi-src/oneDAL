/* file: pca_dense_svd_online_impl.i */
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

#ifndef __PCA_DENSE_SVD_ONLINE_IMPL_I__
#define __PCA_DENSE_SVD_ONLINE_IMPL_I__

#include "service_math.h"
#include "service_memory.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{

template <typename interm, CpuType cpu>
void PCASVDOnlineKernel<interm, cpu>::compute(const data_management::NumericTablePtr &data,
                                              data_management::NumericTablePtr &nObservations,
                                              data_management::NumericTablePtr &auxiliaryTable,
                                              data_management::NumericTablePtr &sumSVD,
                                              data_management::NumericTablePtr &sumSquaresSVD)
{
    if(this->_type == correlation)
    {
        this->_errors->add(services::ErrorInputCorrelationNotSupportedInOnlineAndDistributed);
        return;
    }

    _data = data;

    _auxiliaryTable = auxiliaryTable;
    _sumSquaresSVD = sumSquaresSVD;
    _sumSVD = sumSVD;

    _nObservations = _data->getNumberOfRows();
    _nFeatures = _data->getNumberOfColumns();

    BlockDescriptor<int> oldObservationsBlock;
    nObservations->getBlockOfRows(0, 1, data_management::readWrite, oldObservationsBlock);
    int *oldObservations = oldObservationsBlock.getBlockPtr();
    _nOldObservations = *oldObservations;

    normalizeDataset();

    if (this->_errors->size() != 0) { return; }

    decompose();

    *oldObservations += _nObservations;
    nObservations->releaseBlockOfRows(oldObservationsBlock);
}

template <typename interm, CpuType cpu>
void PCASVDOnlineKernel<interm, cpu>::finalizeMerge(const data_management::NumericTablePtr &nObservationsTable,
                                                    data_management::NumericTablePtr &eigenvalues,
                                                    data_management::NumericTablePtr &eigenvectors,
                                                    data_management::DataCollectionPtr &rTables)
{
    if(this->_type == correlation)
    {
        this->_errors->add(services::ErrorInputCorrelationNotSupportedInOnlineAndDistributed);
    }

    BlockDescriptor<int> block;
    nObservationsTable->getBlockOfRows(0, 1, data_management::readOnly, block);
    int *nObservations = block.getBlockPtr();

    nObservationsTable->releaseBlockOfRows(block);

    svd::Parameter kmPar;
    kmPar.leftSingularMatrix = svd::notRequired;

    size_t np = rTables->size();

    size_t nInputs = np * 2;
    data_management::NumericTable **svdInputs = new data_management::NumericTable*[nInputs];
    for(size_t i = 0; i < np; i++)
    {
        svdInputs[i     ] = static_cast<data_management::NumericTable *>(rTables->get(i).get());
        svdInputs[i + np] = 0;
    }

    size_t nResults = 3;
    data_management::NumericTable *svdResults[3];

    svdResults[0] = eigenvalues.get();
    svdResults[1] = 0;
    svdResults[2] = eigenvectors.get();

    daal::algorithms::svd::internal::SVDOnlineKernel<interm, svd::defaultDense, cpu> svdKernel;
    svdKernel.finalizeCompute(nInputs, svdInputs, nResults, svdResults, &kmPar);

    if(svdKernel.getErrorCollection()->size() > 0) { this->_errors->add(svdKernel.getErrorCollection()); }

    delete[] svdInputs;

    this->scaleSingularValues(eigenvalues.get(), *nObservations);
}

template <typename interm, CpuType cpu>
void PCASVDOnlineKernel<interm, cpu>::decompose()
{
    svd::Parameter kmPar;
    kmPar.leftSingularMatrix = svd::notRequired;

    size_t na = 1;
    const NumericTable *normalizedDataTable = _normalizedData.get();
    const NumericTable *const *svdInputs = &normalizedDataTable;

    size_t m = _normalizedData->getNumberOfColumns();
    size_t n = _normalizedData->getNumberOfRows();

    size_t nr = 2;

    data_management::NumericTable *svdResults[2] = {0, _auxiliaryTable.get()};

    daal::algorithms::svd::internal::SVDOnlineKernel<interm, svd::defaultDense, cpu> svdKernel;
    svdKernel.compute(na, svdInputs, nr, svdResults, &kmPar);

    if(svdKernel.getErrorCollection()->size() > 0) { this->_errors->add(svdKernel.getErrorCollection()); }
}

namespace
{
template <typename interm, CpuType cpu>
inline void computeSumsAndSsq(const size_t nObservations, const size_t _nFeatures, const interm *data, interm *sums, interm *ssq)
{
    for (size_t i = 0; i < nObservations; i++)
    {
        for (size_t j = 0; j < _nFeatures; j++)
        {
            sums[j] += data[i * _nFeatures + j];
            ssq[j] += data[i * _nFeatures + j] * data[i * _nFeatures + j];
        }
    }
}

template <typename interm, CpuType cpu>
inline void computeMean(const size_t nObservations, const size_t _nFeatures, const interm *sums, interm *mean)
{
    for (size_t j = 0; j < _nFeatures; j++)
    {
        mean[j] = sums[j] / nObservations;
    }
}

template <typename interm, CpuType cpu>
inline void computeVariance(const size_t nObservations, const size_t _nFeatures,
                            const interm *sums, const interm *ssq, const interm *mean, interm *variance)
{
    for (size_t j = 0; j < _nFeatures; j++)
    {
        variance[j] =  daal::internal::Math<interm,cpu>::sSqrt((ssq[j] - 2 * mean[j] * sums[j] + nObservations * mean[j] * mean[j]) / (nObservations - 1));
    }
}

template <typename interm, CpuType cpu>
inline void normalizeData(const size_t nObservations, const size_t _nFeatures, const interm *data, const interm *mean,
                          const interm *variance, interm *normalizedData)
{
    for (size_t i = 0; i < nObservations; i++)
    {
        for (size_t j = 0; j < _nFeatures; j++)
        {
            normalizedData[i * _nFeatures + j] = (data[i * _nFeatures + j] - mean[j]) / variance[j];
        }
    }
}
}

template <typename interm, CpuType cpu>
void PCASVDOnlineKernel<interm, cpu>::normalizeDataset()
{
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

    BlockDescriptor<interm> blockSums;
    _sumSVD->getBlockOfRows(0, 1, data_management::readWrite, blockSums);
    interm *sums = blockSums.getBlockPtr();

    BlockDescriptor<interm> blockSsq;
    _sumSquaresSVD->getBlockOfRows(0, 1, data_management::readWrite, blockSsq);
    interm *ssq = blockSsq.getBlockPtr();

    size_t totalObservations = _nOldObservations + _nObservations;

    interm *mean = (interm *)daal::services::daal_malloc(_nFeatures * sizeof(interm));
    interm *variance = (interm *)daal::services::daal_malloc(_nFeatures * sizeof(interm));

    computeSumsAndSsq<interm, cpu>(_nObservations, _nFeatures, dataArray, sums, ssq);

    computeMean<interm, cpu>(totalObservations, _nFeatures, sums, mean);

    computeVariance<interm, cpu>(totalObservations, _nFeatures, sums, ssq, mean, variance);

    normalizeData<interm, cpu>(_nObservations, _nFeatures, dataArray, mean, variance, normalizedDataArray);

    _sumSVD->releaseBlockOfRows(blockSums);
    _sumSquaresSVD->releaseBlockOfRows(blockSsq);


    daal::services::daal_free(mean);
    daal::services::daal_free(variance);

    _data->releaseBlockOfRows(block);
    _normalizedData->releaseBlockOfRows(normalizedBlock);
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
