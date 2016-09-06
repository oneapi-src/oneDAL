/* file: pca_dense_svd_distr_step2_impl.i */
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

#ifndef __PCA_DENSE_SVD_DISTR_STEP2_IMPL_I__
#define __PCA_DENSE_SVD_DISTR_STEP2_IMPL_I__

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
void PCASVDStep2MasterKernel<interm, cpu>::finalizeMerge(const data_management::DataCollectionPtr &inputPartialResults,
                                                         data_management::NumericTablePtr &eigenvalues,
                                                         data_management::NumericTablePtr &eigenvectors)
{
    if(this->_type == correlation)
    {
        this->_errors->add(services::ErrorInputCorrelationNotSupportedInOnlineAndDistributed);
    }

    size_t nObservations = 0;

    size_t nPartialResults = inputPartialResults->size();
    data_management::DataCollection rTables;

    for(size_t i = 0; i < nPartialResults; i++)
    {
        services::SharedPtr<PartialResult<svdDense> > partialRes =
            services::staticPointerCast<PartialResult<svdDense>, data_management::SerializationIface>(inputPartialResults->get(i));

        size_t nBlocks = partialRes->get(pca::auxiliaryData)->size();
        for(size_t j = 0; j < nBlocks; j++)
        {
            rTables.push_back(partialRes->get(pca::auxiliaryData, j));
        }

        NumericTable *nCurrentObservationsTable = partialRes->get(pca::nObservationsSVD).get();
        BlockDescriptor<interm> block;
        nCurrentObservationsTable->getBlockOfRows(0, 1, data_management::readOnly, block);
        interm *nCurrentObservations = block.getBlockPtr();

        nObservations += *nCurrentObservations;

        nCurrentObservationsTable->releaseBlockOfRows(block);
    }

    svd::Parameter kmPar;
    kmPar.leftSingularMatrix = svd::notRequired;

    size_t nPartialBlocks = rTables.size();

    size_t nInputs = nPartialBlocks * 2;
    NumericTable **svdInputs = new NumericTable*[nInputs];
    for(size_t i = 0; i < nPartialBlocks; i++)
    {
        svdInputs[i] = static_cast<NumericTable *>(rTables[i].get());
        svdInputs[i + nPartialBlocks] = 0;
    }

    size_t nResults = 3;
    NumericTable *svdResults[3];

    svdResults[0] = eigenvalues.get();
    svdResults[1] = 0;
    svdResults[2] = eigenvectors.get();

    daal::algorithms::svd::internal::SVDOnlineKernel<interm, svd::defaultDense, cpu> svdKernel;
    svdKernel.finalizeCompute(nInputs, svdInputs, nResults, svdResults, &kmPar);

    if(svdKernel.getErrorCollection()->size() > 0) { this->_errors->add(svdKernel.getErrorCollection()); }

    delete[] svdInputs;

    this->scaleSingularValues(eigenvalues.get(), nObservations);
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
