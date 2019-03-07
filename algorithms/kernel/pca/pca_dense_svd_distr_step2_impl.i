/* file: pca_dense_svd_distr_step2_impl.i */
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
using namespace daal::internal;
using namespace daal::data_management;

template <typename algorithmFPType, CpuType cpu>
services::Status PCASVDStep2MasterKernel<algorithmFPType, cpu>::finalizeMerge(InputDataType type,
    const DataCollectionPtr &inputPartialResults,
    NumericTable &eigenvalues, NumericTable &eigenvectors)
{
    if(type == correlation)
        return services::Status(services::ErrorInputCorrelationNotSupportedInOnlineAndDistributed);

    size_t nObservations = 0;

    size_t nPartialResults = inputPartialResults->size();
    DataCollection rTables;

    for(size_t i = 0; i < nPartialResults; i++)
    {
        services::SharedPtr<PartialResult<svdDense> > partialRes =
            services::staticPointerCast<PartialResult<svdDense>, SerializationIface>(inputPartialResults->get(i));

        size_t nBlocks = partialRes->get(pca::auxiliaryData)->size();
        for(size_t j = 0; j < nBlocks; j++)
        {
            rTables.push_back(partialRes->get(pca::auxiliaryData, j));
        }

        NumericTable *nCurrentObservationsTable = partialRes->get(pca::nObservationsSVD).get();
        nObservations += nCurrentObservationsTable->getValue<int>(0, 0);
    }

    svd::Parameter kmPar;
    kmPar.leftSingularMatrix = svd::notRequired;

    const size_t nPartialBlocks = rTables.size();
    const size_t nInputs = nPartialBlocks * 2;
    TArray<NumericTable*, cpu> svdInputs(nInputs);
    DAAL_CHECK_MALLOC(svdInputs.get());
    for(size_t i = 0; i < nPartialBlocks; i++)
    {
        svdInputs[i] = static_cast<NumericTable *>(rTables[i].get());
        svdInputs[i + nPartialBlocks] = 0;
    }

    const size_t nResults = 3;
    NumericTable *svdResults[nResults] = { &eigenvalues, nullptr, &eigenvectors };

    daal::algorithms::svd::internal::SVDOnlineKernel<algorithmFPType, svd::defaultDense, cpu> svdKernel;
    services::Status s = svdKernel.finalizeCompute(nInputs, svdInputs.get(), nResults, svdResults, &kmPar);

    if(s)
        s = this->scaleSingularValues(eigenvalues, nObservations);
    return s;
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
