/* file: em_gmm_init_dense_batch_fpt.cpp */
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
//  Implementation of EMforKernel
//--
*/

#include "em_gmm_init_dense_default_batch_kernel.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace init
{
namespace internal
{

template<typename algorithmFPType>
ErrorID EMforKernel<algorithmFPType>::run(data_management::NumericTable &inputData,
                data_management::NumericTable &inputWeights,
                data_management::NumericTable &inputMeans,
                data_management::DataCollectionPtr &inputCov,
                const em_gmm::CovarianceStorageId covType,
                algorithmFPType &loglikelyhood)
{
    this->input.set(daal::algorithms::em_gmm::data, NumericTablePtr(&inputData, EmptyDeleter()));
    this->input.set(daal::algorithms::em_gmm::inputWeights, NumericTablePtr(&inputWeights, EmptyDeleter()));
    this->input.set(daal::algorithms::em_gmm::inputMeans, NumericTablePtr(&inputMeans, EmptyDeleter()));
    this->input.set(daal::algorithms::em_gmm::inputCovariances, inputCov);
    this->parameter.covarianceStorage = covType;

    daal::algorithms::em_gmm::ResultPtr emResult(new daal::algorithms::em_gmm::Result());
    emResult->set(daal::algorithms::em_gmm::weights, NumericTablePtr(&inputWeights, EmptyDeleter()));
    emResult->set(daal::algorithms::em_gmm::means, NumericTablePtr(&inputMeans, EmptyDeleter()));
    emResult->set(daal::algorithms::em_gmm::covariances, inputCov);

    services::Status status;
    SharedPtr<HomogenNumericTable<algorithmFPType> > loglikelyhoodValueTable = HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status);
    if(!status)
    {
        return ErrorMemoryAllocationFailed;
    }

    NumericTablePtr nIterationsValueTable = HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status);
    if(!status)
    {
        return ErrorMemoryAllocationFailed;
    }
    emResult->set(daal::algorithms::em_gmm::goalFunction, loglikelyhoodValueTable);
    emResult->set(daal::algorithms::em_gmm::nIterations, nIterationsValueTable);

    this->setResult(emResult);
    services::Status s = this->computeNoThrow();
    if(!s)
    {
        return ErrorEMInitNoTrialConverges;
    }
    loglikelyhood = loglikelyhoodValueTable->getArray()[0];
    return ErrorID(0);
}

template class EMforKernel<DAAL_FPTYPE>;

}
} // namespace init
} // namespace em_gmm
} // namespace algorithms
} // namespace daal
