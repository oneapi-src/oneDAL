/* file: em_gmm_init_dense_batch_fpt.cpp */
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

template<typename AlgorithmFPType>
ErrorID EMforKernel<AlgorithmFPType>::run(const data_management::NumericTablePtr& inputData,
    const data_management::NumericTablePtr& inputWeights,
    const data_management::NumericTablePtr& inputMeans,
    const data_management::DataCollectionPtr& inputCov,
    AlgorithmFPType& loglikelyhood)
{
    this->input.set(daal::algorithms::em_gmm::data, inputData);
    this->input.set(daal::algorithms::em_gmm::inputWeights, inputWeights);
    this->input.set(daal::algorithms::em_gmm::inputMeans, inputMeans);
    this->input.set(daal::algorithms::em_gmm::inputCovariances, inputCov);

    SharedPtr<daal::algorithms::em_gmm::Result> emResult(new daal::algorithms::em_gmm::Result());
    emResult->set(daal::algorithms::em_gmm::weights, inputWeights);
    emResult->set(daal::algorithms::em_gmm::means, inputMeans);
    emResult->set(daal::algorithms::em_gmm::covariances, inputCov);

    SharedPtr<HomogenNumericTable<AlgorithmFPType> > loglikelyhoodValueTable(
        new HomogenNumericTable<AlgorithmFPType>(1, 1, NumericTable::doAllocate));

    SharedPtr<HomogenNumericTable<int> > nIterationsValueTable(new HomogenNumericTable<int>(1, 1, NumericTable::doAllocate));
    emResult->set(daal::algorithms::em_gmm::goalFunction, loglikelyhoodValueTable);
    emResult->set(daal::algorithms::em_gmm::nIterations, nIterationsValueTable);

    this->setResult(emResult);
    this->computeNoThrow();
    if(this->getErrors()->size() != 0)
        return ErrorEMInitNoTrialConverges;
    loglikelyhood = loglikelyhoodValueTable->getArray()[0];
    return ErrorID(0);
}

template class EMforKernel<DAAL_FPTYPE>;

}
} // namespace init
} // namespace em_gmm
} // namespace algorithms
} // namespace daal
