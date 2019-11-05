/* file: gbt_regression_training_partial_result_fpt.cpp */
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
//  Implementation of the gradient boosted trees algorithm interface
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_training_types.h"
#include "gbt_regression_model_impl.h"
#include "gbt_regression_tree_impl.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{

using namespace daal::data_management;

/**
 * Allocates memory to store the results of gradient boosted trees model-based training
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep1::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const DistributedInput<step1Local> *algInput = static_cast<const DistributedInput<step1Local> *>(input);

    services::Status status;

    const Parameter *par = static_cast<const Parameter *>(parameter);
    const size_t nRows = algInput->get(step1BinnedData)->getNumberOfRows();

    set(response, algInput->get(step1InputResponse));
    set(optCoeffs, HomogenNumericTable<algorithmFPType>::create(2, nRows, NumericTable::doAllocate, &status));
    set(treeOrder, HomogenNumericTable<int>::create(1, nRows, NumericTable::doAllocate, &status));
    set(finalizedTree, algInput->get(step1InputTreeStructure));
    set(step1TreeStructure, daal::algorithms::gbt::regression::internal::TreeTableConnector<algorithmFPType>::createGBTree(par->maxTreeDepth, &status));

    return status;
}

/**
 * Allocates memory to store the results of gradient boosted trees model-based training
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep2::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const DistributedInput<step2Local> *algInput = static_cast<const DistributedInput<step2Local> *>(input);

    services::Status status;

    set(finishedFlag, HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));

    return status;
}

/**
 * Allocates memory to store the results of gradient boosted trees model-based training
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep3::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const DistributedInput<step3Local> *algInput = static_cast<const DistributedInput<step3Local> *>(input);

    services::Status status;

    set(histograms, DataCollectionPtr(new DataCollection()));

    return status;
}

/**
 * Allocates memory to store the results of gradient boosted trees model-based training
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep4::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const DistributedInput<step4Local> *algInput = static_cast<const DistributedInput<step4Local> *>(input);

    services::Status status;

    set(totalHistograms, DataCollectionPtr(new DataCollection()));
    set(bestSplits, DataCollectionPtr(new DataCollection()));

    return status;
}

/**
 * Allocates memory to store the results of gradient boosted trees model-based training
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep5::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const DistributedInput<step5Local> *algInput = static_cast<const DistributedInput<step5Local> *>(input);

    services::Status status;

    set(step5TreeStructure, algInput->get(step5InputTreeStructure));
    set(step5TreeOrder, algInput->get(step5InputTreeOrder));

    return status;
}

/**
 * Allocates memory to store the results of gradient boosted trees model-based training
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep6::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const DistributedInput<step6Local> *algInput = static_cast<const DistributedInput<step6Local> *>(input);

    const size_t nFeatures = algInput->get(step6BinValues)->size();

    services::Status status;
    set(partialModel, daal::algorithms::gbt::regression::Model::create(nFeatures, &status));
    return status;
}

template DAAL_EXPORT Status DistributedPartialResultStep1::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep2::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep3::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep4::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep5::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep6::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
