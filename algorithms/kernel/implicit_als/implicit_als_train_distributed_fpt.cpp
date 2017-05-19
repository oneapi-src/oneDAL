/* file: implicit_als_train_distributed_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "implicit_als_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace interface1
{
/**
 * Allocates memory to store a partial result of the implicit ALS training algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep1::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors = algParameter->nFactors;
    Argument::set(outputOfStep1ForStep2, data_management::SerializationIfacePtr(
            new data_management::HomogenNumericTable<algorithmFPType>(
                    nFactors, nFactors, data_management::NumericTable::doAllocate)));
    return services::Status();
}

/**
 * Allocates memory to store a partial result of the implicit ALS training algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep2::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors = algParameter->nFactors;
    Argument::set(outputOfStep2ForStep4, data_management::SerializationIfacePtr(
            new data_management::HomogenNumericTable<algorithmFPType>(
                    nFactors, nFactors, data_management::NumericTable::doAllocate)));
    return services::Status();
}

/**
 * Allocates memory to store a partial result of the implicit ALS training algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep3::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const DistributedInput<step3Local> *algInput = static_cast<const DistributedInput<step3Local> *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    size_t nBlocks = algInput->getNumberOfBlocks();
    size_t offset = algInput->getOffset();

    services::Collection<size_t> _keys;
    data_management::DataCollection _values;
    for (size_t i = 0; i < nBlocks; i++)
    {
        data_management::NumericTablePtr outBlockIndices = algInput->getOutBlockIndices(i);
        if (!outBlockIndices) { continue; }
        _keys.push_back(i);
        _values.push_back(data_management::SerializationIfacePtr(
            new PartialModel(*algParameter, offset, outBlockIndices, (algorithmFPType)0.0)));
    }
    data_management::KeyValueDataCollectionPtr modelsCollection =
        data_management::KeyValueDataCollectionPtr (new data_management::KeyValueDataCollection(_keys, _values));
    set(outputOfStep3ForStep4, modelsCollection);
    return services::Status();
}

/**
 * Allocates memory to store a partial result of the implicit ALS training algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep4::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const DistributedInput<step4Local> *algInput = static_cast<const DistributedInput<step4Local> *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    set(outputOfStep4ForStep1, PartialModelPtr(
        new PartialModel(*algParameter, algInput->getNumberOfRows(), (algorithmFPType)0.0)));
    return services::Status();
}

template DAAL_EXPORT services::Status DistributedPartialResultStep1::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input,
                                                                               const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status DistributedPartialResultStep2::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input,
                                                                               const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status DistributedPartialResultStep3::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input,
                                                                               const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status DistributedPartialResultStep4::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input,
                                                                               const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace training
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
