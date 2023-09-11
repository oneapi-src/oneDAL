/* file: implicit_als_train_distributed_fpt.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "algorithms/implicit_als/implicit_als_training_types.h"

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
/**
 * Allocates memory to store a partial result of the implicit ALS training algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status DistributedPartialResultStep1::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                           const int method)
{
    const Parameter * algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors                = algParameter->nFactors;
    Status st;
    set(outputOfStep1ForStep2, HomogenNumericTable<algorithmFPType>::create(nFactors, nFactors, NumericTable::doAllocate, &st));
    return st;
}

/**
 * Allocates memory to store a partial result of the implicit ALS training algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status DistributedPartialResultStep2::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                           const int method)
{
    const Parameter * algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors                = algParameter->nFactors;
    Status st;
    set(outputOfStep2ForStep4, HomogenNumericTable<algorithmFPType>::create(nFactors, nFactors, NumericTable::doAllocate, &st));
    return st;
}

/**
 * Allocates memory to store a partial result of the implicit ALS training algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status DistributedPartialResultStep3::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                           const int method)
{
    const DistributedInput<step3Local> * algInput = static_cast<const DistributedInput<step3Local> *>(input);
    const Parameter * algParameter                = static_cast<const Parameter *>(parameter);

    const size_t nBlocks = algInput->getNumberOfBlocks();
    const size_t offset  = algInput->getOffset();

    Collection<size_t> _keys;
    Collection<SerializationIfacePtr> _values;
    for (size_t i = 0; i < nBlocks; i++)
    {
        NumericTablePtr outBlockIndices = algInput->getOutBlockIndices(i);
        if (!outBlockIndices)
        {
            continue;
        }
        _keys.push_back(i);
        _values.push_back(SerializationIfacePtr(new PartialModel(*algParameter, offset, outBlockIndices, (algorithmFPType)0.0)));
    }
    KeyValueDataCollectionPtr modelsCollection = KeyValueDataCollectionPtr(new KeyValueDataCollection(_keys, _values));
    set(outputOfStep3ForStep4, modelsCollection);
    return Status();
}

/**
 * Allocates memory to store a partial result of the implicit ALS training algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status DistributedPartialResultStep4::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                           const int method)
{
    const DistributedInput<step4Local> * algInput = static_cast<const DistributedInput<step4Local> *>(input);
    const Parameter * algParameter                = static_cast<const Parameter *>(parameter);

    set(outputOfStep4ForStep1, PartialModelPtr(new PartialModel(*algParameter, algInput->getNumberOfRows(), (algorithmFPType)0.0)));
    return Status();
}

template DAAL_EXPORT Status DistributedPartialResultStep1::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep2::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep3::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep4::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                 const daal::algorithms::Parameter * parameter, const int method);

} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
