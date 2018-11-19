/* file: implicit_als_train_distributed_fpt.cpp */
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
DAAL_EXPORT Status DistributedPartialResultStep1::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors = algParameter->nFactors;
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
DAAL_EXPORT Status DistributedPartialResultStep2::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors = algParameter->nFactors;
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
DAAL_EXPORT Status DistributedPartialResultStep3::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const DistributedInput<step3Local> *algInput = static_cast<const DistributedInput<step3Local> *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    size_t nBlocks = algInput->getNumberOfBlocks();
    size_t offset = algInput->getOffset();

    Collection<size_t> _keys;
    Collection<SerializationIfacePtr> _values;
    for (size_t i = 0; i < nBlocks; i++)
    {
        NumericTablePtr outBlockIndices = algInput->getOutBlockIndices(i);
        if (!outBlockIndices) { continue; }
        _keys.push_back(i);
        _values.push_back(SerializationIfacePtr(
            new PartialModel(*algParameter, offset, outBlockIndices, (algorithmFPType)0.0)));
    }
    KeyValueDataCollectionPtr modelsCollection =
        KeyValueDataCollectionPtr (new KeyValueDataCollection(_keys, _values));
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
DAAL_EXPORT Status DistributedPartialResultStep4::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const DistributedInput<step4Local> *algInput = static_cast<const DistributedInput<step4Local> *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    set(outputOfStep4ForStep1, PartialModelPtr(
        new PartialModel(*algParameter, algInput->getNumberOfRows(), (algorithmFPType)0.0)));
    return Status();
}

template DAAL_EXPORT Status DistributedPartialResultStep1::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input,
                                                                               const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep2::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input,
                                                                               const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep3::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input,
                                                                               const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep4::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input,
                                                                               const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace training
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
