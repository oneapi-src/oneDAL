/* file: sorting.cpp */
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
//  Implementation of sorting algorithm and types methods.
//--
*/

#include "sorting_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace sorting
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_SORTING_RESULT_ID);
Input::Input() : daal::algorithms::Input(lastInputId + 1) {}
Input::Input(const Input& other) : daal::algorithms::Input(other){}

/**
 * Returns an input object for the sorting algorithm
 * \param[in] id    Identifier of the %input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Sets the input object of the sorting algorithm
 * \param[in] id    Identifier of the %input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Check the correctness of the %Input object
 * \param[in] method    Algorithm computation method
 * \param[in] par       Pointer to the parameters of the algorithm
 */
Status Input::check(const Parameter *par, int method) const
{
    const int unexpectedLayouts = packed_mask;
    return checkNumericTable(get(data).get(), dataStr(), unexpectedLayouts);
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the final result of the sorting algorithm
 * \param[in] id   Identifier of the final result, \ref ResultId
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Sets the Result object of the sorting algorithm
 * \param[in] id        Identifier of the Result object
 * \param[in] value     Pointer to the Result object
 */
void Result::set(ResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] in     Pointer to the object
 * \param[in] par     %Parameter of algorithm
 * \param[in] method Algorithm computation method
 */
Status Result::check(const daal::algorithms::Input *in, const Parameter *par, int method) const
{
    const Input *input = static_cast<const Input *>(in);

    const size_t nFeatures = input->get(data)->getNumberOfColumns();
    const size_t nVectors  = input->get(data)->getNumberOfRows();
    const int unexpectedLayouts = packed_mask;

    return checkNumericTable(get(sortedData).get(), sortedDataStr(), unexpectedLayouts, 0, nFeatures, nVectors);
}

}// namespace interface1
}// namespace sorting
}// namespace algorithms
}// namespace daal
