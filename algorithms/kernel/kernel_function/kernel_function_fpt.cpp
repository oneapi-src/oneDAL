/* file: kernel_function_fpt.cpp */
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
//  Implementation of kernel function algorithm and types methods.
//--
*/

#include "kernel_function_types.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace interface1
{
/**
 * Allocates memory to store results of the kernel function algorithm
 * \param[in] input  Pointer to the structure with the input objects
 * \param[in] par    Pointer to the structure of the algorithm parameters
 * \param[in] method       Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    const Input *algInput = static_cast<const Input *>(input);

    const size_t nVectors1 = algInput->get(X)->getNumberOfRows();
    const size_t nVectors2 = algInput->get(Y)->getNumberOfRows();

    services::Status status;
    set(values, data_management::HomogenNumericTable<algorithmFPType>::create(nVectors2, nVectors1, data_management::NumericTable::doAllocate, &status));
    return status;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

}// namespace interface1
}// namespace kernel function
}// namespace algorithms
}// namespace daal
