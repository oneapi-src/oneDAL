/* file: cosdistance_fpt.cpp */
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
//  Implementation of cosine distance algorithm and types methods.
//--
*/

#include "cosine_distance_types.h"

namespace daal
{
namespace algorithms
{
namespace cosine_distance
{
namespace interface1
{
/**
 * Allocates memory to store results of the cosine distance algorithm
 * \param[in] input  Pointer to input structure
 * \param[in] par    Pointer to parameter structure
 * \param[in] method Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t dim = algInput->get(data)->getNumberOfRows();
    Argument::set(cosineDistance, data_management::SerializationIfacePtr(
                      new data_management::PackedSymmetricMatrix<data_management::NumericTableIface::lowerPackedSymmetricMatrix, algorithmFPType>(
                          dim, data_management::NumericTable::doAllocate)));
    return services::Status();
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

}// namespace interface1
}// namespace cosine_distance
}// namespace algorithms
}// namespace daal
