/* file: quantiles_fpt.cpp */
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
//  Implementation of quantiles algorithm and types methods.
//--
*/

#include "quantiles_types.h"

namespace daal
{
namespace algorithms
{
namespace quantiles
{
namespace interface1
{
/**
 * Allocates memory to store final results of the quantile algorithms
 * \param[in] input     Input objects for the quantiles algorithm
 * \param[in] parameter Parameters of the quantiles algorithm
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    const Input *in = static_cast<const Input *>(input);
    const Parameter *par = static_cast<const Parameter *>(parameter);

    size_t nFeatures = in->get(data)->getNumberOfColumns();
    size_t nQuantileOrders = par->quantileOrders->getNumberOfColumns();

    set(quantiles, data_management::HomogenNumericTable<algorithmFPType>::create(nQuantileOrders, nFeatures,
                                                                                data_management::NumericTable::doAllocate, &s));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

}// namespace interface1
}// namespace quantiles
}// namespace algorithms
}// namespace daal
