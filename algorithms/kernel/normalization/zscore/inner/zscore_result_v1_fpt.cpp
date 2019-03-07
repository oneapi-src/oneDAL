/* file: zscore_result_v1_fpt.cpp */
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
//  Implementation of zscore algorithm and types methods.
//--
*/

#include "zscore_result_v1.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{

namespace interface1
{
/**
* Allocates memory to store final results of the z-score normalization algorithms
* \param[in] input     Input objects for the z-score normalization algorithm
* \param[in] method    Algorithm computation method
*/
template <typename algorithmFPType>
Status ResultImpl::allocate(const daal::algorithms::Input *input)
{
    const Input *in = static_cast<const Input *>(input);
    DAAL_CHECK(in, ErrorNullInput);

    NumericTablePtr dataTable = in->get(zscore::data);
    DAAL_CHECK(dataTable, ErrorNullInputNumericTable);

    const size_t nFeatures = dataTable->getNumberOfColumns();
    const size_t nVectors = dataTable->getNumberOfRows();

    Status status;
    (*this)[normalizedData] = HomogenNumericTable<algorithmFPType>::create
            (nFeatures, nVectors, NumericTable::doAllocate, &status);
    return status;
}

template DAAL_EXPORT Status ResultImpl::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input);

}// namespace interface1

}// namespace zscore
}// namespace normalization
}// namespace algorithms
}// namespace daal
