/* file: linear_model_predict_batch_fpt.cpp */
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
//  Implementation of the regression algorithm interface
//--
*/

#include "algorithms/linear_model/linear_model_predict_types.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{
using namespace daal::services;
using namespace daal::data_management;

template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    const Input *in = static_cast<const Input *>(input);
    size_t nVectors            = in->get(data )->getNumberOfRows();
    size_t nDependentVariables = in->get(model)->getNumberOfResponses();
    Status st;
    set(prediction, HomogenNumericTable<algorithmFPType>::create(nDependentVariables, nVectors, NumericTable::doAllocate, &st));
    return st;
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

} // namespace prediction
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
