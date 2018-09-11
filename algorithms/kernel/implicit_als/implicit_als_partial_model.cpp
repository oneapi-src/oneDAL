/* file: implicit_als_partial_model.cpp */
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
//  Implementation of the class defining the implicit als model
//--
*/

#include "implicit_als_model.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
DAAL_EXPORT PartialModel::PartialModel()
{}

DAAL_EXPORT PartialModel::PartialModel(data_management::NumericTablePtr factors,
                                       data_management::NumericTablePtr indices) :
    _factors(factors), _indices(indices)
{}

}// namespace implicit_als
}// namespace algorithms
}// namespace daal
