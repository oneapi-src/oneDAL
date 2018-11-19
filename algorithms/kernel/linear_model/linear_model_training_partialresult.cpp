/* file: linear_model_training_partialresult.cpp */
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
//  Implementation of the class defining the partial result of the linear model training algorithm
//--
*/

#include "services/daal_defines.h"
#include "algorithms/linear_model/linear_model_training_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace training
{
namespace interface1
{
PartialResult::PartialResult(size_t nElements) : regression::training::PartialResult(nElements)
{}
}
}
}
}
}
