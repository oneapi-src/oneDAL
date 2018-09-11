/* file: implicit_als_train_init_parameter.h */
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
//  Implementation of auxiliary implicit als methods.
//--
*/

#include "algorithms/implicit_als/implicit_als_training_init_types.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{
services::SharedPtr<data_management::HomogenNumericTable<int> > getPartition(const init::DistributedParameter *parameter, services::Status &st);

}// namespace internal
}// namespace init
}// namespace training
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
