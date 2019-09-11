/* file: neural_networks_training_partial_result.cpp */
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

#include "neural_networks_training_partial_result.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace training
{
PartialResult::PartialResult() : daal::algorithms::PartialResult(lastStep1LocalPartialResultId + 1)
{}

NumericTablePtr PartialResult::get(Step1LocalPartialResultId id) const
{
    return NumericTable::cast(Argument::get(id));
}

void PartialResult::set(Step1LocalPartialResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

Status PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    return checkNumericTable(get(batchSize).get(), batchSizeStr(), 0, 0, 1, 1);
}

DistributedPartialResult::DistributedPartialResult() : daal::algorithms::PartialResult(lastStep2MasterPartialResultId + 1)
{
    set(resultFromMaster, training::ResultPtr(new Result()));
}

training::ResultPtr DistributedPartialResult::get(Step2MasterPartialResultId id) const
{
    return Result::cast(Argument::get(id));
}

void DistributedPartialResult::set(Step2MasterPartialResultId id, const training::ResultPtr &value)
{
    Argument::set(id, value);
}

Status DistributedPartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    return Status();
}

}
}
}
}
