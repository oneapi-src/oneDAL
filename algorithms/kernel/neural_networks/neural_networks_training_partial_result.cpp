/* file: neural_networks_training_partial_result.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
