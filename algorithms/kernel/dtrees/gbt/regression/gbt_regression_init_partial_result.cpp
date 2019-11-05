/* file: gbt_regression_init_partial_result.cpp */
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

/*
//++
//  Implementation of gbt regression classes.
//--
*/

#include "gbt_regression_init_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace init
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep1, SERIALIZATION_GBT_REGRESSION_TRAINING_INIT_PARTIAL_RESULT_STEP1_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep2, SERIALIZATION_GBT_REGRESSION_TRAINING_INIT_PARTIAL_RESULT_STEP2_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep3, SERIALIZATION_GBT_REGRESSION_TRAINING_INIT_PARTIAL_RESULT_STEP3_ID);


DistributedPartialResultStep1::DistributedPartialResultStep1() : daal::algorithms::PartialResult(lastDistributedPartialResultStep1Id + 1) {}

NumericTablePtr DistributedPartialResultStep1::get(DistributedPartialResultStep1Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep1::set(DistributedPartialResultStep1Id id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep1::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const DistributedInput<step1Local> *algInput = static_cast<const DistributedInput<step1Local> *>(input);

    return Status();
}


DistributedPartialResultStep2::DistributedPartialResultStep2() : daal::algorithms::PartialResult(lastDistributedPartialResultStep2CollectionId + 1) {}

NumericTablePtr DistributedPartialResultStep2::get(DistributedPartialResultStep2NumericTableId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep2::set(DistributedPartialResultStep2NumericTableId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

DataCollectionPtr DistributedPartialResultStep2::get(DistributedPartialResultStep2CollectionId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep2::set(DistributedPartialResultStep2CollectionId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep2::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const DistributedInput<step2Master> *algInput = static_cast<const DistributedInput<step2Master> *>(input);

    return Status();
}


DistributedPartialResultStep3::DistributedPartialResultStep3() : daal::algorithms::PartialResult(lastDistributedPartialResultStep3Id + 1) {}

NumericTablePtr DistributedPartialResultStep3::get(DistributedPartialResultStep3Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep3::set(DistributedPartialResultStep3Id id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep3::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const DistributedInput<step3Local> *algInput = static_cast<const DistributedInput<step3Local> *>(input);

    return Status();
}

} // namespace interface1
} // namespace init
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
