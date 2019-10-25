/* file: gbt_regression_training_partial_result.cpp */
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
//  Implementation of gradient boosted trees algorithm classes.
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_training_types.h"
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
namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep1,  SERIALIZATION_GBT_REGRESSION_TRAINING_DISTRIBUTED_PARTIAL_RESULT_STEP1_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep2,  SERIALIZATION_GBT_REGRESSION_TRAINING_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep3,  SERIALIZATION_GBT_REGRESSION_TRAINING_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep4,  SERIALIZATION_GBT_REGRESSION_TRAINING_DISTRIBUTED_PARTIAL_RESULT_STEP4_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep5,  SERIALIZATION_GBT_REGRESSION_TRAINING_DISTRIBUTED_PARTIAL_RESULT_STEP5_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep6,  SERIALIZATION_GBT_REGRESSION_TRAINING_DISTRIBUTED_PARTIAL_RESULT_STEP6_ID);


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
    DAAL_CHECK(algInput, services::ErrorNullInput);

//    const size_t nRows = algInput->get(step1OptCoeffs)->getNumberOfRows();

//    NumericTablePtr ntOptCoeffs = get(updatedOptCoeffs);
//    DAAL_CHECK_EX(ntOptCoeffs, ErrorNullNumericTable, ArgumentName, gbtStep1updatedOptCoeffsStr());

    const int unexpectedLayouts = (int)packed_mask;
//    DAAL_CHECK_STATUS_VAR(checkNumericTable(ntOptCoeffs.get(), gbtStep1updatedOptCoeffsStr(), unexpectedLayouts, 0, 2, nRows));

    return Status();
}


DistributedPartialResultStep2::DistributedPartialResultStep2() : daal::algorithms::PartialResult(lastDistributedPartialResultStep2Id + 1) {}

NumericTablePtr DistributedPartialResultStep2::get(DistributedPartialResultStep2Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep2::set(DistributedPartialResultStep2Id id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep2::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const DistributedInput<step2Local> *algInput = static_cast<const DistributedInput<step2Local> *>(input);

    return Status();
}


DistributedPartialResultStep3::DistributedPartialResultStep3() : daal::algorithms::PartialResult(lastDistributedPartialResultStep3Id + 1) {}

DataCollectionPtr DistributedPartialResultStep3::get(DistributedPartialResultStep3Id id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep3::set(DistributedPartialResultStep3Id id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep3::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const DistributedInput<step3Local> *algInput = static_cast<const DistributedInput<step3Local> *>(input);

    return Status();
}


DistributedPartialResultStep4::DistributedPartialResultStep4() : daal::algorithms::PartialResult(lastDistributedPartialResultStep4Id + 1) {}

DataCollectionPtr DistributedPartialResultStep4::get(DistributedPartialResultStep4Id id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep4::set(DistributedPartialResultStep4Id id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep4::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const DistributedInput<step4Local> *algInput = static_cast<const DistributedInput<step4Local> *>(input);

    return Status();
}


DistributedPartialResultStep5::DistributedPartialResultStep5() : daal::algorithms::PartialResult(lastDistributedPartialResultStep5Id + 1) {}

NumericTablePtr DistributedPartialResultStep5::get(DistributedPartialResultStep5Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep5::set(DistributedPartialResultStep5Id id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep5::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const DistributedInput<step5Local> *algInput = static_cast<const DistributedInput<step5Local> *>(input);

    return Status();
}


DistributedPartialResultStep6::DistributedPartialResultStep6() : daal::algorithms::PartialResult(lastDistributedPartialResultStep6Id + 1) {}

gbt::regression::ModelPtr DistributedPartialResultStep6::get(DistributedPartialResultStep6Id id) const
{
    return staticPointerCast<gbt::regression::Model, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep6::set(DistributedPartialResultStep6Id id, const gbt::regression::ModelPtr &ptr)
{
    Argument::set(id, ptr);
}

Status DistributedPartialResultStep6::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const DistributedInput<step6Local> *algInput = static_cast<const DistributedInput<step6Local> *>(input);

    return Status();
}

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
