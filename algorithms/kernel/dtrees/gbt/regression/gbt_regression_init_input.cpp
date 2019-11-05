/* file: gbt_regression_init_input.cpp */
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
#include "daal_defines.h"
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

DistributedInput<step1Local>::DistributedInput() : daal::algorithms::Input(lastStep1LocalInputId + 1) {}

/**
 * Returns an input object for the model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step1Local>::get(Step1LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Sets an input object for the model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step1Local>::set(Step1LocalNumericTableInputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the input objects and parameters of the model-based training
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step1Local>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *par = static_cast<const Parameter *>(parameter);

    return services::Status();
}

DistributedInput<step2Master>::DistributedInput() : daal::algorithms::Input(lastStep2MasterCollectionInputId + 1)
{
    Argument::set(step2MeanDependentVariable, DataCollectionPtr(new DataCollection()));
    Argument::set(step2NumberOfRows, DataCollectionPtr(new DataCollection()));
    Argument::set(step2BinBorders, DataCollectionPtr(new DataCollection()));
    Argument::set(step2BinSizes, DataCollectionPtr(new DataCollection()));
}

/**
 * Returns an input object for the model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step2Master>::get(Step2MasterCollectionInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step2Master>::set(Step2MasterCollectionInputId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step2Master>::add(Step2MasterCollectionInputId id, const NumericTablePtr &ptr)
{
    data_management::DataCollectionPtr collection =
            services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection) { return; }
    if (!ptr)        { return; }
    collection->push_back(ptr);
}

/**
 * Checks the input objects and parameters of the model-based training
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step2Master>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *par = static_cast<const Parameter *>(parameter);

    return services::Status();
}

DistributedInput<step3Local>::DistributedInput() : daal::algorithms::Input(lastStep3LocalInputId + 1) {}

/**
 * Returns an input object for the model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr DistributedInput<step3Local>::get(Step3LocalNumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
 * Sets an input object for the model-based training
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step3Local>::set(Step3LocalNumericTableInputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the input objects and parameters of the model-based training
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step3Local>::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *par = static_cast<const Parameter *>(parameter);

    return services::Status();
}

} // namespace interface1
} // namespace init
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
