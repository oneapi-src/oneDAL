/* file: linear_regression_training_distributed_input.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of linear regression algorithm classes.
//--
*/

#include "algorithms/linear_regression/linear_regression_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace interface1
{
DistributedInput<step2Master>::DistributedInput() : daal::algorithms::Input(lastStep2MasterInputId + 1)
{
    Argument::set(partialModels, DataCollectionPtr(new DataCollection()));
};
/**
 * Gets an input object for linear regression model-based training
 * in the second step of the distributed processing mode
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput<step2Master>::get(Step2MasterInputId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}
/**
 * Sets an input object for linear regression model-based training
 * in the second step of the distributed processing mode
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   %Input object
 */
void DistributedInput<step2Master>::set(Step2MasterInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}
/**
 Adds an input object for linear regression model-based training in the second step
 * of the distributed processing mode
 * \param[in] id      Identifier of the input object
 * \param[in] partialResult   %Input object
 */
void DistributedInput<step2Master>::add(Step2MasterInputId id, const PartialResultPtr & partialResult)
{
    DataCollectionPtr collection = staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
    collection->push_back(staticPointerCast<SerializationIface, linear_regression::Model>(partialResult->get(training::partialModel)));
}
/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t DistributedInput<step2Master>::getNumberOfFeatures() const
{
    DataCollectionPtr partialModelsCollection = static_cast<DataCollectionPtr>(get(partialModels));
    if (partialModelsCollection->size() == 0)
    {
        return 0;
    }
    linear_regression::Model * partialModel = static_cast<daal::algorithms::linear_regression::Model *>(((*partialModelsCollection)[0]).get());
    return partialModel->getNumberOfFeatures();
}
/**
 * Returns the number of dependent variables
 * \return Number of dependent variables
 */
size_t DistributedInput<step2Master>::getNumberOfDependentVariables() const
{
    DataCollectionPtr partialModelsCollection = static_cast<DataCollectionPtr>(get(partialModels));
    if (partialModelsCollection->size() == 0)
    {
        return 0;
    }
    linear_regression::Model * partialModel = static_cast<daal::algorithms::linear_regression::Model *>(((*partialModelsCollection)[0]).get());
    return partialModel->getNumberOfResponses();
}
/**
 * Checks an input object for linear regression model-based training in the second step
 * of the distributed processing mode
 */
services::Status DistributedInput<step2Master>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DataCollectionPtr collection = DataCollection::cast(get(partialModels));
    DAAL_CHECK(collection, ErrorNullInputDataCollection);

    size_t nBlocks = collection->size();
    DAAL_CHECK(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables);

    DAAL_CHECK((*collection)[0], ErrorNullModel);
    linear_regression::ModelPtr firstPartialModel = linear_regression::Model::cast((*collection)[0]);
    DAAL_CHECK(firstPartialModel, ErrorIncorrectElementInPartialResultCollection);

    size_t nBeta      = firstPartialModel->getNumberOfBetas();
    size_t nResponses = firstPartialModel->getNumberOfResponses();

    services::Status s;

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK((*collection)[i], ErrorNullModel);
        linear_regression::ModelPtr partialModel = linear_regression::Model::cast((*collection)[i]);

        DAAL_CHECK(partialModel, ErrorIncorrectElementInPartialResultCollection);

        DAAL_CHECK_STATUS(s, linear_regression::checkModel(partialModel.get(), *parameter, nBeta, nResponses, method));
    }

    return s;
}

} // namespace interface1
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
