/* file: naivebayes_train_distributed_input.cpp */
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
//  Implementation of input for multinomial naive bayes training algorithm
//  in distributed computing mode.
//--
*/

#include "algorithms/naive_bayes/multinomial_naive_bayes_training_types.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{
using namespace daal::data_management;
using namespace daal::services;

DistributedInput::DistributedInput() : classifier::training::InputIface(lastStep2MasterInputId + 1)
{
    Argument::set(partialModels, DataCollectionPtr(new DataCollection()));
}

size_t DistributedInput::getNumberOfFeatures() const
{
    DataCollectionPtr models = get(partialModels);
    if (!models) return 0;
    PartialModelPtr firstModel = multinomial_naive_bayes::PartialModel::cast((*models)[0]);
    if (!firstModel) return 0;
    return firstModel->getNFeatures();
}

/**
 * Returns input objects of the classification algorithm in the distributed processing mode
 * \param[in] id    Identifier of the input objects
 * \return          Input object that corresponds to the given identifier
 */
DataCollectionPtr DistributedInput::get(Step2MasterInputId id) const
{
    return DataCollection::cast(Argument::get(id));
}

/**
 * Adds input object on the master node in the training stage of the classification algorithm
 * \param[in] id            Identifier of the input object
 * \param[in] partialResult Pointer to the object
 */
void DistributedInput::add(const Step2MasterInputId & id, const PartialResultPtr & partialResult)
{
    DataCollectionPtr collection = get(id);
    if (!collection) return;
    collection->push_back(partialResult->get(classifier::training::partialModel));
}

/**
 * Sets input object in the training stage of the classification algorithm
 * \param[in] id   Identifier of the object
 * \param[in] value Pointer to the object
 */
void DistributedInput::set(Step2MasterInputId id, const DataCollectionPtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks input parameters in the training stage of the classification algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Algorithm method
 */
Status DistributedInput::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status status;

    DataCollectionPtr collection = DataCollection::cast(get(partialModels));
    DAAL_CHECK_EX(collection, ErrorNullInputDataCollection, ArgumentName, inputCollectionStr());

    const size_t size = collection->size();
    DAAL_CHECK_EX(size > 0, ErrorEmptyInputCollection, ArgumentName, numberOfModelsStr());

    size_t nClasses = 0;

    const multinomial_naive_bayes::Parameter * algPar2 = dynamic_cast<const multinomial_naive_bayes::Parameter *>(parameter);
    if (algPar2) nClasses = algPar2->nClasses;
    DAAL_CHECK_EX(nClasses > 0, ErrorNullParameterNotSupported, ArgumentName, nClassesStr());

    size_t nFeatures = 0;

    auto checkModel = [&](const SerializationIfacePtr & model) -> services::Status {
        services::Status s;
        DAAL_CHECK(model, ErrorNullModel);
        multinomial_naive_bayes::PartialModelPtr partialModel = multinomial_naive_bayes::PartialModel::cast(model);
        DAAL_CHECK(partialModel, ErrorIncorrectElementInPartialResultCollection);

        const size_t trainingDataFeatures = partialModel->getNFeatures();
        DAAL_CHECK(trainingDataFeatures, services::ErrorModelNotFullInitialized);

        if (!nFeatures)
        {
            nFeatures = trainingDataFeatures;
        }

        s |= checkNumericTable(partialModel->getClassSize().get(), classSizeStr(), 0, 0, 1, nClasses);
        s |= checkNumericTable(partialModel->getClassGroupSum().get(), groupSumStr(), 0, 0, nFeatures, nClasses);
        return s;
    };

    for (size_t i = 0; i < size; i++)
    {
        status |= checkModel((*collection)[i]);
    }
    return status;
}

Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s; // Error status

    data_management::NumericTablePtr dataTable = get(classifier::training::data);
    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(dataTable.get(), dataStr()));

    const size_t nRows                           = dataTable->getNumberOfRows();
    data_management::NumericTablePtr labelsTable = get(classifier::training::labels);
    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(labelsTable.get(), labelsStr(), 0, 0, 1, nRows));

    data_management::NumericTablePtr weightsTable = get(classifier::training::weights);
    if (weightsTable)
    {
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(weightsTable.get(), weightsStr(), 0, 0, 1, nRows));
    }

    if (parameter != NULL)
    {
        const daal::algorithms::classifier::Parameter * algParameter2 = dynamic_cast<const daal::algorithms::classifier::Parameter *>(parameter);
        if (algParameter2 != NULL)
        {
            DAAL_CHECK_EX((algParameter2->nClasses > 1) && (algParameter2->nClasses < INT_MAX), services::ErrorIncorrectParameter,
                          services::ParameterName, nClassesStr());
        }
        else
        {
            s = services::Status(services::ErrorNullParameterNotSupported);
        }
    }

    return s;
}

} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
