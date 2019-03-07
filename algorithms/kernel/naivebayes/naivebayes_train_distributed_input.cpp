/* file: naivebayes_train_distributed_input.cpp */
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
//  Implementation of input for multinomial naive bayes training algorithm
//  in distributed computing mode.
//--
*/

#include "multinomial_naive_bayes_training_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{
namespace interface1
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
void DistributedInput::add(const Step2MasterInputId &id, const PartialResultPtr &partialResult)
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
void DistributedInput::set(Step2MasterInputId id, const DataCollectionPtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks input parameters in the training stage of the classification algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Algorithm method
 */
Status DistributedInput::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status status;

    DataCollectionPtr collection = DataCollection::cast(get(partialModels));
    DAAL_CHECK_EX(collection, ErrorNullInputDataCollection, ArgumentName, inputCollectionStr());

    const size_t size = collection->size();
    DAAL_CHECK_EX(size > 0, ErrorEmptyInputCollection, ArgumentName, numberOfModelsStr());

    const multinomial_naive_bayes::Parameter *algPar = static_cast<const multinomial_naive_bayes::Parameter *>(parameter);
    size_t nFeatures=0;

    auto checkModel = [&](const SerializationIfacePtr &model) -> services::Status
    {
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

        s |= checkNumericTable(partialModel->getClassSize().get(), classSizeStr(), 0, 0, 1, algPar->nClasses);
        s |= checkNumericTable(partialModel->getClassGroupSum().get(), groupSumStr(), 0, 0, nFeatures, algPar->nClasses);
        return s;
    };

    for (size_t i = 0; i < size; i++)
    {
        status|=checkModel((*collection)[i]);
    }
    return status;
}
} // namespace interface1
} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
