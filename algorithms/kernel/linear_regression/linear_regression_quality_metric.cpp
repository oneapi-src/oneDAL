/* file: linear_regression_quality_metric.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of linear regression quality metric methods.
//--
*/

#include "linear_regression_quality_metric_set_batch.h"
#include "linear_regression_single_beta_batch.h"
#include "linear_regression_single_beta_dense_default_batch_container.h"
#include "linear_regression_group_of_betas_dense_default_batch_container.h"
#include "linear_regression_ne_model.h"
#include "linear_regression_qr_model.h"
#include "daal_defines.h"

#define DAAL_CHECK_TABLE(type, error) DAAL_CHECK(get(type).get(), error);


using namespace daal::data_management;
using namespace daal::services;


namespace daal
{
namespace algorithms
{
namespace linear_regression
{

namespace quality_metric_set
{
namespace interface1
{

Parameter::Parameter(size_t nBeta, size_t nBetaReducedModel, double alphaVal, double accuracyVal):
        numBeta(nBeta), numBetaReducedModel(nBetaReducedModel), alpha(alphaVal), accuracyThreshold(accuracyVal) {}

void Parameter::check() const
{
    DAAL_CHECK((alpha >= 0) && (alpha <= 1), ErrorSignificanceLevel);
    DAAL_CHECK(accuracyThreshold > 0., ErrorAccuracyThreshold);
    DAAL_CHECK(numBeta > 0, ErrorIncorrectNumberOfBetas);
    DAAL_CHECK((numBetaReducedModel > 0) && (numBetaReducedModel < numBeta), ErrorIncorrectNumberOfBetasInReducedModel);
}

void Batch::initializeQualityMetrics()
{
    inputAlgorithms[singleBeta] = SharedPtr<linear_regression::quality_metric::single_beta::Batch<> >(
        new linear_regression::quality_metric::single_beta::Batch<>());
    _inputData->add(singleBeta, SharedPtr<linear_regression::quality_metric::single_beta::Input>(
        new linear_regression::quality_metric::single_beta::Input));

    inputAlgorithms[groupOfBetas] = SharedPtr<linear_regression::quality_metric::group_of_betas::Batch<> >(
        new linear_regression::quality_metric::group_of_betas::Batch<>(parameter.numBeta, parameter.numBetaReducedModel));
    _inputData->add(groupOfBetas, SharedPtr<linear_regression::quality_metric::group_of_betas::Input>(
        new linear_regression::quality_metric::group_of_betas::Input));
}

/**
 * Returns the result of the quality metrics algorithm
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
SharedPtr<algorithms::Result> ResultCollection::getResult(QualityMetricId id) const
{
    return staticPointerCast<Result, SerializationIface>((*this)[(size_t)id]);
}

/**
 * Returns the input object of the quality metrics algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
SharedPtr<algorithms::Input> InputDataCollection::getInput(QualityMetricId id) const
{
    return algorithms::quality_metric_set::InputDataCollection::getInput((size_t)id);
}

}//namespace interface1
}//namespace quality_metric_set

namespace quality_metric
{
namespace single_beta
{
namespace interface1
{

Parameter::Parameter(double alphaVal, double accuracyVal): alpha(alphaVal), accuracyThreshold(accuracyVal) {}

/** Default constructor */
Input::Input() : daal::algorithms::Input(3) {}

/**
* Returns an input object for linear regression quality metric
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
data_management::NumericTablePtr Input::get(DataInputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
* Sets an input object for linear regression quality metric
* \param[in] id      Identifier of the input object
* \param[in] value   Pointer to the object
*/
void Input::set(DataInputId id, const data_management::NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
* Returns an input object representing linear regression model
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
services::SharedPtr<linear_regression::Model> Input::get(ModelInputId id) const
{
    return services::staticPointerCast<linear_regression::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
* Sets an input object representing linear regression model
* \param[in] id      Identifier of the input object
* \param[in] value   %Input object
*/
void Input::set(ModelInputId id, const services::SharedPtr<linear_regression::Model> &value)
{
    Argument::set(id, value);
}

/**
* Checks an input object for the linear regression algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
    */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 3, ErrorIncorrectNumberOfInputNumericTables);

    NumericTablePtr yTable = get(expectedResponses);
    NumericTablePtr zTable = get(predictedResponses);

    DAAL_CHECK(yTable, ErrorNullInputNumericTable);
    const size_t n = yTable->getNumberOfRows();
    const size_t k = yTable->getNumberOfColumns();
    DAAL_CHECK(n, ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(k, ErrorIncorrectNumberOfColumnsInInputNumericTable);

    DAAL_CHECK(zTable, ErrorNullInputNumericTable);
    const size_t n1 = zTable->getNumberOfRows();
    const size_t k1 = zTable->getNumberOfColumns();
    DAAL_CHECK(n1, ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(k1, ErrorIncorrectNumberOfColumnsInInputNumericTable);

    DAAL_CHECK(n1 == n, ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(k1 == k, ErrorIncorrectNumberOfColumnsInInputNumericTable);

    SharedPtr<linear_regression::Model> beta = get(model);
    DAAL_CHECK(beta, ErrorNullModel);
    NumericTablePtr betaTable = beta->getBeta();
    DAAL_CHECK(betaTable, ErrorNullModel);
    const size_t nRowsInModelTable = betaTable->getNumberOfRows();
    DAAL_CHECK(k == nRowsInModelTable, ErrorIncorrectNumberOfRowsInInputNumericTable);
}


Result::Result() : daal::algorithms::Result(6) {};

/**
* Returns the result of linear regression quality metrics
* \param[in] id    Identifier of the result
* \return          Result that corresponds to the given identifier
*/
data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
* Sets the result of linear regression quality metrics
* \param[in] id      Identifier of the input object
* \param[in] value   %Input object
*/
void Result::set(ResultId id, const data_management::NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
* Returns the result of linear regression quality metrics
* \param[in] id    Identifier of the result
* \param[in] index Index in result collection
* \return          Result that corresponds to the given identifier
*/
data_management::NumericTablePtr Result::get(ResultDataCollectionId id, size_t index) const
{
    data_management::DataCollectionPtr coll = this->get(id);
    return services::staticPointerCast<NumericTable, SerializationIface>((*coll)[index]);
}

/**
* Returns the result of linear regression quality metrics
* \param[in] id    Identifier of the result
* \return          Result that corresponds to the given identifier
*/
data_management::DataCollectionPtr Result::get(ResultDataCollectionId id) const
{
    return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
}

/**
* Sets the result of linear regression quality metrics
* \param[in] id      Identifier of the input object
* \param[in] value   %Input object
*/
void Result::set(ResultDataCollectionId id, const data_management::DataCollectionPtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the result of linear regression quality metrics
 * \param[in] input   %Input object
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 6, ErrorIncorrectNumberOfElementsInResultCollection);

    DAAL_CHECK_TABLE(rms, ErrorNullOutputNumericTable);
    DAAL_CHECK_TABLE(variance, ErrorNullOutputNumericTable);
    DAAL_CHECK_TABLE(zScore, ErrorNullOutputNumericTable);
    DAAL_CHECK_TABLE(confidenceIntervals, ErrorNullOutputNumericTable);
    DAAL_CHECK_TABLE(inverseOfXtX, ErrorNullOutputNumericTable);

    const Input* inp = dynamic_cast<const Input*>(input);
    DAAL_CHECK(inp, ErrorNullInput);

    DataCollectionPtr coll = get(betaCovariances);
    DAAL_CHECK(coll.get(), ErrorNullInput);

    DAAL_CHECK(coll->size() == inp->get(expectedResponses)->getNumberOfColumns(), ErrorIncorrectNumberOfElementsInResultCollection);
    const auto nBeta = inp->get(model)->getNumberOfBetas();
    for(auto i = 0; i < coll->size(); ++i)
    {
        auto it = (*coll)[i];
        NumericTable* tbl = dynamic_cast<NumericTable*>(it.get());
        DAAL_CHECK(tbl, ErrorNullOutputNumericTable);
        DAAL_CHECK(tbl->getNumberOfColumns() == nBeta, ErrorIncorrectNumberOfColumnsInOutputNumericTable);
        DAAL_CHECK(tbl->getNumberOfRows() == nBeta, ErrorIncorrectNumberOfRowsInOutputNumericTable);
    }
}

void Parameter::check() const
{
    DAAL_CHECK((alpha >= 0) && (alpha <= 1), ErrorSignificanceLevel);
    DAAL_CHECK(accuracyThreshold > 0., ErrorAccuracyThreshold);
}

}//namespace interface1

namespace internal
{

SingleBetaOutput::SingleBetaOutput(size_t nResponses) : rms(nullptr), variance(nullptr),
betaCovariances(nullptr), zScore(nullptr), confidenceIntervals(nullptr), inverseOfXtX(nullptr)
{
    betaCovariances = new NumericTable*[nResponses];
    for(size_t i = 0; i < nResponses; ++i)
        betaCovariances[i] = nullptr;
}

SingleBetaOutput::~SingleBetaOutput()
{
    delete[] betaCovariances;
}

const NumericTable* getXtXTable(const linear_regression::Model& model, bool& bModelNe)
{
    ModelNormEq* modelNe = dynamic_cast<ModelNormEq*>(const_cast<linear_regression::Model*>(&model));
    if(modelNe)
    {
        bModelNe = true;
        return modelNe->getXTXTable().get();
    }
    bModelNe = false;
    ModelQR* modelQr = dynamic_cast<ModelQR*>(const_cast<linear_regression::Model*>(&model));
    return modelQr ? modelQr->getRTable().get() : nullptr;
}

}//namespace internal

}//namespace single_beta

namespace group_of_betas
{
namespace interface1
{

void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter* prm = dynamic_cast<const Parameter*>(par);
    DAAL_CHECK(prm->numBeta > 0, ErrorIncorrectNumberOfFeatures); //TODO
    DAAL_CHECK((prm->numBetaReducedModel > 0) && (prm->numBetaReducedModel < prm->numBeta), ErrorIncorrectNumberOfFeatures); //TODO
    DAAL_CHECK(Argument::size() == 3, ErrorIncorrectNumberOfInputNumericTables);

    NumericTablePtr yTable = get(expectedResponses);
    NumericTablePtr zTable = get(predictedResponses);
    NumericTablePtr zReducedTable = get(predictedResponses);

    DAAL_CHECK(yTable, ErrorNullInputNumericTable);
    const size_t n = yTable->getNumberOfRows();
    const size_t k = yTable->getNumberOfColumns();
    DAAL_CHECK(n, ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(k, ErrorIncorrectNumberOfColumnsInInputNumericTable);

    DAAL_CHECK(prm->numBeta < n, ErrorIncorrectNumberOfFeatures); //TODO
    DAAL_CHECK(prm->numBetaReducedModel < n, ErrorIncorrectNumberOfFeatures); //TODO

    DAAL_CHECK(zTable, ErrorNullInputNumericTable);
    const size_t n1 = zTable->getNumberOfRows();
    const size_t k1 = zTable->getNumberOfColumns();
    DAAL_CHECK(n1, ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(k1, ErrorIncorrectNumberOfColumnsInInputNumericTable);

    DAAL_CHECK(n1 == n, ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(k1 == k, ErrorIncorrectNumberOfColumnsInInputNumericTable);

    DAAL_CHECK(zReducedTable, ErrorNullInputNumericTable);
    const size_t n2 = zReducedTable->getNumberOfRows();
    const size_t k2 = zReducedTable->getNumberOfColumns();
    DAAL_CHECK(n2, ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(k2, ErrorIncorrectNumberOfColumnsInInputNumericTable);

    DAAL_CHECK(n2 == n, ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(k2 == k, ErrorIncorrectNumberOfColumnsInInputNumericTable);
}

void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 7, ErrorIncorrectNumberOfElementsInResultCollection);

    const Input* inp = dynamic_cast<const Input*>(input);
    const size_t k = inp->get(expectedResponses)->getNumberOfColumns();

    for(size_t i = 0; i < 7; ++i)
    {
        DAAL_CHECK_TABLE(ResultId(i), ErrorNullOutputNumericTable);
        auto it = get(ResultId(i));
        DAAL_CHECK(it->getNumberOfColumns() == k, ErrorIncorrectNumberOfColumnsInOutputNumericTable);
        DAAL_CHECK(it->getNumberOfRows() == 1, ErrorIncorrectNumberOfRowsInOutputNumericTable);
    }
}

void Parameter::check() const
{
    DAAL_CHECK(accuracyThreshold > 0., ErrorAccuracyThreshold);
    DAAL_CHECK(numBeta > 0, ErrorIncorrectNumberOfBetas);
    DAAL_CHECK((numBetaReducedModel > 0) && (numBetaReducedModel < numBeta), ErrorIncorrectNumberOfBetasInReducedModel);
}

}//namespace interface1
}//namespace group_of_betas

}//namespace quality_metric
}//namespace linear_regression
}// namespace algorithms
}// namespace daal
