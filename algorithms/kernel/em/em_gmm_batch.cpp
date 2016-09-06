/* file: em_gmm_batch.cpp */
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
//  Implementation of the EM for GMM algorithm interface.
//--
*/

#include "em_gmm_types.h"
#include "daal_defines.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace interface1
{

/**
 * Constructs the parameter of EMM for GMM algorithm
 * \param[in] nComponents              Number of components in the Gaussian mixture model
 * \param[in] maxIterations            Maximal number of iterations of the algorithm
 * \param[in] accuracyThreshold        Threshold for the termination of the algorithm
 * \param[in] covariance               Pointer to the algorithm that computes the covariance
 */
Parameter::Parameter(const size_t nComponents,
          const SharedPtr<covariance::BatchIface> &covariance,
          const size_t maxIterations,
          const double accuracyThreshold) :
    nComponents(nComponents),
    maxIterations(maxIterations),
    accuracyThreshold(accuracyThreshold),
    covariance(covariance)
{}

Parameter::Parameter(const Parameter &other) :
    nComponents(other.nComponents),
    maxIterations(other.maxIterations),
    accuracyThreshold(other.accuracyThreshold),
    covariance(other.covariance)
{}

void Parameter::check() const
{
    DAAL_CHECK_EX(accuracyThreshold >= 0, ErrorEMIncorrectToleranceToConverge, ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations > 0, ErrorEMIncorrectMaxNumberOfIterations, ParameterName, maxIterationsStr());
    DAAL_CHECK_EX(nComponents > 0, ErrorEMIncorrectNumberOfComponents, ParameterName, nComponentsStr());
    DAAL_CHECK_EX(covariance, ErrorNullAuxiliaryAlgorithm, ParameterName, covarianceStr());
}

/** Default constructor */
Input::Input() : daal::algorithms::Input(4)
{}

/**
 * Sets one input object for the EM for GMM algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input covariance object for the EM for GMM algorithm
 * \param[in] id    Identifier of the input covariance collection object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputCovariancesId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets input objects for the EM for GMM algorithm[IE1]
 * \param[in] id    Identifier of the input values object. Result of the EM for GMM initialization algorithm can be used.
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputValuesId id, const SharedPtr<init::Result> &ptr)
{
    if(ptr)
    {
        set(inputWeights,     ptr->get(init::weights));
        set(inputMeans,       ptr->get(init::means));
        set(inputCovariances, ptr->get(init::covariances));
    }
}

/**
 * Returns the input numeric table for the EM for GMM algorithm
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Returns the collection of input covariances for the EM for GMM algorithm
 * \param[in] id    Identifier of the  collection of input covariances
 * \return          %Input object that corresponds to the given identifier
 */
DataCollectionPtr Input::get(InputCovariancesId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
 * Returns a covariance with a given index from the collection of input covariances
 * \param[in] id    Identifier of the collection of input covariances
 * \param[in] index Index of the covariance to be returned
 * \return          Pointer to the table with the input covariance
 */
NumericTablePtr Input::get(InputCovariancesId id, size_t index) const
{
    DataCollectionPtr covCollection = this->get(id);
    return staticPointerCast<NumericTable, SerializationIface>((*covCollection)[index]);
}

/**
 * Checks the correctness of the input result
 * \param[in] par       Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    if (!checkNumericTable(get(data).get(), this->_errors.get(), dataStr())) { return; }

    Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    size_t nComponents = algParameter->nComponents;
    size_t nFeatures = get(data)->getNumberOfColumns();

    if (!checkNumericTable(get(inputWeights).get(), this->_errors.get(), inputWeightsStr(), 0, 0, nComponents, 1)) { return; }
    if (!checkNumericTable(get(inputMeans).get(), this->_errors.get(), inputMeansStr(), 0, 0, nFeatures, nComponents)) { return; }

    DataCollectionPtr inputCovCollection = get(inputCovariances);
    DAAL_CHECK(inputCovCollection, ErrorNullInputDataCollection);
    DAAL_CHECK(inputCovCollection->size() == nComponents, ErrorIncorrectNumberOfInputNumericTables);

    int unexpectedLayoutCovariance = (int)(NumericTableIface::upperPackedTriangularMatrix | NumericTableIface::lowerPackedTriangularMatrix);
    for(size_t i = 0; i < nComponents; i++)
    {
        SerializationIfacePtr collectionElement = (*inputCovCollection)[i];
        DAAL_CHECK_EX(collectionElement, ErrorNullNumericTable, ArgumentName, inputCovariancesStr());

        NumericTablePtr nt = NumericTable::cast(collectionElement);
        DAAL_CHECK_EX(nt, ErrorIncorrectElementInCollection, ArgumentName, inputCovariancesStr());
        if (!checkNumericTable(nt.get(), this->_errors.get(), inputCovariancesStr(), unexpectedLayoutCovariance, 0, nFeatures, nFeatures)) { return; }
    }
}

/** Default constructor */
Result::Result() : daal::algorithms::Result(5)
{
    Argument::set(covariances, DataCollectionPtr(new DataCollection()));
}

/**
 * Sets the result of the EM for GMM algorithm
 * \param[in] id    %Result identifier
 * \param[in] ptr   Pointer to the numeric table with the result
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the collection of covariances for the EM for GMM algorithm
 * \param[in] id    Identifier of the collection of covariances
 * \param[in] ptr   Pointer to the collection of covariances
 */
void Result::set(ResultCovariancesId id, const DataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the result of the EM for GMM algorithm
 * \param[in] id   %Result identifier
 * \return         %Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Returns the collection of computed covariances of the EM for GMM algorithm
 * \param[in] id   Identifier of the collection of computed covariances
 * \return         Collection of computed covariances that corresponds to the given identifier
 */
DataCollectionPtr Result::get(ResultCovariancesId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
 * Returns the covariance with a given index from the collection of computed covariances
 * \param[in] id    Identifier of the collection of covariances
 * \param[in] index Index of the covariance to be returned
 * \return          Pointer to the table with the computed covariance
 */
NumericTablePtr Result::get(ResultCovariancesId id, size_t index) const
{
    DataCollectionPtr covCollection = this->get(id);
    return staticPointerCast<NumericTable, SerializationIface>((*covCollection)[index]);
}

/**
* Checks the result parameter of the EM for GMM algorithm
* \param[in] input   %Input of the algorithm
* \param[in] par     %Parameter of algorithm
* \param[in] method  Computation method
*/
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));

    size_t nComponents = algParameter->nComponents;
    size_t nFeatures = algInput->get(data)->getNumberOfColumns();

    int unexpectedLayouts = packed_mask;
    if (!checkNumericTable(get(weights).get(), this->_errors.get(), weightsStr(), unexpectedLayouts, 0, nComponents, 1)) { return; }
    if (!checkNumericTable(get(means).get(), this->_errors.get(), meansStr(), unexpectedLayouts, 0, nFeatures, nComponents)) { return; }

    int unexpectedLayoutCSR = (int)NumericTableIface::csrArray;
    if (!checkNumericTable(get(goalFunction).get(), this->_errors.get(), goalFunctionStr(), unexpectedLayoutCSR, 0, 1, 1)) { return; }
    if (!checkNumericTable(get(nIterations).get(), this->_errors.get(), nIterationsStr(), unexpectedLayoutCSR, 0, 1, 1)) { return; }

    DataCollectionPtr resultCovCollection = get(covariances);
    DAAL_CHECK(resultCovCollection, ErrorNullOutputDataCollection);
    DAAL_CHECK(resultCovCollection->size() == nComponents, ErrorIncorrectNumberOfOutputNumericTables);

    int unexpectedLayoutCovariance = (int)(NumericTableIface::csrArray | NumericTableIface::upperPackedTriangularMatrix | NumericTableIface::lowerPackedTriangularMatrix);
    for(size_t i = 0; i < nComponents; i++)
    {
        SerializationIfacePtr collectionElement = (*resultCovCollection)[i];
        DAAL_CHECK_EX(collectionElement, ErrorNullNumericTable, ArgumentName, covarianceStr());

        NumericTablePtr nt = NumericTable::cast(collectionElement);
        DAAL_CHECK_EX(nt, ErrorIncorrectElementInCollection, ArgumentName, covarianceStr());
        if (!checkNumericTable(nt.get(), this->_errors.get(), covarianceStr(), unexpectedLayoutCovariance, 0, nFeatures, nFeatures)) { return; }
    }
}

} // namespace interface1
} // namespace em_gmm
} // namespace algorithms
} // namespace daal
