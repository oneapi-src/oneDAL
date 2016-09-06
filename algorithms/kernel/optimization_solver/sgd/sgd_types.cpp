/* file: sgd_types.cpp */
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
//  Implementation of sgd solver classes.
//--
*/

#include "algorithms/optimization_solver/sgd/sgd_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace interface1
{
/**
 * Constructs the parameter base class of the Stochastic gradient descent algorithm
 * \param[in] function             Objective function represented as sum of functions
 * \param[in] nIterations          Maximal number of iterations of the algorithm
 * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
 * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function.
 *                                 If no indices are provided, the implementation will generate random indices.
 * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
 * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
 */
BaseParameter::BaseParameter(
    const sum_of_functions::BatchPtr& function,
    size_t nIterations,
    double accuracyThreshold,
    data_management::NumericTablePtr batchIndices,
    data_management::NumericTablePtr learningRateSequence,
    size_t seed) :
    optimization_solver::iterative_solver::Parameter(function, nIterations, accuracyThreshold),
    batchIndices(batchIndices),
    learningRateSequence(learningRateSequence),
    seed(seed)
{}

/**
 * Checks the correctness of the parameter
 */
void BaseParameter::check() const
{
    iterative_solver::Parameter::check();

    if(learningRateSequence.get() != NULL)
    {
        services::SharedPtr<services::Error> error(new services::Error());
        if(learningRateSequence->getNumberOfRows() != nIterations && learningRateSequence->getNumberOfRows() != 1)
        {
            error->setId(services::ErrorIncorrectNumberOfObservations);
        }
        if(learningRateSequence->getNumberOfColumns() != 1)
        {
            error->setId(services::ErrorIncorrectNumberOfFeatures);
        }
        if(error->id() != services::NoErrorMessageFound)
        {
            error->addStringDetail(services::ArgumentName, "learningRateSequence");
            this->_errors->add(error);
            return;
        }
    }
}
/**
 * \param[in] function             Objective function represented as sum of functions
 * \param[in] nIterations          Maximal number of iterations of the algorithm
 * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
 * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are
                                   provided, the implementation will generate random indices.
 * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
 * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
 */
Parameter<defaultDense>::Parameter(
    const sum_of_functions::BatchPtr& function,
    size_t nIterations,
    double accuracyThreshold,
    data_management::NumericTablePtr batchIndices,
    data_management::NumericTablePtr learningRateSequence,
    size_t seed) :
    BaseParameter(
        function,
        nIterations,
        accuracyThreshold,
        batchIndices,
        learningRateSequence,
        seed
    )
{}

/**
 * Checks the correctness of the parameter
 */
void Parameter<defaultDense>::check() const
{
    BaseParameter::check();
    if(batchIndices.get() != NULL)
    {
        services::SharedPtr<services::Error> error(new services::Error());
        if(batchIndices->getNumberOfRows() != nIterations)    { error->setId(services::ErrorIncorrectNumberOfObservations); }
        if(batchIndices->getNumberOfColumns() != 1)           { error->setId(services::ErrorIncorrectNumberOfFeatures);     }
        if(error->id() != services::NoErrorMessageFound)
        {
            error->addStringDetail(services::ArgumentName, "batchIndices");
            this->_errors->add(error);
        }
        return;
    }
}

/**
 * Constructs the parameter class of the Stochastic gradient descent algorithm
 * \param[in] function             Objective function represented as sum of functions
 * \param[in] nIterations          Maximal number of iterations of the algorithm
 * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
 * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function. If no indices
                                   are provided, the implementation will generate random indices.
 * \param[in] batchSize            Number of batch indices to compute the stochastic gradient. If batchSize is equal to the number of terms
                                   in objective function then no random sampling is performed, and all terms are used to calculate the gradient.
                                   This parameter is ignored if batchIndices is provided.
 * \param[in] conservativeSequence Numeric table of values of the conservative coefficient sequence
 * \param[in] innerNIterations     Number of inner iterations
 * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
 * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
 */
Parameter<miniBatch>::Parameter(
    const sum_of_functions::BatchPtr& function,
    size_t nIterations,
    double accuracyThreshold,
    data_management::NumericTablePtr batchIndices,
    size_t batchSize,
    data_management::NumericTablePtr conservativeSequence,
    size_t innerNIterations,
    data_management::NumericTablePtr learningRateSequence,
    size_t seed) :
    BaseParameter(
        function,
        nIterations,
        accuracyThreshold,
        batchIndices,
        learningRateSequence,
        seed
    ),
    batchSize(batchSize),
    conservativeSequence(conservativeSequence),
    innerNIterations(innerNIterations)
{}

/**
 * Checks the correctness of the parameter
 */
void Parameter<miniBatch>::check() const
{
    BaseParameter::check();
    if(batchIndices.get() != NULL)
    {
        services::SharedPtr<services::Error> error(new services::Error());
        if(batchIndices->getNumberOfRows() != nIterations) {error->setId(services::ErrorIncorrectNumberOfObservations);}
        if(batchIndices->getNumberOfColumns() != batchSize) {error->setId(services::ErrorIncorrectNumberOfFeatures);}
        if(error->id() != services::NoErrorMessageFound)
        {
            error->addStringDetail(services::ArgumentName, "batchIndices");
            this->_errors->add(error);
        }
        return;
    }

    if(conservativeSequence.get() != NULL)
    {
        services::SharedPtr<services::Error> error(new services::Error());
        if(conservativeSequence->getNumberOfRows() != nIterations && conservativeSequence->getNumberOfRows() != 1)
        {
            error->setId(services::ErrorIncorrectNumberOfObservations);
        }
        if(conservativeSequence->getNumberOfColumns() != 1)
        {
            error->setId(services::ErrorIncorrectNumberOfFeatures);
        }
        if(error->id() != services::NoErrorMessageFound)
        {
            error->addStringDetail(services::ArgumentName, "conservativeSequence");
            this->_errors->add(error);
        }
    }

    if(batchSize > function->sumOfFunctionsParameter->numberOfTerms || batchSize == 0)
    {
        this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, "batchSize"));
    }
}

} // namespace interface1
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
