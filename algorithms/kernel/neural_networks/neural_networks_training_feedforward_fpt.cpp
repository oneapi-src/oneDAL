/* file: neural_networks_training_feedforward_fpt.cpp */
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
//  Implementation of common functions for optimization solver
//  used in neural network
//--
*/

#include "neural_networks_training_feedforward.h"
#include "data_management/data/homogen_numeric_table.h"

using namespace daal::services;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;

template<typename algorithmFPType>
daal::algorithms::neural_networks::internal::Solver<algorithmFPType>::Solver():
    precomputed(new ObjectiveFunction()),
    nIterations(new HomogenNumericTable<int>(1, 1, NumericTable::doAllocate, 0))
{}

template<typename algorithmFPType>
daal::algorithms::neural_networks::internal::Solver<algorithmFPType>::~Solver()
{}

template<typename algorithmFPType>
void daal::algorithms::neural_networks::internal::Solver<algorithmFPType>::init(const IterativeSolverPtr &_solver)
{
    solver = _solver->clone();

    solver->parameter->function = precomputed;
    solver->parameter->nIterations = 1;

    solver->createResult();
    solverResult = solver->getResult();
    solverResult->set(iterative_solver::nIterations, nIterations);
}

template<typename algorithmFPType>
SharedPtr<KernelErrorCollection> daal::algorithms::neural_networks::internal::Solver<algorithmFPType>::updateWeightsAndBiases(
            const NumericTablePtr &weightsAndBiases,
            const NumericTablePtr &weightsAndBiasesDerivatives)
{
    auto precomputedResult = precomputed->getResult();
    precomputedResult->set(objective_function::gradientIdx, weightsAndBiasesDerivatives);
    precomputed->setResult(precomputedResult);
    solver->input->set(iterative_solver::inputArgument, weightsAndBiases);
    solverResult->set(iterative_solver::minimum, weightsAndBiases);

    solver->input->set(iterative_solver::optionalArgument,
        solverResult->get(iterative_solver::optionalResult));
    solver->computeNoThrow();
    return solver->getErrors()->getErrors();
}

template<typename algorithmFPType>
NumericTablePtr daal::algorithms::neural_networks::internal::Solver<algorithmFPType>::getMinimum()
{
    return solverResult->get(iterative_solver::minimum);
}

template DAAL_EXPORT daal::algorithms::neural_networks::internal::Solver<DAAL_FPTYPE>::Solver();
template DAAL_EXPORT daal::algorithms::neural_networks::internal::Solver<DAAL_FPTYPE>::~Solver();
template DAAL_EXPORT SharedPtr<KernelErrorCollection> daal::algorithms::neural_networks::internal::Solver<DAAL_FPTYPE>::updateWeightsAndBiases(
                const NumericTablePtr &weightsAndBiases,
                const NumericTablePtr &weightsAndBiasesDerivatives);
template DAAL_EXPORT NumericTablePtr daal::algorithms::neural_networks::internal::Solver<DAAL_FPTYPE>::getMinimum();
template DAAL_EXPORT void daal::algorithms::neural_networks::internal::Solver<DAAL_FPTYPE>::init(
                const services::SharedPtr<iterative_solver::Batch> &_solver);
