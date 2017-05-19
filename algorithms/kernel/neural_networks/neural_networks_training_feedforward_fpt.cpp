/* file: neural_networks_training_feedforward_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace internal
{
using namespace daal::services;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;

template<typename algorithmFPType>
Solver<algorithmFPType>::Solver():
    precomputed(new ObjectiveFunction()),
    nIterations(new HomogenNumericTable<int>(1, 1, NumericTable::doAllocate, 0))
{}

template<typename algorithmFPType>
Solver<algorithmFPType>::~Solver()
{}

template<typename algorithmFPType>
Status Solver<algorithmFPType>::init(const IterativeSolverPtr &_solver)
{
    DAAL_CHECK_MALLOC(precomputed.get())
    DAAL_CHECK_MALLOC(nIterations.get())

    solver = _solver->clone();

    solver->parameter->function = precomputed;

    _nIterationSolver = solver->parameter->nIterations;
    _batchSize = solver->parameter->batchSize;

    solver->parameter->optionalResultRequired = true;

    Status s;
    DAAL_CHECK_STATUS(s, solver->createResult());
    solverResult = solver->getResult();
    solverResult->set(iterative_solver::nIterations, nIterations);
    return s;
}

template<typename algorithmFPType>
Status Solver<algorithmFPType>::updateWeightsAndBiases(
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
    solver->parameter->nIterations = 1;
    solver->parameter->batchSize = 1;
    return solver->computeNoThrow();
}

template<typename algorithmFPType>
NumericTablePtr Solver<algorithmFPType>::getMinimum()
{
    return solverResult->get(iterative_solver::minimum);
}


template DAAL_EXPORT Solver<DAAL_FPTYPE>::Solver();
template DAAL_EXPORT Solver<DAAL_FPTYPE>::~Solver();
template DAAL_EXPORT Status Solver<DAAL_FPTYPE>::updateWeightsAndBiases(
                const NumericTablePtr &weightsAndBiases,
                const NumericTablePtr &weightsAndBiasesDerivatives);
template DAAL_EXPORT NumericTablePtr Solver<DAAL_FPTYPE>::getMinimum();
template DAAL_EXPORT Status Solver<DAAL_FPTYPE>::init(const SharedPtr<iterative_solver::Batch> &_solver);
}
}
}
}
