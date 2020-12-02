/* file: logistic_regression_online.hpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
!  Content:
!    C++ example of logistic regression training algorithm implementation for online
!    computing mode with SGD Minibatch solver and DPC++ interfaces.
!******************************************************************************/

#include "daal_sycl.h"

namespace daal_dm = daal::data_management;
namespace daal_al = daal::algorithms;
namespace daal_solver = daal::algorithms::optimization_solver;
namespace daal_sgd = daal::algorithms::optimization_solver::sgd;

template <typename Fptype = float>
class LogisticRegressionOnline
{
    using LogLoss = daal_solver::logistic_loss::Batch<Fptype>;
    using LogLossPtr = daal::services::SharedPtr<LogLoss>;

    using CrossEntropyLoss = daal_solver::cross_entropy_loss::Batch<Fptype>;
    using CrossEntropyLossPtr = daal::services::SharedPtr<CrossEntropyLoss>;

    using SGDMinibatchSolver = daal_sgd::Batch<Fptype, daal_sgd::miniBatch>;
    using SGDMinibatchSolverPtr = daal::services::SharedPtr<SGDMinibatchSolver>;

public:
    LogisticRegressionOnline()
        : _batchRowCount(0), _nFeatures(0), _nClasses(0), _nBetas(0), _nBetaCols(0), _isIntercept(false) {}

    // Sets the algorithm parameters
    void setParams(size_t nClasses,
                   size_t nFeatures,
                   bool isIntercept,
                   Fptype l1_penalty,
                   Fptype l2_penalty,
                   size_t batchRowCount,
                   size_t batchIterationCount = 1)
    {
        _batchRowCount = batchRowCount;
        _nClasses = nClasses;
        _nFeatures = nFeatures;
        _isIntercept = isIntercept;

        if(nClasses == 2)
        {
            LogLossPtr l (new LogLoss(_batchRowCount));
            _solver = SGDMinibatchSolverPtr(new SGDMinibatchSolver(l));

            // solver clones LogLoss so we need to retrieve an object it holds
            _logLoss = daal::services::staticPointerCast<LogLoss>(_solver->parameter.function);
            _logLoss->parameter().penaltyL1 = l1_penalty;
            _logLoss->parameter().penaltyL2 = l2_penalty;
            _logLoss->parameter().interceptFlag = isIntercept;
        }
        else
        {
            CrossEntropyLossPtr l (new CrossEntropyLoss(nClasses, _batchRowCount));
            _solver = SGDMinibatchSolverPtr(new SGDMinibatchSolver(l));

            // solver clones LogLoss so we need to retrieve an object it holds
            _crossEntropyLoss = daal::services::staticPointerCast<CrossEntropyLoss>(_solver->parameter.function);
            _crossEntropyLoss->parameter().penaltyL1 = l1_penalty;
            _crossEntropyLoss->parameter().penaltyL2 = l2_penalty;
            _crossEntropyLoss->parameter().interceptFlag = isIntercept;
        }

        _nBetas = nClasses == 2 ? 1 : nClasses;
        _nBetaCols = nFeatures + 1;

        setDefaultState(_nClasses, _nBetas, _nBetaCols);
        _solver->parameter.accuracyThreshold    = 0.0;
        _solver->parameter.nIterations          = batchIterationCount;
        _solver->parameter.innerNIterations     = batchIterationCount;
        _solver->parameter.batchSize            = _batchRowCount;
        _solver->parameter.conservativeSequence = daal_dm::SyclHomogenNumericTable<Fptype>::create(1, 1, daal_dm::NumericTable::doAllocate, 0.0);
    }

    // Sets the parameters of current iteration of learning process
    void setIterationParams(Fptype learningRate)
    {
        {
            daal_dm::BlockDescriptor<Fptype> bd;
            _learningRateTable->getBlockOfRows(0, 1, daal_dm::readWrite, bd);
            auto value = bd.getBlockPtr();
            value[0] = learningRate;
            _learningRateTable->releaseBlockOfRows(bd);
        }
        _solver->parameter.learningRateSequence = _learningRateTable;
    }

    // Sets the input for current iteration
    void setInput(const daal_dm::NumericTablePtr& x, const daal_dm::NumericTablePtr& y)
    {
        if (_nClasses == 2)
        {
            _logLoss->input.set(daal_solver::logistic_loss::data, x);
            _logLoss->input.set(daal_solver::logistic_loss::dependentVariables, y);
        }
        else
        {
            _crossEntropyLoss->input.set(daal_solver::cross_entropy_loss::data, x);
            _crossEntropyLoss->input.set(daal_solver::cross_entropy_loss::dependentVariables, y);
        }
    }

    // Performs the single iteration of training on the data provided by setInput() method
    void compute()
    {
        namespace solver = daal::algorithms::optimization_solver::iterative_solver;

        _solver->input.set(solver::inputArgument, _workPoint);
        _solver->compute();

        auto partialResult = _solver->getResult();
        _workPoint = partialResult->get(solver::minimum);
    }

    // Creates final beta parameter values and resets internal algorithm state to default values
    void finalizeCompute()
    {
        daal_dm::BlockDescriptor<Fptype> bBetas;
        _workPoint->getBlockOfRows(0, _nBetas*_nBetaCols, daal_dm::readOnly, bBetas);
        if (_isIntercept)
        {
            _finalBetas = daal_dm::SyclHomogenNumericTable<Fptype>::create(bBetas.getBuffer(), _nBetaCols, _nBetas);
        }
        else
        {
            _finalBetas = daal_dm::SyclHomogenNumericTable<Fptype>::create(_nFeatures, _nBetas, daal_dm::NumericTableIface::doAllocate);
            daal_dm::BlockDescriptor<Fptype> bFinalBetas;
            _finalBetas->getBlockOfRows(0, _nBetas, daal_dm::writeOnly, bFinalBetas);

            auto src = bBetas.getBlockPtr();
            auto dst = bFinalBetas.getBlockPtr();

            for (size_t i = 0; i < _nBetas; i++)
            {
                for (size_t j = 0; j < _nFeatures; j++)
                {
                    dst[i*_nFeatures + j] = src[i*_nBetaCols + j + 1];
                }
            }

            _finalBetas->releaseBlockOfRows(bFinalBetas);
        }
        _workPoint->releaseBlockOfRows(bBetas);

        setDefaultState(_nClasses, _nBetas, _nBetaCols);
    }

    // Provides beta coefficients from the last iteration
    auto getPartialModel() const
    {
        return _workPoint;
    }

    // Provides final model
    auto getModel() const
    {
        daal::algorithms::logistic_regression::ModelBuilder<Fptype> mBuilder (_nFeatures, _nClasses);
        mBuilder.setBeta(_finalBetas);
        return mBuilder.getModel();
    }

private:
    void setDefaultState(size_t nClasses, size_t nBetas, size_t nBetaCols)
    {
        _learningRateTable = daal_dm::SyclHomogenNumericTable<Fptype>::create(1, 1, daal_dm::NumericTable::doAllocate, 1.0);
        _workPoint  = daal_dm::SyclHomogenNumericTable<Fptype>::create(1, nBetas*nBetaCols, daal_dm::NumericTable::doAllocate, 0.0);

        if (nClasses != 2)
        {
            daal_dm::BlockDescriptor<Fptype> bd;
            _workPoint->getBlockOfRows(0, nBetas*nBetaCols, daal_dm::writeOnly, bd);
            auto ptr = bd.getBlockPtr();

            for (size_t i = 0; i < bd.getNumberOfRows(); i += nBetaCols)
            {
                ptr[i] = Fptype(1e-3);
            }

            _workPoint->releaseBlockOfRows(bd);
        }
    }

private:
    SGDMinibatchSolverPtr _solver;

    LogLossPtr _logLoss;
    CrossEntropyLossPtr _crossEntropyLoss;

    daal_dm::NumericTablePtr _workPoint;
    daal_dm::NumericTablePtr _learningRateTable;
    daal_dm::NumericTablePtr _finalBetas;

    size_t _batchRowCount;
    size_t _nFeatures;
    size_t _nClasses;
    size_t _nBetas;
    size_t _nBetaCols;
    bool _isIntercept;
};
