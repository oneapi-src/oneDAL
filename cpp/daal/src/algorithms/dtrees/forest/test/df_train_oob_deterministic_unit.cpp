/*******************************************************************************
* Copyright contributors to the oneDAL project
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

/// Out-of-bag results and variable importance of decision forest training must not depend on
/// the number of threads. The cases below train the same forest with one thread and with
/// several threads, twice, and require bitwise identical results.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "daal.h"

namespace daal::test
{
using namespace daal::data_management;
namespace df = daal::algorithms::decision_forest;

constexpr size_t nRows     = 4000;
constexpr size_t nFeatures = 8;
constexpr size_t nTrees    = 128;
constexpr size_t nClasses  = 3;
constexpr size_t seed      = 777;

struct Dataset
{
    std::vector<double> x;
    std::vector<double> yRegression;
    std::vector<double> yClassification;
};

/// Deterministic data. The regression response carries a full mantissa, so a reordered
/// floating-point sum of predictions differs in the last bits.
Dataset makeDataset()
{
    Dataset d;
    d.x.resize(nRows * nFeatures);
    d.yRegression.resize(nRows);
    d.yClassification.resize(nRows);
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    auto uniform   = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) * 0x1p-53;
    };
    for (size_t i = 0; i < nRows; ++i)
    {
        double sum = 0;
        for (size_t j = 0; j < nFeatures; ++j)
        {
            const double v         = uniform();
            d.x[i * nFeatures + j] = v;
            sum += static_cast<double>(j + 1) * v;
        }
        d.yRegression[i]     = sum + 0.1 * uniform();
        d.yClassification[i] = static_cast<double>((d.x[i * nFeatures] + d.x[i * nFeatures + 1] > 1.0) + (d.x[i * nFeatures + 2] > 0.5));
    }
    return d;
}

template <typename FP>
std::vector<FP> convert(const std::vector<double> & v)
{
    return std::vector<FP>(v.begin(), v.end());
}

std::vector<double> readTable(const NumericTablePtr & table)
{
    std::vector<double> values;
    if (!table) return values;
    const size_t nRowsTable = table->getNumberOfRows();
    BlockDescriptor<double> block;
    REQUIRE(table->getBlockOfRows(0, nRowsTable, readOnly, block).ok());
    values.assign(block.getBlockPtr(), block.getBlockPtr() + nRowsTable * table->getNumberOfColumns());
    REQUIRE(table->releaseBlockOfRows(block).ok());
    return values;
}

/// Result tables of one training run, flattened in a fixed order.
using Outputs = std::vector<std::vector<double> >;

template <typename FP, df::regression::training::Method method>
Outputs trainRegression(const Dataset & d, df::training::VariableImportanceMode varImportance)
{
    namespace training = df::regression::training;
    std::vector<FP> x  = convert<FP>(d.x);
    std::vector<FP> y  = convert<FP>(d.yRegression);
    training::Batch<FP, method> algorithm;
    algorithm.input.set(training::data, HomogenNumericTable<FP>::create(x.data(), nFeatures, nRows));
    algorithm.input.set(training::dependentVariable, HomogenNumericTable<FP>::create(y.data(), 1, nRows));
    algorithm.parameter().nTrees           = nTrees;
    algorithm.parameter().engine           = daal::algorithms::engines::mt2203::Batch<>::create(seed);
    algorithm.parameter().varImportance    = varImportance;
    algorithm.parameter().resultsToCompute = df::training::computeOutOfBagError | df::training::computeOutOfBagErrorPerObservation
                                             | df::training::computeOutOfBagErrorR2 | df::training::computeOutOfBagErrorPrediction;
    algorithm.compute();
    const training::ResultPtr res = algorithm.getResult();
    return { readTable(res->get(training::outOfBagError)), readTable(res->get(training::outOfBagErrorPerObservation)),
             readTable(res->get(training::outOfBagErrorR2)), readTable(res->get(training::outOfBagErrorPrediction)),
             readTable(res->get(training::variableImportance)) };
}

template <typename FP, df::classification::training::Method method>
Outputs trainClassification(const Dataset & d, df::training::VariableImportanceMode varImportance,
                            DAAL_UINT64 resultsToCompute = df::training::computeOutOfBagError | df::training::computeOutOfBagErrorPerObservation
                                                           | df::training::computeOutOfBagErrorAccuracy
                                                           | df::training::computeOutOfBagErrorDecisionFunction)
{
    namespace training = df::classification::training;
    std::vector<FP> x  = convert<FP>(d.x);
    std::vector<FP> y  = convert<FP>(d.yClassification);
    training::Batch<FP, method> algorithm(nClasses);
    algorithm.input.set(daal::algorithms::classifier::training::data, HomogenNumericTable<FP>::create(x.data(), nFeatures, nRows));
    algorithm.input.set(daal::algorithms::classifier::training::labels, HomogenNumericTable<FP>::create(y.data(), 1, nRows));
    algorithm.parameter().nTrees           = nTrees;
    algorithm.parameter().engine           = daal::algorithms::engines::mt2203::Batch<>::create(seed);
    algorithm.parameter().varImportance    = varImportance;
    algorithm.parameter().resultsToCompute = resultsToCompute;
    algorithm.compute();
    const training::ResultPtr res = algorithm.getResult();
    return { readTable(res->get(training::outOfBagError)), readTable(res->get(training::outOfBagErrorPerObservation)),
             readTable(res->get(training::outOfBagErrorAccuracy)), readTable(res->get(training::outOfBagErrorDecisionFunction)),
             readTable(res->get(training::variableImportance)) };
}

class ThreadCountGuard
{
public:
    explicit ThreadCountGuard(size_t count) : _saved(services::Environment::getInstance()->getNumberOfThreads())
    {
        services::Environment::getInstance()->setNumberOfThreads(count);
    }
    ~ThreadCountGuard() { services::Environment::getInstance()->setNumberOfThreads(_saved); }

private:
    size_t _saved;
};

size_t manyThreads()
{
    return std::max<size_t>(services::Environment::getInstance()->getNumberOfThreads(), 4);
}

void requireBitwiseEqual(const Outputs & expected, const Outputs & actual)
{
    REQUIRE(expected.size() == actual.size());
    for (size_t k = 0; k < expected.size(); ++k)
    {
        CAPTURE(k);
        REQUIRE(expected[k].size() == actual[k].size());
        REQUIRE(std::memcmp(expected[k].data(), actual[k].data(), expected[k].size() * sizeof(double)) == 0);
    }
}

template <typename Train>
void checkThreadCountIndependence(const Train & train)
{
    const Dataset d = makeDataset();
    Outputs single;
    {
        ThreadCountGuard guard(1);
        single = train(d);
    }
    REQUIRE(single[0].size() == 1);
    REQUIRE(single[0][0] > 0.0);
    REQUIRE(single[1].size() == nRows);
    REQUIRE(single[4].size() == nFeatures);
    ThreadCountGuard guard(manyThreads());
    requireBitwiseEqual(single, train(d));
    requireBitwiseEqual(single, train(d));
}

TEST("regression OOB results and variable importance do not depend on the thread count", "[decision_forest][unit]")
{
    namespace training = df::regression::training;
    SECTION("double, defaultDense, MDI")
    {
        checkThreadCountIndependence([](const Dataset & d) { return trainRegression<double, training::defaultDense>(d, df::training::MDI); });
    }
    SECTION("double, hist, MDA_Scaled")
    {
        checkThreadCountIndependence([](const Dataset & d) { return trainRegression<double, training::hist>(d, df::training::MDA_Scaled); });
    }
    SECTION("float, defaultDense, MDA_Raw")
    {
        checkThreadCountIndependence([](const Dataset & d) { return trainRegression<float, training::defaultDense>(d, df::training::MDA_Raw); });
    }
    SECTION("float, hist, MDI")
    {
        checkThreadCountIndependence([](const Dataset & d) { return trainRegression<float, training::hist>(d, df::training::MDI); });
    }
}

TEST("classification OOB results and variable importance do not depend on the thread count", "[decision_forest][unit]")
{
    namespace training = df::classification::training;
    SECTION("double, defaultDense, MDI")
    {
        checkThreadCountIndependence([](const Dataset & d) { return trainClassification<double, training::defaultDense>(d, df::training::MDI); });
    }
    SECTION("double, hist, MDA_Scaled")
    {
        checkThreadCountIndependence([](const Dataset & d) { return trainClassification<double, training::hist>(d, df::training::MDA_Scaled); });
    }
    SECTION("float, defaultDense, MDA_Raw")
    {
        checkThreadCountIndependence([](const Dataset & d) { return trainClassification<float, training::defaultDense>(d, df::training::MDA_Raw); });
    }
}

TEST("regression OOB prediction of a single tree matches the model prediction", "[decision_forest][unit]")
{
    namespace training   = df::regression::training;
    namespace prediction = df::regression::prediction;
    const Dataset d      = makeDataset();
    std::vector<double> x(d.x);
    std::vector<double> y(d.yRegression);
    const NumericTablePtr xTable = HomogenNumericTable<double>::create(x.data(), nFeatures, nRows);
    training::Batch<double, training::hist> algorithm;
    algorithm.input.set(training::data, xTable);
    algorithm.input.set(training::dependentVariable, HomogenNumericTable<double>::create(y.data(), 1, nRows));
    algorithm.parameter().nTrees           = 1;
    algorithm.parameter().engine           = daal::algorithms::engines::mt2203::Batch<>::create(seed);
    algorithm.parameter().resultsToCompute = df::training::computeOutOfBagErrorPerObservation | df::training::computeOutOfBagErrorPrediction;
    algorithm.compute();
    const training::ResultPtr res = algorithm.getResult();

    prediction::Batch<double> predictor;
    predictor.input.set(prediction::data, xTable);
    predictor.input.set(prediction::model, res->get(training::model));
    predictor.compute();

    const std::vector<double> errPerObs = readTable(res->get(training::outOfBagErrorPerObservation));
    const std::vector<double> oobPred   = readTable(res->get(training::outOfBagErrorPrediction));
    const std::vector<double> modelPred = readTable(predictor.getResult()->get(prediction::prediction));
    REQUIRE(errPerObs.size() == nRows);
    REQUIRE(oobPred.size() == nRows);
    REQUIRE(modelPred.size() == nRows);
    size_t nOOB = 0;
    for (size_t i = 0; i < nRows; ++i)
    {
        CAPTURE(i);
        if (errPerObs[i] < 0)
        {
            REQUIRE(oobPred[i] == 0.0);
            continue;
        }
        ++nOOB;
        REQUIRE(oobPred[i] == modelPred[i]);
        REQUIRE(errPerObs[i] == (modelPred[i] - y[i]) * (modelPred[i] - y[i]));
    }
    // a bootstrap sample of nRows rows leaves about 37% of them out
    REQUIRE(nOOB > nRows / 4);
    REQUIRE(nOOB < nRows / 2);
}

TEST("classification OOB accuracy alone matches the accuracy computed with the other OOB results", "[decision_forest][unit]")
{
    namespace training = df::classification::training;
    const Dataset d    = makeDataset();
    const Outputs all  = trainClassification<double, training::defaultDense>(d, df::training::none);
    const Outputs only = trainClassification<double, training::defaultDense>(d, df::training::none, df::training::computeOutOfBagErrorAccuracy);
    REQUIRE(all[2].size() == 1);
    REQUIRE(only[2].size() == 1);
    REQUIRE(only[2][0] == all[2][0]);
}

} // namespace daal::test
