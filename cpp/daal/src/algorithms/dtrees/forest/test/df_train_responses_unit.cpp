/*
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
 */

#include "oneapi/dal/test/engine/common.hpp"
#include "algorithms/decision_forest/decision_forest_classification_training_batch.h"
#include "algorithms/decision_forest/decision_forest_regression_training_batch.h"
#include "data_management/data/homogen_numeric_table.h"

#include <atomic>
#include <cmath>
#include <random>

namespace daal::algorithms::decision_forest::internal
{
using namespace daal::data_management;

// float64 table that counts the reads converting it to float32.
class FloatReadCountingTable : public HomogenNumericTable<double>
{
public:
    FloatReadCountingTable(const services::SharedPtr<double> & data, size_t nRows, services::Status & st)
        : HomogenNumericTable<double>(DictionaryIface::equal, data, 1, nRows, st)
    {}

    using HomogenNumericTable<double>::getBlockOfRows;
    services::Status getBlockOfRows(size_t idx, size_t n, ReadWriteMode rw, BlockDescriptor<float> & block) override
    {
        ++floatReads;
        return HomogenNumericTable<double>::getBlockOfRows(idx, n, rw, block);
    }

    std::atomic<size_t> floatReads { 0 };
};

constexpr size_t nRows     = 2000;
constexpr size_t nFeatures = 4;
constexpr size_t nClasses  = 3;
constexpr size_t nTrees    = 20;

struct Dataset
{
    NumericTablePtr x;
    NumericTablePtr yFloat;
    services::SharedPtr<FloatReadCountingTable> yDouble;
};

Dataset makeDataset(bool classification)
{
    services::SharedPtr<float> x(new float[nRows * nFeatures]);
    services::SharedPtr<float> yFloat(new float[nRows]);
    services::SharedPtr<double> yDouble(new double[nRows]);

    std::mt19937 rng(7777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < nRows; ++i)
    {
        float * row = x.get() + i * nFeatures;
        for (size_t j = 0; j < nFeatures; ++j) row[j] = dist(rng);
        yFloat.get()[i]  = classification ? float((row[0] + row[1] > 0.0f) + (row[2] > 0.5f)) : row[0] * row[0] + row[1] - 0.5f * row[2];
        yDouble.get()[i] = yFloat.get()[i];
    }

    services::Status st;
    Dataset ds;
    ds.x       = HomogenNumericTable<float>::create(x, nFeatures, nRows, &st);
    ds.yFloat  = HomogenNumericTable<float>::create(yFloat, 1, nRows, &st);
    ds.yDouble = services::SharedPtr<FloatReadCountingTable>(new FloatReadCountingTable(yDouble, nRows, st));
    REQUIRE(st.ok());
    return ds;
}

double readScalar(const NumericTablePtr & table)
{
    BlockDescriptor<double> block;
    table->getBlockOfRows(0, 1, readOnly, block);
    const double value = block.getBlockPtr()[0];
    table->releaseBlockOfRows(block);
    return value;
}

template <typename Parameter>
void setParameters(Parameter & par)
{
    par.nTrees                      = nTrees;
    par.observationsPerTreeFraction = 0.3;
    par.resultsToCompute            = training::computeOutOfBagError;
}

double trainRegression(const NumericTablePtr & x, const NumericTablePtr & y)
{
    namespace dfr = regression::training;
    dfr::Batch<float> alg;
    alg.input.set(dfr::data, x);
    alg.input.set(dfr::dependentVariable, y);
    setParameters(alg.parameter());
    REQUIRE(alg.compute().ok());
    return readScalar(alg.getResult()->get(dfr::outOfBagError));
}

double trainClassification(const NumericTablePtr & x, const NumericTablePtr & y)
{
    namespace dfc = classification::training;
    dfc::Batch<float> alg(nClasses);
    alg.input.set(classifier::training::data, x);
    alg.input.set(classifier::training::labels, y);
    setParameters(alg.parameter());
    REQUIRE(alg.compute().ok());
    return readScalar(alg.getResult()->get(dfc::outOfBagError));
}

// Every tree and every OOB row reads the responses as float32; a float64 table is converted only once.
template <typename Train>
void checkConvertedOnce(bool classification, Train train)
{
    const Dataset ds       = makeDataset(classification);
    const double refOob    = train(ds.x, ds.yFloat);
    const double doubleOob = train(ds.x, ds.yDouble);
    REQUIRE(ds.yDouble->floatReads == 1);
    // OOB partial sums are reduced in a nondeterministic order.
    REQUIRE(std::abs(refOob - doubleOob) <= 1e-5 * (1.0 + std::abs(refOob)));
}

TEST("DF regression converts float64 responses once per fit", "[unit]")
{
    checkConvertedOnce(false, trainRegression);
}

TEST("DF classification converts float64 labels once per fit", "[unit]")
{
    checkConvertedOnce(true, trainClassification);
}
} // namespace daal::algorithms::decision_forest::internal
