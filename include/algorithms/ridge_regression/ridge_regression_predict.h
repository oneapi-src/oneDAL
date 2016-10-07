/* file: ridge_regression_predict.h */
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
//  Implementation of the interface for ridge regression model-based prediction
//--
*/

#ifndef __RIDGE_REGRESSION_PREDICT_H__
#define __RIDGE_REGRESSION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/ridge_regression/ridge_regression_predict_types.h"

#include "algorithms/ridge_regression/ridge_regression_model.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace prediction
{

namespace interface1
{
/**
 * @defgroup ridge_regression_prediction_batch Batch
 * @ingroup ridge_regression_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__PREDICTION__BATCHCONTAINER"></a>
 *  \brief Class containing computation methods for ridge regression model-based prediction
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for ridge regression model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    ~BatchContainer();
    /**
     *  Computes the result of ridge regression model-based prediction
     */
    void compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the ridge regression model-based prediction
 * \n<a href="DAAL-REF-RIDGEREGRESSION-ALGORITHM">Ridge regression algorithm description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for ridge regression model-based prediction
 *                          in the batch processing mode, double or float
 * \tparam method           Computation method in the batch processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for ridge regression model-based prediction
 *
 * \par References
 *      - \ref ridge_regression::interface1::Parameter "ridge_regression::Parameter" class
 *      - \ref ridge_regression::interface1::Model "ridge_regression::Model" class
 *      - \ref ridge_regression::interface1::ModelNormEq "ridge_regression::ModelNormEq" class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 */
template<typename algorithmFPType = double, Method method = defaultDense>
class Batch : public daal::algorithms::Prediction
{
public:
    /** Default constructor */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs a ridge regression prediction algorithm by copying input objects and parameters
     * of another ridge regression prediction algorithm
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other)
    {
        initialize();
        input.set(data,  other.input.get(data));
        input.set(model, other.input.get(model));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store the result of ridge regression model-based prediction
     * \param[in] res    Structure to store the result of ridge regression model-based prediction
     */
    void setResult(const services::SharedPtr<Result>& res)
    {
        DAAL_CHECK(res, ErrorNullResult)
        _result = res;
        _res = _result.get();
    }

    /**
     * Returns the structure that contains the result of ridge regression model-based prediction
     * \return Structure that contains the result of the ridge regression model-based prediction
     */
    services::SharedPtr<Result> getResult() { return _result; }

/**
     * Returns a pointer to a newly allocated ridge regression prediction algorithm
     * with a copy of the input objects for this ridge regression prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    Input input; /*!< %Input data structure */
    Parameter parameter; /*!< Parameters of prediction */

protected:
    services::SharedPtr<Result> _result;

    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    void allocateResult() DAAL_C11_OVERRIDE
    {
        _result->allocate<algorithmFPType>(_in, 0, 0);
        _res = _result.get();
    }

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result = services::SharedPtr<Result>(new Result());
    }
};
/** @} */
} // namespace interface1

using interface1::BatchContainer;
using interface1::Batch;

} // namespace prediction
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
