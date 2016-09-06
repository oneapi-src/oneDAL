/* file: neural_networks_prediction.h */
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
//  Implementation of the interface for neural network model-based prediction
//  in the batch processing mode
//--
*/

#ifndef __NEURAL_NETWORKS_PREDICTION_H__
#define __NEURAL_NETWORKS_PREDICTION_H__

#include "algorithms/algorithm.h"

#include "services/daal_defines.h"
#include "algorithms/neural_networks/neural_networks_types.h"
#include "algorithms/neural_networks/neural_networks_prediction_types.h"
#include "algorithms/neural_networks/neural_networks_prediction_model.h"
#include "algorithms/neural_networks/layers/layer.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for neural network model-based training and prediction
 */
namespace neural_networks
{
namespace prediction
{
namespace interface1
{
/**
 * @defgroup neural_networks_prediction_batch Batch
 * @ingroup neural_networks_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__BATCHCONTAINER"></a>
 * \brief Class containing methods to train neural network model using algorithmFPType precision arithmetic
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for neural network model-based prediction with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of neural network model-based prediction in the batch processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__BATCH"></a>
* \brief Provides methods for neural network model-based prediction in the batch processing mode
* \n<a href="DAAL-REF-NEURALNETWORK-ALGORITHM">Neural network algorithm description and usage models</a>
*
* \tparam algorithmFPType  Data type to use in intermediate computations for neural network model-based prediction, double or float
* \tparam method           Neural network prediction method, prediction::Method
*
* \par Enumerations
*      - prediction::Method  Computation methods
*
* \par References
*      - \ref interface1::Parameter "Parameter" class
*      - \ref neural_networks::prediction::interface1::Model "neural_networks::prediction::Model" class
*      - \ref prediction::interface1::Batch "prediction::Batch" class
*/
template<typename algorithmFPType = float, Method method = defaultDense>
class Batch : public daal::algorithms::Prediction
{
public:
    /** Default constructor */
    Batch()
    {
        initialize();
    };

    /**
     * Constructs neural network by copying input objects and parameters of another neural network
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other)
    {
        initialize();
        input.set(data,   other.input.get(data));
        input.set(model,  other.input.get(model));
        parameter = other.parameter;
    }

    virtual ~Batch() {}

    /**
    * Returns the structure that contains the results of the neural network algorithm
    * \return Structure that contains the results of the neural network algorithm
    */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Register user-allocated memory to store the results of the neural network algorithm
     * \return Structure to store the results of the neural network algorithm
     */
    void setResult(const services::SharedPtr<Result>& res)
    {
        DAAL_CHECK(res, ErrorNullResult)
        _result = res;
        _res = _result.get();
    }

    /**
     * Returns a pointer to the newly allocated neural network
     * with a copy of input objects and parameters of this neural network
     * \return Pointer to the newly allocated layer
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    Input input; /*!< %Input data structure */
    Parameter parameter; /*!< Prediction parameters */

protected:
    void initialize()
    {
        Prediction::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result = services::SharedPtr<Result>(new Result());
    }

    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        _result->allocate<algorithmFPType>(&input, &parameter, (int) method);
        _res = _result.get();
    }

private:
    services::SharedPtr<Result> _result;
};

/** @} */
} // namespace interface1
using interface1::Batch;
using interface1::BatchContainer;

} // namespace prediction
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
