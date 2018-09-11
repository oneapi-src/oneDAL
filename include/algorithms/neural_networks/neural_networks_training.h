/* file: neural_networks_training.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of the interface for neural network model-based training
//  in the batch processing mode
//--
*/

#ifndef __NEURAL_NETWORKS_TRAINING_H__
#define __NEURAL_NETWORKS_TRAINING_H__

#include "algorithms/algorithm.h"

#include "services/daal_defines.h"
#include "algorithms/neural_networks/neural_networks_types.h"
#include "algorithms/neural_networks/neural_networks_training_types.h"
#include "algorithms/neural_networks/neural_networks_training_model.h"
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
namespace training
{
namespace interface1
{
/**
 * @defgroup neural_networks_training_batch Batch
 * @ingroup neural_networks_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__BATCHCONTAINER"></a>
 * \brief Class containing methods to train neural network model using algorithmFPType precision arithmetic
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for neural network model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of neural network model-based training in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    services::Status setupCompute() DAAL_C11_OVERRIDE;
    services::Status resetCompute() DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__BATCH"></a>
* \brief Provides methods for neural network model-based training in the batch processing mode
* <!-- \n<a href="DAAL-REF-NEURALNETWORK-ALGORITHM">Neural network algorithm description and usage models</a> -->
*
* \tparam algorithmFPType  Data type to use in intermediate computations for neural network model-based training, double or float
* \tparam method           Neural network training method, training::Method
*
* \par Enumerations
*      - training::Method  Computation methods
*
* \par References
*      - \ref neural_networks::training::interface1::Model "neural_networks::training::Model" class
*      - \ref prediction::interface1::Batch "prediction::Batch" class
*/
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public daal::algorithms::Training<batch>
{
public:
    typedef algorithms::neural_networks::training::Input     InputType;
    typedef algorithms::neural_networks::training::Parameter ParameterType;
    typedef algorithms::neural_networks::training::Result    ResultType;

    /** Default constructor */
    Batch(services::SharedPtr<optimization_solver::iterative_solver::Batch > optimizationSolver_) : parameter(optimizationSolver_)
    {
        initialize();
    };

    /**
     * Constructs neural network by copying input objects and parameters of another neural network
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : parameter(other.parameter), input(other.input)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
     * Initializes the neural network topology
     * \param[in] sampleSize Dimensionality of the batch for the input to the first layer
     * \param[in] topology Neural network topology
     *
     * \return Status of computations
     */
    services::Status initialize(const services::Collection<size_t> &sampleSize, const training::Topology &topology)
    {
        ResultPtr result = getResult();
        if (!result || !result->get(neural_networks::training::model))
        {
            return services::Status(services::ErrorNullModel);
        }
        _result->get(neural_networks::training::model)->initialize<algorithmFPType>(sampleSize, topology, parameter);
        return services::Status();
    }

    /**
    * Returns the structure that contains the results of the neural network algorithm
    * \return Structure that contains the results of the neural network algorithm
    */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Register user-allocated memory to store the results of the neural network algorithm
     * \param[in] res Structure to store the results of the neural network algorithm
     *
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr& res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res = _result.get();
        return services::Status();
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

    InputType input; /*!< %Input data structure */
    ParameterType parameter; /*!< %Training parameters */

protected:
    void initialize()
    {
        Training<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, (int) method);
        _res = _result.get();
        return s;
    }

private:
    ResultPtr _result;
};
/** @} */
} // namespace interface1
using interface1::Batch;
using interface1::BatchContainer;

} // namespace training
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
