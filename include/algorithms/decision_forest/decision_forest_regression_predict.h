/* file: decision_forest_regression_predict.h */
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
//  Implementation of the interface for decision forest model-based prediction
//--
*/

#ifndef __DECISION_FOREST_REGRESSION_PREDICT_H__
#define __DECISION_FOREST_REGRESSION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/decision_forest/decision_forest_regression_predict_types.h"
#include "algorithms/regression/regression_predict.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace prediction
{
namespace interface1
{
/**
 * @defgroup decision_forest_regression_prediction_batch Batch
 * @ingroup decision_forest_regression_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__PREDICTION__BATCHCONTAINER"></a>
 *  \brief Class containing computation methods for decision forest model-based prediction
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for decision forest model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    ~BatchContainer();
    /**
     *  Computes the result of decision forest model-based prediction
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the decision forest model-based prediction
 * <!-- \n<a href="DAAL-REF-DECISIONFOREST-ALGORITHM">decision forest algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for decision forest model-based prediction
 *                          in the batch processing mode, double or float
 * \tparam method           Computation method in the batch processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for decision forest model-based prediction
 *
 * \par References
 *      - \ref decision_forest::regression::interface1::Model "decision_forest::regression::Model" class
 *      - \ref training::interface1::Batch "training::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public algorithms::regression::prediction::Batch
{
public:
    typedef algorithms::regression::prediction::Batch super;

    typedef algorithms::decision_forest::regression::prediction::Input  InputType;
    typedef typename super::ParameterType                               ParameterType;
    typedef algorithms::decision_forest::regression::prediction::Result ResultType;

    InputType     input;            /*!< %Input data structure */
    ParameterType parameter;        /*!< \ref algorithms::interface1::Parameter "Parameters" of prediction */

    /** Default constructor */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs a decision forest prediction algorithm by copying input objects and parameters
     * of another decision forest prediction algorithm
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : input(other.input), parameter(other.parameter)
    {
        initialize();
    }

    virtual InputType* getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns the structure that contains the result of decision forest model-based prediction
     * \return Structure that contains the result of the decision forest model-based prediction
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Returns a pointer to a newly allocated decision forest prediction algorithm
     * with a copy of the input objects for this decision forest prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:

    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = getResult()->template allocate<algorithmFPType>(_in, 0, 0);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

}
}
}
}
}
#endif
