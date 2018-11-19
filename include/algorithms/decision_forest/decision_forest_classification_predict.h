/* file: decision_forest_classification_predict.h */
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
//  Implementation of the interface for decision forest classification model-based prediction
//--
*/

#ifndef __DECISION_FOREST_CLASSIFICATION_PREDICT_H__
#define __DECISION_FOREST_CLASSIFICATION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_predict.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "algorithms/decision_forest/decision_forest_classification_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
/**
 * \brief Contains classes for prediction based on decision forest models
 */
namespace prediction
{
/**
 * @defgroup decision_forest_classification_prediction_batch Batch
 * @ingroup decision_forest_classification_prediction
 * @{
 */
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__PREDICTION__METHOD"></a>
 * Available methods for predictions based on the decision_forest model
 */
enum Method
{
    defaultDense = 0        /*!< Default method */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__PREDICTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the decision_forest algorithm.
 *        This class is associated with daal::algorithms::decision_forest::prediction::interface1::Batch class
 *        and supports method to compute decision_forest prediction
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the decision_forest, double or float
 * \tparam method           decision_forest computation method, \ref Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for decision_forest model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of decision_forest model-based prediction
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__PREDICTION__BATCH"></a>
 * \brief Predicts decision_forest classification results
 * <!-- \n<a href="DAAL-REF-decision_forest-ALGORITHM">decision_forest algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the decision_forest algortithm, double or float
 * \tparam method           decision_forest computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                                       decision_forest prediction methods
 *      - \ref classifier::prediction::NumericTableInputId  Identifiers of input Numeric Table objects
 *                                                          for the decision_forest prediction algorithm
 *      - \ref classifier::prediction::ModelInputId         Identifiers of input Model objects of the decision_forest prediction algorithm
 *      - \ref classifier::prediction::ResultId             Identifiers of decision_forest prediction results
 *
 * \par References
 *      - \ref interface1::Model "Model" class
 *      - \ref classifier::prediction::interface1::Input "classifier::prediction::Input" class
 *      - \ref classifier::prediction::interface1::Result "classifier::prediction::Result" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public classifier::prediction::Batch
{
public:
    typedef classifier::prediction::Batch super;

    typedef algorithms::decision_forest::classification::prediction::Input InputType;
    typedef typename super::ParameterType                                  ParameterType;
    typedef typename super::ResultType                                     ResultType;

    InputType input;                /*!< %Input objects of the algorithm */
    ParameterType parameter;        /*!< \ref interface1::Parameter "Parameters" of the algorithm */

    /**
     * Constructs Decision forest prediction algorithm
     * \param[in] nClasses  Number of classes
     */
    Batch(size_t nClasses): parameter(nClasses)
    {
        initialize();
    };

    /**
     * Constructs a Decision forest prediction algorithm by copying input objects and parameters
     * of another Decision forest prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : classifier::prediction::Batch(other),
        input(other.input), parameter(other.parameter)
    {
        initialize();
    }

    ~Batch() {}

    /**
     * Get input objects for the Decision forest prediction algorithm
     * \return %Input objects for the Decision forest prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns a pointer to the newly allocated Decision forest prediction algorithm with a copy of input objects
     * and parameters of this Decision forest prediction algorithm
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
        services::Status s = _result->allocate<algorithmFPType>(&input, 0, 0);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        _in = &input;
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _par = &parameter;
    }
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace daal::algorithms::decision_forest::classification::prediction
}
}
}
} // namespace daal
#endif
