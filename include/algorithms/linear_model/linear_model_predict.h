/* file: linear_model_predict.h */
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
//  Implementation of the interface for the regression model-based prediction
//--
*/

#ifndef __LINEAR_MODEL_PREDICT_H__
#define __LINEAR_MODEL_PREDICT_H__

#include "algorithms/linear_model/linear_model_predict_types.h"
#include "algorithms/regression/regression_predict.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{
namespace interface1
{
/**
 * @defgroup linear_model_prediction_batch Batch
 * @ingroup linear_model_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__PREDICTION__BATCHCONTAINER"></a>
 *  \brief Class containing computation methods for the regression model-based prediction
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for the regression model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    ~BatchContainer();
    /**
     *  Computes the result of the regression model-based prediction
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the regression model-based prediction
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the regression model-based prediction
 *                          in the batch processing mode, double or float
 * \tparam method           Computation method in the batch processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for the regression model-based prediction
 *
 * \par References
 *      - \ref linear_model::interface1::Model "linear_model::Model" class
 *      - \ref training::interface1::Batch "training::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public regression::prediction::Batch
{
public:
    typedef algorithms::linear_model::prediction::Input  InputType;
    typedef algorithms::linear_model::Parameter          ParameterType;
    typedef algorithms::linear_model::prediction::Result ResultType;

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns the structure that contains the result of the regression model-based prediction
     * \return Structure that contains the result of the regression model-based prediction
     */
    ResultPtr getResult() { return ResultType::cast(_result); }
};
/** @} */
}
using interface1::Batch;
using interface1::BatchContainer;
}
}
}
}
#endif
