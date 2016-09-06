/* file: multiclass_confusion_matrix_types.h */
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
//  Declaration of data types for multi-class confusion matrix.
//--
*/

#ifndef __MULTICLASS_CONFUSION_MATRIX_TYPES_H__
#define __MULTICLASS_CONFUSION_MATRIX_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
/**
 * \brief Contains classes for checking the quality of the classifier algorithm
 */
namespace quality_metric
{
/**
 * @defgroup quality_metric_multiclass Quality Metrics for Multi-class Classification Algorithms
 * \copydoc daal::algorithms::classifier::quality_metric::multiclass_confusion_matrix
 * @ingroup quality_metric
 * @{
 */
/**
 * \brief Contains classes for computing the multi-class confusion matrix
 */
namespace multiclass_confusion_matrix
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTICLASS_CONFUSION_MATRIX__METHOD"></a>
 * Available methods for computing the confusion matrix
 */
enum Method
{
    defaultDense = 0    /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTICLASS_CONFUSION_MATRIX__INPUTID"></a>
 * Available identifiers of the input objects of the confusion matrix algorithm
 */
enum InputId
{
    predictedLabels   = 0,     /*!< Labels computed in the prediction stage of the classification  algorithm */
    groundTruthLabels = 1      /*!< Expected labels */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTICLASS_CONFUSION_MATRIX__RESULTID"></a>
 * Available identifiers of the results of the confusion matrix algorithm
 */
enum ResultId
{
    confusionMatrix     = 0,        /*!< Confusion matrix */
    multiClassMetrics   = 1,        /*!< Table that contains quality metrics (precision, recall, etc.) for binary classifiers */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTICLASS_CONFUSION_MATRIX__MULTICLASSMETRICSID"></a>
 * Available values stored in a numeric table corresponding to the ResultId::binaryMatrix index
 */
enum MultiClassMetricsId
{
    averageAccuracy = 0,            /*!< Average accuracy */
    errorRate       = 1,            /*!< Error rate */
    microPrecision  = 2,            /*!< Microprecision */
    microRecall     = 3,            /*!< Microrecall */
    microFscore     = 4,            /*!< Micro-F-score */
    macroPrecision  = 5,            /*!< Macroprecision */
    macroRecall     = 6,            /*!< Macrorecall */
    macroFscore     = 7             /*!< Macro-F-score */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTICLASS_CONFUSION_MATRIX__PARAMETER"></a>
 * \brief Parameters for the compute() method of the multi-class confusion matrix
 *
 * \snippet classifier/multiclass_confusion_matrix_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nClasses = 0, double beta = 1.0);
    virtual ~Parameter() {}

    size_t nClasses;        /*!< Number of classes */
    double beta;            /*!< Parameter of the F-score */

    void check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTICLASS_CONFUSION_MATRIX__INPUT"></a>
 * \brief Base class for the input objects of the confusion matrix algorithm in the training stage of the classification algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();

    virtual ~Input() {}

    /**
     * Returns the input object of the quality metric of the classification algorithm
     * \param[in] id   Identifier of the input object, \ref InputId
     * \return         Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets the input object of the quality metric of the classification algorithm
     * \param[in] id    Identifier of the input object, \ref InputId
     * \param[in] value Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr &value);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTICLASS_CONFUSION_MATRIX__RESULT"></a>
 * \brief Results obtained with the compute() method of the multi-class confusion matrix algorithm
 *        in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    Result();
    virtual ~Result() {}

    /**
     * Returns the quality metric of the classification algorithm
     * \param[in] id    Identifier of the result, \ref ResultId
     * \return          Quality metric of the classification algorithm
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the quality metric of the classification algorithm
     * \param[in] id    Identifier of the result, \ref ResultId
     * \param[in] value Pointer to the training result
     */
    void set(ResultId id, const data_management::NumericTablePtr &value);

    /**
     * Allocates memory for storing the computed quality metric
     * \param[in] input     Pointer to the input structure
     * \param[in] parameter Pointer to the parameter structure
     * \param[in] method    Computation method of the algorithm
     */
    template<typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks the correctness of the Result object
     * \param[in] input     Pointer to the input structure
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_CLASSIFIER_MULTICLASS_CONFUSION_MATRIX_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace daal::algorithms::classifier::quality_metric::multiclass_confusion_matrix
/** @} */
} // namespace daal::algorithms::classifier::quality_metric
} // namespace daal::algorithms::classifier
} // namespace daal::algorithms
} // namespace daal
#endif
