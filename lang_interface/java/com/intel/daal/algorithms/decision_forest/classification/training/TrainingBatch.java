/* file: TrainingBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

/**
 * @defgroup decision_forest_classification_training_batch Batch
 * @ingroup decision_forest_classification_training
 * @{
 */
/**
 * @brief Contains classes for training decision forest classification algorithm
 */
package com.intel.daal.algorithms.decision_forest.classification.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.classifier.training.TrainingInput;
import com.intel.daal.algorithms.decision_forest.classification.training.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__TRAININGBATCH"></a>
 * @brief Trains a model of the decision forest classification algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-DECISION_FOREST__CLASSIFICATION-ALGORITHM">decision forest classification algorithm description and usage models</a> -->
 *
 * \par References
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.decision_forest.classification.Model class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 */
public class TrainingBatch extends com.intel.daal.algorithms.classifier.training.TrainingBatch {
    public Parameter      parameter; /*!< Parameters of the algorithm */
    public TrainingMethod method;   /*!< %Training method for the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the decision forest classification training algorithm by copying input objects and parameters
     * of another decision forest classification training algorithm
     * @param context   Context to manage decision forest classification training
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new TrainingInput(getContext(), cObject, ComputeMode.batch);
        parameter = new Parameter(getContext(),
                cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()));
    }

    /**
     * Constructs the decision forest classification training algorithm
     * @param context   Context to manage decision forest classification training
     * @param cls       Data type to use in intermediate computations for decision forest classification training,
     *                  Double.class or Float.class
     * @param method    decision forest classification training method, @ref TrainingMethod
     * @param nClasses  Number of classes
     */
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, TrainingMethod method, long nClasses) {
        super(context);

        this.method = method;

        if (this.method != TrainingMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue(), nClasses);
        input = new TrainingInput(getContext(), cObject, ComputeMode.batch);
        parameter = new Parameter(getContext(),
                cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()));
    }

    /**
     * Trains a model of the decision forest classification algorithm
     * @return Structure that contains results of the decision forest classification training algorithm
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cObject, ComputeMode.batch);
        return result;
    }

    /**
     * Returns the newly allocated decision forest classification training algorithm with a copy of input objects
     * and parameters of this decision forest classification training algorithm
     * @param context   Context to manage decision forest classification training
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private native long cInit(int prec, int method, long nClasses);

    private native long cInitParameter(long algAddr, int prec, int method, int cmode);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
