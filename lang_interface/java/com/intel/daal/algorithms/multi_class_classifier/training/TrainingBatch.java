/* file: TrainingBatch.java */
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

/**
 * @defgroup multi_class_classifier_training_batch Batch
 * @ingroup multi_class_classifier_training
 * @{
 */
/**
 * @brief Contains classes for multi-class classifier model training
 */
package com.intel.daal.algorithms.multi_class_classifier.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.classifier.training.TrainingInput;
import com.intel.daal.algorithms.multi_class_classifier.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__TRAINING__TRAININGBATCH"></a>
 * @brief Class for multi-class classifier model training
 * <!-- \n<a href="DAAL-REF-MULTICLASSCLASSIFIER-ALGORITHM">Multi-class classifier algorithm description and usage models</a> -->
 *
 * @par References
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.multi_class_classifier.Model class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 */
public class TrainingBatch extends com.intel.daal.algorithms.classifier.training.TrainingBatch {
    public Parameter      parameter; /*!< Parameters of the algorithm */
    public TrainingMethod method;    /*!< %Training method for the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs multi-class classifier training algorithm by copying input objects and parameters
     * of another multi-class classifier training algorithm
     * @param context   Context to manage the multi-class classifier training algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new TrainingInput(getContext(), cObject, ComputeMode.batch);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    private void constructBatch(Class<? extends Number> cls, TrainingMethod method, long nClasses) {
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != TrainingMethod.oneAgainstOne) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue(), nClasses);

        input = new TrainingInput(getContext(), cObject, ComputeMode.batch);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs multi-class classifier training algorithm
     * @DAAL_DEPRECATED
     * @param context   Context to manage the multi-class classifier training algorithm
     * @param cls       Data type to use in intermediate computations of the multi-class classifier training,
     *                  Double.class or Float.class
     * @param method    Multi-class classifier training method, @ref TrainingMethod
     */
    @Deprecated
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        super(context);
        constructBatch(cls, method, 0);
    }

    /**
     * Constructs multi-class classifier training algorithm
     * @param context   Context to manage the multi-class classifier training algorithm
     * @param cls       Data type to use in intermediate computations of the multi-class classifier training,
     *                  Double.class or Float.class
     * @param method    Multi-class classifier training method, @ref TrainingMethod
     * @param nClasses  Number of classes
     */
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, TrainingMethod method, long nClasses) {
        super(context);
        constructBatch(cls, method, nClasses);
    }

    /**
     * Trains multi-class classifier model
     * @return Structure that contains the training results of the multi-class classifier algorithm
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Returns the newly allocated multi-class classifier training algorithm
     * with a copy of input objects and parameters of this multi-class classifier training algorithm
     * @param context   Context to manage the multi-class classifier training algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private native long cInit(int prec, int method, long nClasses);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);

    /**
     * Releases memory allocated for the native algorithm object
     */
    //public void dispose() {
    /*
    if(this.cObject != 0) {
        cDispose(this.cObject);
    }
    */
    //}
}
/** @} */
