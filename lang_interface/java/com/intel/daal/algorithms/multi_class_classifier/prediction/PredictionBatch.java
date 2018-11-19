/* file: PredictionBatch.java */
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
 * @defgroup multi_class_classifier_prediction_batch Batch
 * @ingroup multi_class_classifier_prediction
 * @{
 */
/**
 * @brief Contains classes for making prediction based on the Multi-class classifier models
 */
package com.intel.daal.algorithms.multi_class_classifier.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.multi_class_classifier.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Runs multi-class classifier model based prediction algorithm
 * <!-- \n<a href="DAAL-REF-MULTICLASSCLASSIFIER-ALGORITHM">Multi-class classifier algorithm description and usage models</a> -->
 *
 * @par References
 *      - com.intel.daal.algorithms.classifier.prediction.NumericTableInputId class
 *      - com.intel.daal.algorithms.classifier.prediction.ModelInputId class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResultId class
 *      - com.intel.daal.algorithms.classifier.prediction.PredictionResult class
 */
public class PredictionBatch extends com.intel.daal.algorithms.classifier.prediction.PredictionBatch {
    public PredictionInput  input;     /*!< %Input data */
    public Parameter        parameter; /*!< Parameters of the algorithm */
    public PredictionMethod method;    /*!< %Prediction method for the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs multi-class classifier prediction algorithm by copying input objects and parameters
     * of another multi-class classifier prediction algorithm
     * @param context   Context to manage the multi-class classifier prediction algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new PredictionInput(getContext(), cObject, ComputeMode.batch);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    private void constructBatch(Class<? extends Number> cls, PredictionMethod method, long nClasses) {
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != PredictionMethod.multiClassClassifierWu && this.method != PredictionMethod.voteBased) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue(), nClasses);

        input = new PredictionInput(getContext(), cObject, ComputeMode.batch);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs multi-class classifier prediction algorithm
     * @DAAL_DEPRECATED
     * @param context   Context to manage the multi-class classifier prediction algorithm
     * @param cls       Data type to use in intermediate computations of the multi-class classifier prediction algorithm,
     *                  Double.class or Float.class
     * @param method    Multi-class classifier prediction method, @ref PredictionMethod
     */
    @Deprecated
    public PredictionBatch(DaalContext context, Class<? extends Number> cls, PredictionMethod method) {
        super(context);
        constructBatch(cls, method, 0);
    }

    /**
     * Constructs multi-class classifier prediction algorithm
     * @param context   Context to manage the multi-class classifier prediction algorithm
     * @param cls       Data type to use in intermediate computations of the multi-class classifier prediction algorithm,
     *                  Double.class or Float.class
     * @param method    Multi-class classifier prediction method, @ref PredictionMethod
     * @param nClasses  Number of classes
     */
    public PredictionBatch(DaalContext context, Class<? extends Number> cls, PredictionMethod method, long nClasses) {
        super(context);
        constructBatch(cls, method, nClasses);
    }

    /**
     * Returns the newly allocated multi-class classifier prediction algorithm
     * with a copy of input objects and parameters of this multi-class classifier prediction algorithm
     * @param context   Context to manage the multi-class classifier prediction algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    private native long cInit(int prec, int method, long nClasses);

    private native long cInitParameter(long algAddr, int prec, int method);

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
