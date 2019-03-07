/* file: PredictionBatch.java */
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
 * @defgroup kdtree_knn_classification_prediction_batch Batch
 * @ingroup kdtree_knn_classification_prediction
 * @{
 */
/**
 * @brief Contains classes for making prediction based on the K nearest neighbors models
 */
package com.intel.daal.algorithms.kdtree_knn_classification.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.kdtree_knn_classification.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Runs k nearest neighbors model based prediction algorithm
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">k nearest neighbors algorithm description and usage models</a> -->
 *
 * @par References
 *      - PredictionMethod class
 *      - PredictioInput class
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
     * Constructs k nearest neighbors prediction algorithm by copying input objects and parameters
     * of another k nearest neighbors prediction algorithm
     * @param context   Context to manage the k nearest neighbors prediction algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new PredictionInput(getContext(), cObject);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs k nearest neighbors prediction algorithm
     * @param context   Context to manage the k nearest neighbors prediction algorithm
     * @param cls       Data type to use in intermediate computations of the k nearest neighbors prediction algorithm,
     *                  Double.class or Float.class
     * @param method    K nearest neighbors prediction method, @ref PredictionMethod
     */
    public PredictionBatch(DaalContext context, Class<? extends Number> cls, PredictionMethod method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != PredictionMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new PredictionInput(getContext(), cObject);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the newly allocated k nearest neighbors prediction algorithm
     * with a copy of input objects and parameters of this k nearest neighbors prediction algorithm
     * @param context   Context to manage the k nearest neighbors prediction algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
