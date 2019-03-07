/* file: BinaryConfusionMatrixBatch.java */
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
 * @defgroup quality_metric_binary Quality Metrics for Binary Classification Algorithms
 * @brief Contains classes for computing the binary confusion matrix
 * @ingroup quality_metric
 * @{
 */
/**
 * @defgroup quality_metric_binary_batch Batch
 * @ingroup quality_metric_binary
 * @{
 */
package com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__BINARY_CONFUSION_MATRIX__BINARYCONFUSIONMATRIXBATCH"></a>
 * @brief Computes the confusion matrix for a binary classifier in the batch processing mode.
 *
 * @par Enumerations
 *      - @ref BinaryConfusionMatrixMethod   Method for computing the binary confusion matrix
 *      - @ref BinaryConfusionMatrixInputId  Identifiers of input objects for the binary confusion matrix algorithm
 *      - @ref BinaryConfusionMatrixResultId Result identifiers for the binary confusion matrix
 *      - @ref BinaryMetricId                Identifiers of resulting metrics associated with the binary confusion matrix
 *
 * @par References
 *      - BinaryConfusionMatrixInput class
 *      - BinaryConfusionMatrixParameter class
 */

public class BinaryConfusionMatrixBatch extends com.intel.daal.algorithms.quality_metric.QualityMetricBatch {
    public BinaryConfusionMatrixInput     input;
    public BinaryConfusionMatrixParameter parameter;
    public BinaryConfusionMatrixMethod    method;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the binary confusion matrix algorithm in the batch processing mode
     * by copying input objects and parameters of another confusion matrix algorithm
     * @param context   Context to manage the binary confusion matrix algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public BinaryConfusionMatrixBatch(DaalContext context, BinaryConfusionMatrixBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new BinaryConfusionMatrixInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new BinaryConfusionMatrixParameter(getContext(), cInitParameter(cObject, prec.getValue(), this.method.getValue()));
    }

    /**
     * Constructs the binary confusion matrix algorithm in the batch processing mode
     * @param context   Context to manage the binary confusion matrix algorithm
     * @param cls       Data type to use in intermediate computations of the binary confusion matrix,
     *                  Double.class or Float.class
     * @param method    Method for computing the binary confusion matrix, @ref BinaryConfusionMatrixMethod
     */
    public BinaryConfusionMatrixBatch(DaalContext context, Class<? extends Number> cls,
            BinaryConfusionMatrixMethod method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != BinaryConfusionMatrixMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new BinaryConfusionMatrixInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new BinaryConfusionMatrixParameter(getContext(), cInitParameter(cObject, prec.getValue(), this.method.getValue()));
    }

    /**
     * Computes the binary confusion matrix
     * @return Structure that contains results of the binary confusion matrix algorithm
     */
    @Override
    public BinaryConfusionMatrixResult compute() {
        super.compute();
        BinaryConfusionMatrixResult result = new BinaryConfusionMatrixResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory for storing results of the binary confusion matrix algorithm
     * @param result    Structure for storing results of the binary confusion matrix algorithm
     */
    public void setResult(BinaryConfusionMatrixResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated binary confusion matrix algorithm with a copy of input objects
     * and parameters of this binary confusion matrix algorithm
     * @param context   Context to manage the binary confusion matrix algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public BinaryConfusionMatrixBatch clone(DaalContext context) {
        return new BinaryConfusionMatrixBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */
