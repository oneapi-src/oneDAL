/* file: MultiClassConfusionMatrixBatch.java */
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

package com.intel.daal.algorithms.classifier.quality_metric.multi_class_confusion_matrix;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTI_CLASS_CONFUSION_MATRIX__MULTICLASSCONFUSIONMATRIXBATCH"></a>
 * @brief Computes the confusion matrix for a multi-class classifier in the batch processing mode.
 *
 * @par Enumerations
 *      - @ref MultiClassConfusionMatrixMethod   Method for computing the multi-class confusion matrix
 *      - @ref MultiClassConfusionMatrixInputId  Identifiers of input objects for the multi-class confusion matrix algorithm
 *      - @ref MultiClassConfusionMatrixResultId Result identifiers for the multi-class confusion matrix
 *      - @ref MultiClassMetricId                Identifiers of resulting metrics associated with the multi-class confusion matrix
 *
 * @par References
 *      - MultiClassConfusionMatrixInput class
 *      - MultiClassConfusionMatrixParameter class
 *      - MultiClassConfusionMatrixResult class
 */
public class MultiClassConfusionMatrixBatch extends com.intel.daal.algorithms.quality_metric.QualityMetricBatch {
    public MultiClassConfusionMatrixInput     input;
    public MultiClassConfusionMatrixParameter parameter;
    public MultiClassConfusionMatrixMethod    method;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the multi-class confusion matrix algorithm in the batch processing mode
     * by copying input objects and parameters of another multi-class confusion matrix algorithm
     * @param context   Context to manage the multi-class confusion matrix algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public MultiClassConfusionMatrixBatch(DaalContext context, MultiClassConfusionMatrixBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new MultiClassConfusionMatrixInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new MultiClassConfusionMatrixParameter(getContext(), cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the multi-class confusion matrix algorithm in the batch processing mode
     * @param context   Context to manage the multi-class confusion matrix algorithm
     * @param cls       Data type to use in intermediate computations of the multi-class confusion matrix,
     *                  Double.class or Float.class
     * @param method    Method for computing the multi-class confusion matrix, @ref MultiClassConfusionMatrixMethod
     */
    public MultiClassConfusionMatrixBatch(DaalContext context, Class<? extends Number> cls,
            MultiClassConfusionMatrixMethod method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != MultiClassConfusionMatrixMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new MultiClassConfusionMatrixInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new MultiClassConfusionMatrixParameter(getContext(), cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the multi-class confusion matrix
     * @return Structure that contains results of the multi-class confusion matrix algorithm
     */
    @Override
    public MultiClassConfusionMatrixResult compute() {
        super.compute();
        MultiClassConfusionMatrixResult result = new MultiClassConfusionMatrixResult(getContext(), cGetResult(cObject, prec.getValue(),
                method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory for storing results of the multi-class confusion matrix algorithm
     * @param result    Structure for storing results of the multi-class confusion matrix algorithm
     */
    public void setResult(MultiClassConfusionMatrixResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated multi-class confusion matrix algorithm with a copy of input objects
     * and parameters of this multi-class confusion matrix algorithm
     * @param context   Context to manage the multi-class confusion matrix algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public MultiClassConfusionMatrixBatch clone(DaalContext context) {
        return new MultiClassConfusionMatrixBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
