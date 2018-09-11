/* file: ExplainedVarianceBatch.java */
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
 * @defgroup pca_quality_metric_explained_variance_batch Batch
 * @ingroup pca_quality_metric_explained_variance
 * @{
 */
package com.intel.daal.algorithms.pca.quality_metric;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINEDVARIANCEBATCH"></a>
 * @brief Computes the PCA explained variance quality metrics in batch processing mode
 *
 * @par Enumerations
 *      - @ref ExplainedVarianceMethod                     Method for computing the explained variance quality metrics
 *      - @ref ExplainedVarianceInputId                    Identifiers of input objects for the explained variance quality metrics algorithm
 *      - @ref ExplainedVarianceResultId                   Result identifiers for the explained variance quality metrics
 *
 * @par References
 *      - ExplainedVarianceInput class
 *      - ExplainedVarianceParameter class
 */
public class ExplainedVarianceBatch extends com.intel.daal.algorithms.quality_metric.QualityMetricBatch {
    public ExplainedVarianceInput     input;
    public ExplainedVarianceParameter parameter;
    public ExplainedVarianceMethod    method;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the explained variance quality metrics algorithm in the batch processing mode
     * by copying input objects and parameters of another explained variance quality metrics algorithm
     * @param context   Context to manage the explained variance quality metrics algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public ExplainedVarianceBatch(DaalContext context, ExplainedVarianceBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new ExplainedVarianceInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new ExplainedVarianceParameter(getContext(), cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the explained variance quality metrics algorithm in the batch processing mode
     * @param context   Context to manage the explained variance quality metrics algorithm
     * @param cls       Data type to use in intermediate computations of the explained variance quality metrics,
     *                  Double.class or Float.class
     * @param method    Method for computing the explained variance quality metrics, @ref ExplainedVarianceMethod
     */
    public ExplainedVarianceBatch(DaalContext context, Class<? extends Number> cls,
            ExplainedVarianceMethod method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != ExplainedVarianceMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new ExplainedVarianceInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new ExplainedVarianceParameter(getContext(), cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the explained variance quality metrics
     * @return Structure that contains results of the explained variance quality metrics algorithm
     */
    @Override
    public ExplainedVarianceResult compute() {
        super.compute();
        ExplainedVarianceResult result = new ExplainedVarianceResult(getContext(), cGetResult(cObject, prec.getValue(),
                method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory for storing results of the explained variance quality metrics algorithm
     * @param result    Structure for storing results of the explained variance quality metrics algorithm
     */
    public void setResult(ExplainedVarianceResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated explained variance quality metrics algorithm with a copy of input objects
     * and parameters of this explained variance quality metrics algorithm
     * @param context   Context to manage the explained variance quality metrics algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public ExplainedVarianceBatch clone(DaalContext context) {
        return new ExplainedVarianceBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
