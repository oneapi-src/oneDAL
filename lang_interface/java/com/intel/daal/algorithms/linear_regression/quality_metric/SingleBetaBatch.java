/* file: SingleBetaBatch.java */
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

package com.intel.daal.algorithms.linear_regression.quality_metric;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLEBETABATCH"></a>
 * @brief Computes the linear regression regression single beta quality metrics in batch processing mode
 *
 * @par Enumerations
 *      - @ref SingleBetaMethod                     Method for computing the single beta quality metrics
 *      - @ref SingleBetaDataInputId                Identifiers of input objects for the single beta quality metrics algorithm
 *      - @ref SingleBetaModelInputId               Identifiers of input objects for the single beta quality metrics algorithm
 *      - @ref SingleBetaResultId                   Result identifiers for the single beta quality metrics
 *      - @ref SingleBetaResultDataCollectionId     Result identifiers for the single beta quality metrics
 *
 * @par References
 *      - SingleBetaInput class
 *      - SingleBetaParameter class
 *      - SingleBetaResult class
 */
public class SingleBetaBatch extends com.intel.daal.algorithms.quality_metric.QualityMetricBatch {
    public SingleBetaInput     input;
    public SingleBetaParameter parameter;
    public SingleBetaMethod    method;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the single beta quality metrics algorithm in the batch processing mode
     * by copying input objects and parameters of another single beta quality metrics algorithm
     * @param context   Context to manage the single beta quality metrics algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public SingleBetaBatch(DaalContext context, SingleBetaBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new SingleBetaInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SingleBetaParameter(getContext(), cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the single beta quality metrics algorithm in the batch processing mode
     * @param context   Context to manage the single beta quality metrics algorithm
     * @param cls       Data type to use in intermediate computations of the single beta quality metrics,
     *                  Double.class or Float.class
     * @param method    Method for computing the single beta quality metrics, @ref SingleBetaMethod
     */
    public SingleBetaBatch(DaalContext context, Class<? extends Number> cls,
            SingleBetaMethod method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != SingleBetaMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new SingleBetaInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SingleBetaParameter(getContext(), cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the single beta quality metrics
     * @return Structure that contains results of the single beta quality metrics algorithm
     */
    @Override
    public SingleBetaResult compute() {
        super.compute();
        SingleBetaResult result = new SingleBetaResult(getContext(), cGetResult(cObject, prec.getValue(),
                method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory for storing results of the single beta quality metrics algorithm
     * @param result    Structure for storing results of the single beta quality metrics algorithm
     */
    public void setResult(SingleBetaResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated single beta quality metrics algorithm with a copy of input objects
     * and parameters of this single beta quality metrics algorithm
     * @param context   Context to manage the single beta quality metrics algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public SingleBetaBatch clone(DaalContext context) {
        return new SingleBetaBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
