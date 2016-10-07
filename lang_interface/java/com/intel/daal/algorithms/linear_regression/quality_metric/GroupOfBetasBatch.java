/* file: GroupOfBetasBatch.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUPOFBETASBATCH"></a>
 * @brief Computes the linear regression regression group of betas quality metrics in batch processing mode
 *
 * @par Enumerations
 *      - @ref GroupOfBetasMethod               Method for computing the group of betas quality metrics
 *      - @ref GroupOfBetasInputId              Identifiers of input objects for the group of betas quality metrics algorithm
 *      - @ref GroupOfBetasResultId             Result identifiers for the group of betas quality metrics
 *
 * @par References
 *      - GroupOfBetasInput class
 *      - GroupOfBetasParameter class
 *      - GroupOfBetasResult class
 */
public class GroupOfBetasBatch extends com.intel.daal.algorithms.quality_metric.QualityMetricBatch {
    public GroupOfBetasInput     input;
    public GroupOfBetasParameter parameter;
    public GroupOfBetasMethod    method;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the group of betas quality metrics algorithm in the batch processing mode
     * by copying input objects and parameters of another single beta quality metrics algorithm
     * @param context   Context to manage the group of betas quality metrics algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public GroupOfBetasBatch(DaalContext context, GroupOfBetasBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new GroupOfBetasInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new GroupOfBetasParameter(getContext(),
            cInitParameter(cObject, prec.getValue(), method.getValue(),
                other.parameter.getNBeta(), other.parameter.getNBetaReducedModel()));
    }

    /**
     * Constructs the group of betas quality metrics algorithm in the batch processing mode
     * @param context   Context to manage the group of betas quality metrics algorithm
     * @param cls       Data type to use in intermediate computations of the group of betas quality metrics,
     *                  Double.class or Float.class
     * @param method    Method for computing the group of betasquality metrics, @ref GroupOfBetasMethod
     * @param nBeta     Number of beta coefficients (p) of linear regression model used for prediction
     * @param nBetaReducedModel    Number of beta coefficients (p0) used for prediction with reduced linear
     *                             regression model where p - p0 of p beta coefficients are set to 0
     */
    public GroupOfBetasBatch(DaalContext context, Class<? extends Number> cls,
            GroupOfBetasMethod method, long nBeta, long nBetaReducedModel) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != GroupOfBetasMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue(), nBeta, nBetaReducedModel);

        input = new GroupOfBetasInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new GroupOfBetasParameter(getContext(),
            cInitParameter(cObject, prec.getValue(), method.getValue(), nBeta, nBetaReducedModel));
    }

    /**
     * Computes the group of betasquality metrics
     * @return Structure that contains results of the group of betasquality metrics algorithm
     */
    @Override
    public GroupOfBetasResult compute() {
        super.compute();
        GroupOfBetasResult result = new GroupOfBetasResult(getContext(), cGetResult(cObject, prec.getValue(),
                method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory for storing results of the group of betas quality metrics algorithm
     * @param result    Structure for storing results of the group of betas quality metrics algorithm
     */
    public void setResult(GroupOfBetasResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated single beta quality metrics algorithm with a copy of input objects
     * and parameters of this single beta quality metrics algorithm
     * @param context   Context to manage the group of betas quality metrics algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public GroupOfBetasBatch clone(DaalContext context) {
        return new GroupOfBetasBatch(context, this);
    }

    private native long cInit(int prec, int method, long nBeta, long nBetaReducedModel);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);

    private native long cInitParameter(long algAddr, int prec, int method, long nBeta, long nBetaReducedModel);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
