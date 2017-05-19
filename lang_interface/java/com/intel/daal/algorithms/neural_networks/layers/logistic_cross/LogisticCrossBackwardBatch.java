/* file: LogisticCrossBackwardBatch.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

package com.intel.daal.algorithms.neural_networks.layers.logistic_cross;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC_CROSS__LOGISTICCROSSBACKWARDBATCH"></a>
 * \brief Class that computes the results of the backward logistic cross-entropy layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOGISTIC_CROSSBACKWARD-ALGORITHM">Backward logistic cross-entropy layer description and usage models</a> -->
 *
 * \par References
 *      - @ref LogisticCrossLayerDataId class
 */
public class LogisticCrossBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.loss.LossBackwardBatch {
    public  LogisticCrossBackwardInput input;     /*!< %Input data */
    public  LogisticCrossMethod        method;    /*!< Computation method for the layer */
    public  LogisticCrossParameter     parameter; /*!< LogisticCrossParameters of the layer */
    private Precision     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward logistic cross-entropy layer by copying input objects of backward logistic cross-entropy layer
     * @param context    Context to manage the backward logistic cross-entropy layer
     * @param other      A backward logistic cross-entropy layer to be used as the source to initialize the input objects of the backward logistic cross-entropy layer
     */
    public LogisticCrossBackwardBatch(DaalContext context, LogisticCrossBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new LogisticCrossBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LogisticCrossParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward logistic cross-entropy layer
     * @param context    Context to manage the backward logistic cross-entropy layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LogisticCrossMethod
     */
    public LogisticCrossBackwardBatch(DaalContext context, Class<? extends Number> cls, LogisticCrossMethod method) {
        super(context);

        this.method = method;

        if (method != LogisticCrossMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new LogisticCrossBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LogisticCrossParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward logistic layer
     * @param context    Context to manage the backward logistic layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LogisticMethod
     */
    LogisticCrossBackwardBatch(DaalContext context, Class<? extends Number> cls, LogisticCrossMethod method, long cObject) {
        super(context, cObject);

        this.method = method;

        if (method != LogisticCrossMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cObject;
        input = new LogisticCrossBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LogisticCrossParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward logistic cross-entropy layer
     * @return  Backward logistic cross-entropy layer result
     */
    @Override
    public LogisticCrossBackwardResult compute() {
        super.compute();
        LogisticCrossBackwardResult result = new LogisticCrossBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward logistic cross-entropy layer
     * @param result    Structure to store the result of the backward logistic cross-entropy layer
     */
    public void setResult(LogisticCrossBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public LogisticCrossBackwardResult getLayerResult() {
        return new LogisticCrossBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public LogisticCrossBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public LogisticCrossParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward logistic cross-entropy layer
     * with a copy of input objects of this backward logistic cross-entropy layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward logistic cross-entropy layer
     */
    @Override
    public LogisticCrossBackwardBatch clone(DaalContext context) {
        return new LogisticCrossBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
