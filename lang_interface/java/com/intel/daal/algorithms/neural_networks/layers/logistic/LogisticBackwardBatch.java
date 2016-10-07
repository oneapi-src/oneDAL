/* file: LogisticBackwardBatch.java */
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

package com.intel.daal.algorithms.neural_networks.layers.logistic;

import com.intel.daal.algorithms.neural_networks.layers.Parameter;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC__LOGISTICBACKWARDBATCH"></a>
 * \brief Class that computes the results of the logistic layer in the batch processing mode
 * \n<a href="DAAL-REF-LOGISTICBACKWARD">Backward logistic layer description and usage models</a>
 *
 * \par References
 *      - @ref LogisticMethod class
 *      - @ref LogisticLayerDataId class
 *      - @ref LogisticBackwardInput class
 *      - @ref LogisticBackwardResult class
 */
public class LogisticBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  LogisticBackwardInput input;    /*!< %Input data */
    public  LogisticMethod        method;   /*!< Computation method for the layer */
    private Precision     prec;     /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward logistic layer by copying input objects from another backward logistic layer
     * @param context    Context to manage the backward logistic layer
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of the backward logistic layer
     */
    public LogisticBackwardBatch(DaalContext context, LogisticBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new LogisticBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward logistic layer
     * @param context    Context to manage the backward logistic layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LogisticMethod
     */
    public LogisticBackwardBatch(DaalContext context, Class<? extends Number> cls, LogisticMethod method) {
        super(context);

        this.method = method;

        if (method != LogisticMethod.defaultDense) {
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
        input = new LogisticBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward logistic layer
     * @param context    Context to manage the backward logistic layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LogisticMethod
     */
    LogisticBackwardBatch(DaalContext context, Class<? extends Number> cls, LogisticMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != LogisticMethod.defaultDense) {
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
        input = new LogisticBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward logistic layer
     * @return  Backward logistic layer result
     */
    @Override
    public LogisticBackwardResult compute() {
        super.compute();
        LogisticBackwardResult result = new LogisticBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward logistic layer
     * @param result    Structure to store the result of the backward logistic layer
     */
    public void setResult(LogisticBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public LogisticBackwardResult getLayerResult() {
        return new LogisticBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public LogisticBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public Parameter getLayerParameter() {
        return null;
    }

    /**
     * Returns the newly allocated backward logistic layer
     * with a copy of input objects of this backward logistic layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward logistic layer
     */
    @Override
    public LogisticBackwardBatch clone(DaalContext context) {
        return new LogisticBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
