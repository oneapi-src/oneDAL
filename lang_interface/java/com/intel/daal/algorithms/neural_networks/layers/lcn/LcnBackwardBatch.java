/* file: LcnBackwardBatch.java */
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

package com.intel.daal.algorithms.neural_networks.layers.lcn;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__LCNBACKWARDBATCH"></a>
 * \brief Class that computes the results of the local contrast normalization layer in the batch processing mode
 * \n<a href="DAAL-REF-LCNBACKWAR">Backward local contrast normalization layer description and usage models</a>
 *
 * \par References
 *      - @ref LcnMethod class
 *      - @ref LcnLayerDataId class
 *      - @ref LcnBackwardInput class
 *      - @ref LcnBackwardResult class
 */
public class LcnBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  LcnBackwardInput input;     /*!< %Input data */
    public  LcnMethod        method;    /*!< Computation method for the layer */
    public  LcnParameter     parameter; /*!< LcnParameters of the layer */
    private Precision     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward local contrast normalization layer by copying input objects of backward local contrast normalization layer
     * @param context    Context to manage the backward local contrast normalization layer
     * @param other      A backward local contrast normalization layer to be used as the source to initialize the input objects of the backward local contrast normalization layer
     */
    public LcnBackwardBatch(DaalContext context, LcnBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new LcnBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LcnParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward local contrast normalization layer
     * @param context    Context to manage the backward local contrast normalization layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LcnMethod
     */
    public LcnBackwardBatch(DaalContext context, Class<? extends Number> cls, LcnMethod method) {
        super(context);

        this.method = method;

        if (method != LcnMethod.defaultDense) {
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
        input = new LcnBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LcnParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }


    /**
     * Constructs the backward local contrast normalization layer
     * @param context    Context to manage the backward local contrast normalization layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LcnMethod
     */
    LcnBackwardBatch(DaalContext context, Class<? extends Number> cls, LcnMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != LcnMethod.defaultDense) {
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
        input = new LcnBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LcnParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward local contrast normalization layer
     * @return  Backward local contrast normalization layer result
     */
    @Override
    public LcnBackwardResult compute() {
        super.compute();
        LcnBackwardResult result = new LcnBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward local contrast normalization layer
     * @param result    Structure to store the result of the backward local contrast normalization layer
     */
    public void setResult(LcnBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public LcnBackwardResult getLayerResult() {
        return new LcnBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public LcnBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public LcnParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward local contrast normalization layer
     * with a copy of input objects of this backward local contrast normalization layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward local contrast normalization layer
     */
    @Override
    public LcnBackwardBatch clone(DaalContext context) {
        return new LcnBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
