/* file: SmoothreluBackwardBatch.java */
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

package com.intel.daal.algorithms.neural_networks.layers.smoothrelu;

import com.intel.daal.algorithms.neural_networks.layers.Parameter;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SMOOTHRELU__SMOOTHRELUBACKWARDBATCH"></a>
 * \brief Class that computes the results of the backward smooth rectified linear unit (smoothrelu) layer in the batch processing mode
 * \n<a href="DAAL-REF-SMOOTHRELUBACKWARD">Backward smooth relu layer description and usage models</a>
 *
 * \par References
 *      - @ref SmoothreluMethod class
 *      - @ref SmoothreluLayerDataId class
 *      - @ref SmoothreluBackwardInput class
 *      - @ref SmoothreluBackwardResult class
 */
public class SmoothreluBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  SmoothreluBackwardInput input;    /*!< %Input data */
    public  SmoothreluMethod        method;   /*!< Computation method for the layer */
    private Precision     prec;     /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward smoothrelu layer by copying input objects of backward smoothrelu layer
     * @param context    Context to manage the backward smoothrelu layer
     * @param other      A backward smoothrelu layer to be used as the source to initialize the input objects of the backward smoothrelu layer
     */
    public SmoothreluBackwardBatch(DaalContext context, SmoothreluBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new SmoothreluBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward smoothrelu layer
     * @param context    Context to manage the backward smoothrelu layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref SmoothreluMethod
     */
    public SmoothreluBackwardBatch(DaalContext context, Class<? extends Number> cls, SmoothreluMethod method) {
        super(context);

        this.method = method;

        if (method != SmoothreluMethod.defaultDense) {
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
        input = new SmoothreluBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward smoothrelu layer
     * @param context    Context to manage the backward smoothrelu layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref SmoothreluMethod
     */
    SmoothreluBackwardBatch(DaalContext context, Class<? extends Number> cls, SmoothreluMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != SmoothreluMethod.defaultDense) {
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
        input = new SmoothreluBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward smoothrelu layer
     * @return  Backward smoothrelu layer result
     */
    @Override
    public SmoothreluBackwardResult compute() {
        super.compute();
        SmoothreluBackwardResult result = new SmoothreluBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward smoothrelu layer
     * @param result    Structure to store the result of the backward smoothrelu layer
     */
    public void setResult(SmoothreluBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public SmoothreluBackwardResult getLayerResult() {
        return new SmoothreluBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public SmoothreluBackwardInput getLayerInput() {
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
     * Returns the newly allocated backward smoothrelu layer
     * with a copy of input objects of this backward smoothrelu layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward smoothrelu layer
     */
    @Override
    public SmoothreluBackwardBatch clone(DaalContext context) {
        return new SmoothreluBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
