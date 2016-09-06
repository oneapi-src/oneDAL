/* file: MaximumPooling3dBackwardBatch.java */
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

package com.intel.daal.algorithms.neural_networks.layers.maximum_pooling3d;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.neural_networks.layers.pooling3d.Pooling3dIndices;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING3D__MAXIMUMPOOLING3DBACKWARDBATCH"></a>
 * \brief Class that computes the results of the three-dimensional maximum pooling layer in the batch processing mode
 * \n<a href="DAAL-REF-MAXIMUMPOOLING3DBACKWARD-ALGORITHM">Backward three-dimensional maximum pooling layer description and usage models</a>
 *
 * \par References
 *      - @ref MaximumPooling3dMethod class
 *      - @ref MaximumPooling3dLayerDataId class
 *      - @ref MaximumPooling3dBackwardInput class
 *      - @ref MaximumPooling3dBackwardResult class
 */
public class MaximumPooling3dBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  MaximumPooling3dBackwardInput input;     /*!< %Input data */
    public  MaximumPooling3dMethod        method;    /*!< Computation method for the layer */
    public  MaximumPooling3dParameter     parameter; /*!< MaximumPooling3dParameters of the layer */
    private Precision     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward three-dimensional maximum pooling layer by copying input objects of
     * backward three-dimensional maximum pooling layer
     * @param context    Context to manage the backward three-dimensional maximum pooling layer
     * @param other      A backward three-dimensional maximum pooling layer to be used as the source
     *                   to initialize the input objects of the backward three-dimensional maximum pooling layer
     */
    public MaximumPooling3dBackwardBatch(DaalContext context, MaximumPooling3dBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new MaximumPooling3dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new MaximumPooling3dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward three-dimensional maximum pooling layer
     * @param context    Context to manage the backward three-dimensional maximum pooling layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref MaximumPooling3dMethod
     * @param nDim       Number of dimensions in input data
     */
    public MaximumPooling3dBackwardBatch(DaalContext context, Class<? extends Number> cls, MaximumPooling3dMethod method, long nDim) {
        super(context);

        this.method = method;

        if (method != MaximumPooling3dMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), nDim);
        input = new MaximumPooling3dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new MaximumPooling3dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    MaximumPooling3dBackwardBatch(DaalContext context, Class<? extends Number> cls, MaximumPooling3dMethod method, long cObject, long nDim) {
        super(context);

        this.method = method;

        if (method != MaximumPooling3dMethod.defaultDense) {
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
        input = new MaximumPooling3dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new MaximumPooling3dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        Pooling3dIndices sd = new Pooling3dIndices(nDim - 3, nDim - 2, nDim - 1);
        parameter.setIndices(sd);
    }

    /**
     * Computes the result of the backward three-dimensional maximum pooling layer
     * @return  Backward three-dimensional maximum pooling layer result
     */
    @Override
    public MaximumPooling3dBackwardResult compute() {
        super.compute();
        MaximumPooling3dBackwardResult result = new MaximumPooling3dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward three-dimensional maximum pooling layer
     * @param result    Structure to store the result of the backward three-dimensional maximum pooling layer
     */
    public void setResult(MaximumPooling3dBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public MaximumPooling3dBackwardResult getLayerResult() {
        return new MaximumPooling3dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public MaximumPooling3dBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public MaximumPooling3dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward three-dimensional maximum pooling layer
     * with a copy of input objects of this backward three-dimensional maximum pooling layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward three-dimensional maximum pooling layer
     */
    @Override
    public MaximumPooling3dBackwardBatch clone(DaalContext context) {
        return new MaximumPooling3dBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
