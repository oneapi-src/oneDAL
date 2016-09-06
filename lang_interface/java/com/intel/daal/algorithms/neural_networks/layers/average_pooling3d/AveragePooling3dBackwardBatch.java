/* file: AveragePooling3dBackwardBatch.java */
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

package com.intel.daal.algorithms.neural_networks.layers.average_pooling3d;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.neural_networks.layers.pooling3d.Pooling3dIndices;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING3D__AVERAGEPOOLING3DBACKWARDBATCH"></a>
 * \brief Class that computes the results of the three-dimensional average pooling layer in the batch processing mode
 * \n<a href="DAAL-REF-AVERAGEPOOLING3DBACKWARD-ALGORITHM">Backward three-dimensional average pooling layer description and usage models</a>
 *
 * \par References
 *      - @ref AveragePooling3dMethod class
 *      - @ref AveragePooling3dLayerDataId class
 *      - @ref AveragePooling3dBackwardInput class
 *      - @ref AveragePooling3dBackwardResult class
 */
public class AveragePooling3dBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  AveragePooling3dBackwardInput input;     /*!< %Input data */
    public  AveragePooling3dMethod        method;    /*!< Computation method for the layer */
    public  AveragePooling3dParameter     parameter; /*!< AveragePooling3dParameters of the layer */
    private Precision                     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward three-dimensional average pooling layer by copying input objects of backward three-dimensional average pooling layer
     * @param context    Context to manage the backward three-dimensional average pooling layer
     * @param other      A backward three-dimensional average pooling layer to be used as the source to initialize the input objects of
     *                   the backward three-dimensional average pooling layer
     */
    public AveragePooling3dBackwardBatch(DaalContext context, AveragePooling3dBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new AveragePooling3dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new AveragePooling3dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward three-dimensional average pooling layer
     * @param context    Context to manage the backward three-dimensional average pooling layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref AveragePooling3dMethod
     * @param nDim       Number of dimensions in input data
     */
    public AveragePooling3dBackwardBatch(DaalContext context, Class<? extends Number> cls, AveragePooling3dMethod method, long nDim) {
        super(context);

        this.method = method;

        if (method != AveragePooling3dMethod.defaultDense) {
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
        input = new AveragePooling3dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new AveragePooling3dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    AveragePooling3dBackwardBatch(DaalContext context, Class<? extends Number> cls, AveragePooling3dMethod method, long cObject, long nDim) {
        super(context);

        this.method = method;

        if (method != AveragePooling3dMethod.defaultDense) {
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
        input = new AveragePooling3dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new AveragePooling3dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        Pooling3dIndices sd = new Pooling3dIndices(nDim - 3, nDim - 2, nDim - 1);
        parameter.setIndices(sd);
    }

    /**
     * Computes the result of the backward three-dimensional average pooling layer
     * @return  Backward three-dimensional average pooling layer result
     */
    @Override
    public AveragePooling3dBackwardResult compute() {
        super.compute();
        AveragePooling3dBackwardResult result = new AveragePooling3dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward three-dimensional average pooling layer
     * @param result    Structure to store the result of the backward three-dimensional average pooling layer
     */
    public void setResult(AveragePooling3dBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public AveragePooling3dBackwardResult getLayerResult() {
        return new AveragePooling3dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public AveragePooling3dBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public AveragePooling3dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward three-dimensional average pooling layer
     * with a copy of input objects of this backward three-dimensional average pooling layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward three-dimensional average pooling layer
     */
    @Override
    public AveragePooling3dBackwardBatch clone(DaalContext context) {
        return new AveragePooling3dBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
