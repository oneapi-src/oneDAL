/* file: SpatialStochasticPooling2dForwardBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 * @defgroup spatial_stochastic_pooling2d_forward_batch Batch
 * @ingroup spatial_stochastic_pooling2d_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.spatial_stochastic_pooling2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.spatial_pooling2d.SpatialPooling2dIndices;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_STOCHASTIC_POOLING2D__SPATIALSTOCHASTICPOOLING2DFORWARDBATCH"></a>
 * @brief Class that computes the results of the forward two-dimensional spatial stochastic pooling layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-STOCHASTICPOOLING2DFORWARD-ALGORITHM">Forward two-dimensional spatial stochastic pooling layer description and usage models</a> -->
 *
 * @par References
 *      - @ref SpatialStochasticPooling2dLayerDataId class
 */
public class SpatialStochasticPooling2dForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  SpatialStochasticPooling2dForwardInput input;     /*!< %Input data */
    public  SpatialStochasticPooling2dMethod       method;    /*!< Computation method for the layer */
    public  SpatialStochasticPooling2dParameter    parameter; /*!< SpatialStochasticPooling2dParameters of the layer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward two-dimensional spatial stochastic pooling layer by copying input objects of
     * another forward two-dimensional spatial stochastic pooling layer
     * @param context    Context to manage the forward two-dimensional spatial stochastic pooling layer
     * @param other      A forward two-dimensional spatial stochastic pooling layer to be used as the source to
     *                   initialize the input objects of the forward two-dimensional spatial stochastic pooling layer
     */
    public SpatialStochasticPooling2dForwardBatch(DaalContext context, SpatialStochasticPooling2dForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new SpatialStochasticPooling2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialStochasticPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward two-dimensional spatial stochastic pooling layer
     * @param context       Context to manage the layer
     * @param cls           Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method        The layer computation method, @ref SpatialStochasticPooling2dMethod
     * @param pyramidHeight The value of pyramid height
     * @param nDim          Number of dimensions in input data
     */
    public SpatialStochasticPooling2dForwardBatch(DaalContext context, Class<? extends Number> cls, SpatialStochasticPooling2dMethod method, long pyramidHeight, long nDim) {
        super(context);

        this.method = method;

        if (method != SpatialStochasticPooling2dMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), pyramidHeight, nDim);
        input = new SpatialStochasticPooling2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialStochasticPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    SpatialStochasticPooling2dForwardBatch(DaalContext context, Class<? extends Number> cls, SpatialStochasticPooling2dMethod method, long cObject, long pyramidHeight, long nDim) {
        super(context);

        this.method = method;

        if (method != SpatialStochasticPooling2dMethod.defaultDense) {
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
        input = new SpatialStochasticPooling2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialStochasticPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        SpatialPooling2dIndices sd = new SpatialPooling2dIndices(nDim - 2, nDim - 1);
        parameter.setIndices(sd);
    }

    /**
     * Computes the result of the forward two-dimensional spatial stochastic pooling layer
     * @return  Forward two-dimensional spatial stochastic pooling layer result
     */
    @Override
    public SpatialStochasticPooling2dForwardResult compute() {
        super.compute();
        SpatialStochasticPooling2dForwardResult result = new SpatialStochasticPooling2dForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward two-dimensional spatial stochastic pooling layer
     * @param result    Structure to store the result of the forward two-dimensional spatial stochastic pooling layer
     */
    public void setResult(SpatialStochasticPooling2dForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public SpatialStochasticPooling2dForwardResult getLayerResult() {
        return new SpatialStochasticPooling2dForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public SpatialStochasticPooling2dForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public SpatialStochasticPooling2dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward two-dimensional spatial stochastic pooling layer
     * with a copy of input objects of this forward two-dimensional spatial stochastic pooling layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward two-dimensional spatial stochastic pooling layer
     */
    @Override
    public SpatialStochasticPooling2dForwardBatch clone(DaalContext context) {
        return new SpatialStochasticPooling2dForwardBatch(context, this);
    }

    private native long cInit(int prec, int method, long pyramidHeight, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
