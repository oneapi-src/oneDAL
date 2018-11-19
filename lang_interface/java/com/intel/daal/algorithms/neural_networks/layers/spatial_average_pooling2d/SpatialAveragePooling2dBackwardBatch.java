/* file: SpatialAveragePooling2dBackwardBatch.java */
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
 * @defgroup spatial_average_pooling2d_backward_batch Batch
 * @ingroup spatial_average_pooling2d_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.spatial_average_pooling2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.spatial_pooling2d.SpatialPooling2dIndices;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_AVERAGE_POOLING2D__SPATIALAVERAGEPOOLING2DBACKWARDBATCH"></a>
 * \brief Class that computes the results of the two-dimensional spatial average pooling layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-AVERAGEPOOLING2DBACKWARD-ALGORITHM">Backward two-dimensional spatial average pooling layer description and usage models</a> -->
 *
 * \par References
 *      - @ref SpatialAveragePooling2dLayerDataId class
 */
public class SpatialAveragePooling2dBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  SpatialAveragePooling2dBackwardInput input;     /*!< %Input data */
    public  SpatialAveragePooling2dMethod        method;    /*!< Computation method for the layer */
    public  SpatialAveragePooling2dParameter     parameter; /*!< SpatialAveragePooling2dParameters of the layer */
    private Precision     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward two-dimensional spatial average pooling layer by copying input objects
     * of backward two-dimensional spatial average pooling layer
     * @param context    Context to manage the backward two-dimensional spatial average pooling layer
     * @param other      A backward two-dimensional spatial average pooling layer to be used as the
     *                   source to initialize the input objects of the backward two-dimensional spatial average pooling layer
     */
    public SpatialAveragePooling2dBackwardBatch(DaalContext context, SpatialAveragePooling2dBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new SpatialAveragePooling2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialAveragePooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward two-dimensional spatial average pooling layer
     * @param context       Context to manage the backward two-dimensional spatial average pooling layer
     * @param cls           Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method        The layer computation method, @ref SpatialAveragePooling2dMethod
     * @param pyramidHeight The value of pyramid height
     * @param nDim          Number of dimensions in input data
     */
    public SpatialAveragePooling2dBackwardBatch(DaalContext context, Class<? extends Number> cls, SpatialAveragePooling2dMethod method, long pyramidHeight, long nDim) {
        super(context);

        this.method = method;

        if (method != SpatialAveragePooling2dMethod.defaultDense) {
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
        input = new SpatialAveragePooling2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialAveragePooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    SpatialAveragePooling2dBackwardBatch(DaalContext context, Class<? extends Number> cls, SpatialAveragePooling2dMethod method, long cObject, long pyramidHeight, long nDim) {
        super(context);

        this.method = method;

        if (method != SpatialAveragePooling2dMethod.defaultDense) {
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
        input = new SpatialAveragePooling2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialAveragePooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        SpatialPooling2dIndices sd = new SpatialPooling2dIndices(nDim - 2, nDim - 1);
        parameter.setIndices(sd);
    }

    /**
     * Computes the result of the backward two-dimensional spatial average pooling layer
     * @return  Backward two-dimensional spatial average pooling layer result
     */
    @Override
    public SpatialAveragePooling2dBackwardResult compute() {
        super.compute();
        SpatialAveragePooling2dBackwardResult result = new SpatialAveragePooling2dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward two-dimensional spatial average pooling layer
     * @param result    Structure to store the result of the backward two-dimensional spatial average pooling layer
     */
    public void setResult(SpatialAveragePooling2dBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public SpatialAveragePooling2dBackwardResult getLayerResult() {
        return new SpatialAveragePooling2dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public SpatialAveragePooling2dBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public SpatialAveragePooling2dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward two-dimensional spatial average pooling layer
     * with a copy of input objects of this backward two-dimensional spatial average pooling layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward two-dimensional spatial average pooling layer
     */
    @Override
    public SpatialAveragePooling2dBackwardBatch clone(DaalContext context) {
        return new SpatialAveragePooling2dBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method, long pyramidHeight, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
