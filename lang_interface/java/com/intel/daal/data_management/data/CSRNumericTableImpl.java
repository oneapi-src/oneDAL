/* file: CSRNumericTableImpl.java */
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

package com.intel.daal.data_management.data;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__CSRNUMERICTABLEIMPL"></a>
 * @brief Numeric table that provides methods to access data that is stored
 *        in the Compressed Sparse Row(CSR) data layout
 */
public class CSRNumericTableImpl extends NumericTableImpl {
    private Class<?> type;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs sparse CSR numeric table from the array of doubles
     *
     * @param context       Context to manage created CSR numeric table
     * @param data          Array of values in the CSR layout
     * @param colIndices    Array of column indices in the CSR layout. The values of indices are determined by the index base
     * @param rowOffsets    Array of row indices in the CSR layout. The size of the array is nVectors+1. The first element is 0/1
     *                          in zero-/one-based indexing. The last element is ptr_size+0/1 in zero-/one-based indexing
     * @param nFeatures     Number of columns in the corresponding dense table
     * @param nVectors      Number of rows in the corresponding dense table
     * @param indexing      %Indexing scheme used to access data in the CSR layout
     *  Note: Present version of Intel(R) Data Analytics Acceleration Library supports 1-based indexing only
     */
    public CSRNumericTableImpl(DaalContext context, double[] data, long[] colIndices, long[] rowOffsets, long nFeatures,
            long nVectors, CSRNumericTable.Indexing indexing) {
        super(context);
        initialize(data, colIndices, rowOffsets, nFeatures, nVectors);
    }

    /**
     * Constructs sparse CSR numeric table from the array of doubles with 1-based indexing
     *
     * @param context       Context to manage created CSR numeric table
     * @param data          Array of values in the CSR layout
     * @param colIndices    Array of column indices in the CSR layout. The values of indices are determined by the index base
     * @param rowOffsets    Array of row indices in the CSR layout. The size of the array is nVectors+1. The first element is 0/1
     *                          in zero-/one-based indexing. The last element is ptr_size+0/1 in zero-/one-based indexing
     * @param nFeatures     Number of columns in the corresponding dense table
     * @param nVectors      Number of rows in the corresponding dense table
     */
    public CSRNumericTableImpl(DaalContext context, double[] data, long[] colIndices, long[] rowOffsets, long nFeatures,
            long nVectors) {
        super(context);
        initialize(data, colIndices, rowOffsets, nFeatures, nVectors);
    }

    /**
     * Constructs sparse CSR numeric table from array of floats
     *
     * @param context       Context to manage created CSR numeric table
     * @param data          Array of values in the CSR layout
     * @param colIndices    Array of column indices in the CSR layout. The values of indices are determined by the index base
     * @param rowOffsets    Array of row indices in the CSR layout. The size of the array is nVectors+1. The first element is 0/1
     *                          in zero-/one-based indexing. The last element is ptr_size+0/1 in zero-/one-based indexing
     * @param nFeatures     Number of columns in the corresponding dense table
     * @param nVectors      Number of rows in the corresponding dense table
     * @param indexing      %Indexing scheme used to access data in the CSR layout
     *  Note: Present version of Intel(R) Data Analytics Acceleration Library supports 1-based indexing only
     */
    public CSRNumericTableImpl(DaalContext context, float[] data, long[] colIndices, long[] rowOffsets, long nFeatures,
            long nVectors, CSRNumericTable.Indexing indexing) {
        super(context);
        initialize(data, colIndices, rowOffsets, nFeatures, nVectors);
    }

    /**
     * Constructs sparse CSR numeric table from the array of float with 1-based indexing
     *
     * @param context       Context to manage created CSR numeric table
     * @param data          Array of values in the CSR layout
     * @param colIndices    Array of column indices in the CSR layout. The values of indices are determined by the index base
     * @param rowOffsets    Array of row indices in the CSR layout. The size of the array is nVectors+1. The first element is 0/1
     *                          in zero-/one-based indexing. The last element is ptr_size+0/1 in zero-/one-based indexing
     * @param nFeatures     Number of columns in the corresponding dense table
     * @param nVectors      Number of rows in the corresponding dense table
     */
    public CSRNumericTableImpl(DaalContext context, float[] data, long[] colIndices, long[] rowOffsets, long nFeatures,
            long nVectors) {
        super(context);
        initialize(data, colIndices, rowOffsets, nFeatures, nVectors);
    }

    /**
     * Constructs sparse CSR numeric table from the array of integers
     *
     * @param context       Context to manage created CSR numeric table
     * @param data          Array of values in the CSR layout
     * @param colIndices    Array of column indices in the CSR layout. The values of indices are determined by the index base
     * @param rowOffsets    Array of row indices in the CSR layout. The size of the array is nVectors+1. The first element is 0/1
     *                          in zero-/one-based indexing. The last element is ptr_size+0/1 in zero-/one-based indexing
     * @param nFeatures     Number of columns in the corresponding dense table
     * @param nVectors      Number of rows in the corresponding dense table
     * @param indexing      %Indexing scheme used to access data in the CSR layout
     *  Note: Present version of Intel(R) Data Analytics Acceleration Library supports 1-based indexing only
     */
    public CSRNumericTableImpl(DaalContext context, int[] data, long[] colIndices, long[] rowOffsets, long nFeatures,
            long nVectors, CSRNumericTable.Indexing indexing) {
        super(context);
        initialize(data, colIndices, rowOffsets, nFeatures, nVectors);
    }

    /**
     * Constructs sparse CSR numeric table from the array of integers with 1-based indexing
     *
     * @param context       Context to manage created CSR numeric table
     * @param data          Array of values in the CSR layout
     * @param colIndices    Array of column indices in the CSR layout. The values of indices are determined by the index base
     * @param rowOffsets    Array of row indices in the CSR layout. The size of the array is nVectors+1. The first element is 0/1
     *                          in zero-/one-based indexing. The last element is ptr_size+0/1 in zero-/one-based indexing
     * @param nFeatures     Number of columns in the corresponding dense table
     * @param nVectors      Number of rows in the corresponding dense table
     */
    public CSRNumericTableImpl(DaalContext context, int[] data, long[] colIndices, long[] rowOffsets, long nFeatures,
            long nVectors) {
        super(context);
        initialize(data, colIndices, rowOffsets, nFeatures, nVectors);
    }

    /**
     * Constructs sparse CSR numeric table from the array of longs
     *
     * @param context       Context to manage created CSR numeric table
     * @param data          Array of values in the CSR layout
     * @param colIndices    Array of column indices in the CSR layout. The values of indices are determined by the index base
     * @param rowOffsets    Array of row indices in the CSR layout. The size of the array is nVectors+1. The first element is 0/1
     *                          in zero-/one-based indexing. The last element is ptr_size+0/1 in zero-/one-based indexing
     * @param nFeatures     Number of columns in the corresponding dense table
     * @param nVectors      Number of rows in the corresponding dense table
     * @param indexing      %Indexing scheme used to access data in the CSR layout
     *  Note: Present version of Intel(R) Data Analytics Acceleration Library supports 1-based indexing only
     */
    public CSRNumericTableImpl(DaalContext context, long[] data, long[] colIndices, long[] rowOffsets, long nFeatures,
            long nVectors, CSRNumericTable.Indexing indexing) {
        super(context);
        initialize(data, colIndices, rowOffsets, nFeatures, nVectors);
    }

    /**
     * Constructs sparse CSR numeric table from the array of longs with 1-based indexing
     *
     * @param context       Context to manage created CSR numeric table
     * @param data          Array of values in the CSR layout
     * @param colIndices    Array of column indices in the CSR layout. The values of indices are determined by the index base
     * @param rowOffsets    Array of row indices in the CSR layout. The size of the array is nVectors+1. The first element is 0/1
     *                          in zero-/one-based indexing. The last element is ptr_size+0/1 in zero-/one-based indexing
     * @param nFeatures     Number of columns in the corresponding dense table
     * @param nVectors      Number of rows in the corresponding dense table
     */
    public CSRNumericTableImpl(DaalContext context, long[] data, long[] colIndices, long[] rowOffsets, long nFeatures,
            long nVectors) {
        super(context);
        initialize(data, colIndices, rowOffsets, nFeatures, nVectors);
    }

    /**
    * Constructs homogeneous numeric table from C++ homogeneous numeric
    *        table
    * @param context   Context to manage the ALS algorithm
    * @param cTable    Pointer to C++ numeric table
    */
    public CSRNumericTableImpl(DaalContext context, long cTable) {
        super(context);
        dataAllocatedInJava = false;
        cObject = cTable;
        int indexType = getIndexType(this.cObject);
        if (indexType == DataFeatureUtils.IndexNumType.DAAL_FLOAT32.getType()) {
            this.type = Float.class;
        } else if (indexType == DataFeatureUtils.IndexNumType.DAAL_FLOAT64.getType()) {
            this.type = Double.class;
        } else if (indexType == DataFeatureUtils.IndexNumType.DAAL_INT64_S.getType()
                || indexType == DataFeatureUtils.IndexNumType.DAAL_INT64_U.getType()) {
            this.type = Long.class;
        } else if (indexType == DataFeatureUtils.IndexNumType.DAAL_INT32_S.getType()
                || indexType == DataFeatureUtils.IndexNumType.DAAL_INT32_U.getType()) {
            this.type = Integer.class;
        }
    }

    public long getDataSize() {
        if (dataAllocatedInJava) {
            return this.colIndices.length;
        } else {
            int dataSize = cGetDataSize(cObject);
            return dataSize;
        }
    }

    /**
     * Gets data as an array of longs
     * @return Table data as an array of longs
     */
    public long[] getRowOffsetsArray() {
        if (dataAllocatedInJava) {
            return this.rowOffsets;
        } else {
            deserializeCObject();
            int nRows = cGetNumberOfRows(cObject);
            ByteBuffer byteBuf = ByteBuffer.allocateDirect((nRows + 1) * 8 /* sizeof(long) */);
            byteBuf.order(ByteOrder.LITTLE_ENDIAN);
            byteBuf = getRowOffsetsBuffer(this.cObject, byteBuf);
            LongBuffer longBuffer = byteBuf.asLongBuffer();
            long[] buffer;
            buffer = new long[longBuffer.capacity()];
            longBuffer.get(buffer);
            return buffer;
        }
    }

    /**
    * Gets data as an array of longs
    * @return Table data as an array of longs
    */
    public long[] getColIndicesArray() {
        if (dataAllocatedInJava) {
            return this.colIndices;
        } else {
            deserializeCObject();
            int dataSize = cGetDataSize(cObject);
            ByteBuffer byteBuf = ByteBuffer.allocateDirect((dataSize) * 8 /* sizeof(long) */);
            byteBuf.order(ByteOrder.LITTLE_ENDIAN);
            byteBuf = getColIndicesBuffer(this.cObject, byteBuf);
            LongBuffer longBuffer = byteBuf.asLongBuffer();
            long[] buffer;
            buffer = new long[longBuffer.capacity()];
            longBuffer.get(buffer);
            return buffer;
        }
    }

    /**
     * Gets data as an array of doubles
     * @return Table data as an array of double
     */
    public double[] getDoubleArray() {
        if (dataAllocatedInJava) {
            return (double[]) jData;
        } else {
            deserializeCObject();
            ByteBuffer byteBuffer = getDoubleBuffer(this.cObject);
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
            DoubleBuffer doubleBuffer = byteBuffer.asDoubleBuffer();

            double[] buffer;
            buffer = new double[doubleBuffer.capacity()];
            doubleBuffer.get(buffer);

            return buffer;
        }
    }

    /**
     * Gets data as an array of floats
     * @return Table data as an array of floats
     */
    public float[] getFloatArray() {
        if (dataAllocatedInJava) {
            return (float[]) jData;
        } else {
            deserializeCObject();
            ByteBuffer byteBuffer = getFloatBuffer(this.cObject);
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
            FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

            float[] buffer;
            buffer = new float[floatBuffer.capacity()];
            floatBuffer.get(buffer);

            return buffer;
        }
    }

    /**
     * Gets data as an array of longs
     * @return Table data as an array of longs
     */
    public long[] getLongArray() {
        if (dataAllocatedInJava) {
            return (long[]) jData;
        } else {
            deserializeCObject();
            ByteBuffer byteBuffer = getLongBuffer(this.cObject);
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
            LongBuffer longBuffer = byteBuffer.asLongBuffer();

            long[] buffer;
            buffer = new long[longBuffer.capacity()];
            longBuffer.get(buffer);

            return buffer;
        }
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int iVectorIndex = (int) vectorIndex;
        int iVectorNum = (int) vectorNum;
        int sparseBlockSize = (int) (rowOffsets[iVectorIndex + iVectorNum] - rowOffsets[iVectorIndex]);

        double[] sparseDataPart = new double[sparseBlockSize];
        DoubleBuffer sparseDataPartBuffer = DoubleBuffer.wrap(sparseDataPart);
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, double.class);
        vectorUpCast.upCast(sparseBlockSize, (int) (rowOffsets[iVectorIndex] - 1), jData, sparseDataPartBuffer);

        int bufferSize = (int) (vectorNum * nColumns);
        double[] denseBlock = new double[bufferSize];
        for (int i = 0; i < bufferSize; i++) {
            denseBlock[i] = 0.0;
        }

        int sparseDataIndex = 0;
        int startRowOffset = (int) rowOffsets[iVectorIndex];
        for (int i = 0; i < (int) vectorNum; i++) {
            int sparseRowSize = (int) (rowOffsets[iVectorIndex + i + 1] - rowOffsets[iVectorIndex + i]);

            for (int k = 0; k < sparseRowSize; k++, sparseDataIndex++) {
                int j = (int) colIndices[startRowOffset + sparseDataIndex - 1] - 1;
                denseBlock[i * nColumns + j] = sparseDataPart[sparseDataIndex];
            }
        }

        buf.position(0);
        buf.put(denseBlock);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int iVectorIndex = (int) vectorIndex;
        int iVectorNum = (int) vectorNum;
        int sparseBlockSize = (int) (rowOffsets[iVectorIndex + iVectorNum] - rowOffsets[iVectorIndex]);

        float[] sparseDataPart = new float[sparseBlockSize];
        FloatBuffer sparseDataPartBuffer = FloatBuffer.wrap(sparseDataPart);
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, float.class);
        vectorUpCast.upCast(sparseBlockSize, (int) (rowOffsets[iVectorIndex] - 1), jData, sparseDataPartBuffer);

        int bufferSize = (int) (vectorNum * nColumns);
        float[] denseBlock = new float[bufferSize];
        for (int i = 0; i < bufferSize; i++) {
            denseBlock[i] = 0.0f;
        }

        int sparseDataIndex = 0;
        int startRowOffset = (int) rowOffsets[iVectorIndex];
        for (int i = 0; i < (int) vectorNum; i++) {
            int sparseRowSize = (int) (rowOffsets[iVectorIndex + i + 1] - rowOffsets[iVectorIndex + i]);

            for (int k = 0; k < sparseRowSize; k++, sparseDataIndex++) {
                int j = (int) colIndices[startRowOffset + sparseDataIndex - 1] - 1;
                denseBlock[i * nColumns + j] = sparseDataPart[sparseDataIndex];
            }
        }

        buf.position(0);
        buf.put(denseBlock);
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        int nColumns = (int) (getNumberOfColumns());
        int iVectorIndex = (int) vectorIndex;
        int iVectorNum = (int) vectorNum;
        int sparseBlockSize = (int) (rowOffsets[iVectorIndex + iVectorNum] - rowOffsets[iVectorIndex]);

        int[] sparseDataPart = new int[sparseBlockSize];
        IntBuffer sparseDataPartBuffer = IntBuffer.wrap(sparseDataPart);
        DataFeature df = dict.getFeature(0);
        DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, int.class);
        vectorUpCast.upCast(sparseBlockSize, (int) (rowOffsets[iVectorIndex] - 1), jData, sparseDataPartBuffer);

        int bufferSize = (int) (vectorNum * nColumns);
        int[] denseBlock = new int[bufferSize];
        for (int i = 0; i < bufferSize; i++) {
            denseBlock[i] = 0;
        }

        int sparseDataIndex = 0;
        int startRowOffset = (int) rowOffsets[iVectorIndex];
        for (int i = 0; i < (int) vectorNum; i++) {
            int sparseRowSize = (int) (rowOffsets[iVectorIndex + i + 1] - rowOffsets[iVectorIndex + i]);

            for (int k = 0; k < sparseRowSize; k++, sparseDataIndex++) {
                int j = (int) colIndices[startRowOffset + sparseDataIndex - 1] - 1;
                denseBlock[i * nColumns + j] = sparseDataPart[sparseDataIndex];
            }
        }

        buf.position(0);
        buf.put(denseBlock);
        return buf;
    }

    /**
     * Reads block of rows from the table and returns it to java.nio.DoubleBuffer
     *
     * @param numType       Type of data requested
     * @param vectorIndex   Index of the first row to include into the block
     * @param vectorNum     Number of rows in the block
     * @param buf           Buffer to store non-zero values
     * @param colIndicesBuf Buffer to store indices of the columns containing values from buf
     * @param rowOffsetsBuf Buffer to store row offsets of the values from buf
     *
     * @return Number of rows obtained from the table
     */
    protected long getSparseBlock(DataFeatureUtils.InternalNumType numType, long vectorIndex, long vectorNum,
            ByteBuffer buf, LongBuffer colIndicesBuf, LongBuffer rowOffsetsBuf) {
        int iVectorIndex = (int) vectorIndex;
        int iVectorNum = (int) vectorNum;
        int sparseBlockSize = (int) (rowOffsets[iVectorIndex + iVectorNum] - rowOffsets[iVectorIndex]);

        buf.position(0);
        DataFeature df = dict.getFeature(0);
        if (numType == DataFeatureUtils.InternalNumType.DAAL_DOUBLE) {
            DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, DataFeatureUtils.getClassByType(numType));
            vectorUpCast.upCast(sparseBlockSize, (int) (rowOffsets[iVectorIndex] - 1), jData, buf.asDoubleBuffer());
        } else if (numType == DataFeatureUtils.InternalNumType.DAAL_SINGLE) {
            DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, DataFeatureUtils.getClassByType(numType));
            vectorUpCast.upCast(sparseBlockSize, (int) (rowOffsets[iVectorIndex] - 1), jData, buf.asFloatBuffer());
        } else if (numType == DataFeatureUtils.InternalNumType.DAAL_INT32) {
            DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, DataFeatureUtils.getClassByType(numType));
            vectorUpCast.upCast(sparseBlockSize, (int) (rowOffsets[iVectorIndex] - 1), jData, buf.asIntBuffer());
        }

        int startRowOffset = (int) rowOffsets[iVectorIndex] - 1;
        long[] colIndicesPart = new long[sparseBlockSize];
        for (int i = 0; i < sparseBlockSize; i++) {
            colIndicesPart[i] = colIndices[startRowOffset + i];
        }
        colIndicesBuf.position(0);
        colIndicesBuf.put(colIndicesPart);

        long[] rowOffsetsPart = new long[iVectorNum + 1];
        for (int i = 0; i < iVectorNum + 1; i++) {
            rowOffsetsPart[i] = rowOffsets[iVectorIndex + i] - startRowOffset;
        }
        rowOffsetsBuf.position(0);
        rowOffsetsBuf.put(rowOffsetsPart);
        return vectorNum;
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,IntBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        int iVectorIndex = (int) vectorIndex;

        DataFeature df = dict.getFeature((int) featureIndex);

        double[] value = new double[1];
        DoubleBuffer valueBuf = DoubleBuffer.wrap(value);

        for (int i = 0; i < (int) vectorNum; i++) {
            buf.put(i, 0.0);
            int sparseRowSize = (int) (rowOffsets[iVectorIndex + i + 1] - rowOffsets[iVectorIndex + i]);

            for (int k = 0; k < sparseRowSize; k++) {
                int index = (int) rowOffsets[iVectorIndex + i + 1] + k - 1;
                if (colIndices[index] - 1 == featureIndex) {
                    DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, double.class);
                    vectorUpCast.upCast(1, index, jData, valueBuf);
                    buf.put(i, value[0]);
                }
            }
        }
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        int iVectorIndex = (int) vectorIndex;

        DataFeature df = dict.getFeature((int) featureIndex);

        float[] value = new float[1];
        FloatBuffer valueBuf = FloatBuffer.wrap(value);

        for (int i = 0; i < (int) vectorNum; i++) {
            buf.put(i, 0.0f);
            int sparseRowSize = (int) (rowOffsets[iVectorIndex + i + 1] - rowOffsets[iVectorIndex + i]);

            for (int k = 0; k < sparseRowSize; k++) {
                int index = (int) rowOffsets[iVectorIndex + i + 1] + k - 1;
                if (colIndices[index] - 1 == featureIndex) {
                    DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, float.class);
                    vectorUpCast.upCast(1, index, jData, valueBuf);
                    buf.put(i, value[0]);
                }
            }
        }
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        int iVectorIndex = (int) vectorIndex;

        DataFeature df = dict.getFeature((int) featureIndex);

        int[] value = new int[1];
        IntBuffer valueBuf = IntBuffer.wrap(value);

        for (int i = 0; i < (int) vectorNum; i++) {
            buf.put(i, 0);
            int sparseRowSize = (int) (rowOffsets[iVectorIndex + i + 1] - rowOffsets[iVectorIndex + i]);

            for (int k = 0; k < sparseRowSize; k++) {
                int index = (int) rowOffsets[iVectorIndex + i + 1] + k - 1;
                if (colIndices[index] - 1 == featureIndex) {
                    DataFeatureUtils.VectorUpCastIface vectorUpCast = DataFeatureUtils.VectorUpCast.getCast(df.type, int.class);
                    vectorUpCast.upCast(1, index, jData, valueBuf);
                    buf.put(i, value[0]);
                }
            }
        }
        return buf;
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
    }

    protected long getSparseBlockSize(long vectorIndex, long vectorNum) {
        return rowOffsets[(int) (vectorIndex + vectorNum)] - rowOffsets[(int) vectorIndex];
    }

    protected long getDoubleSparseBlock(long vectorIndex, long vectorNum, ByteBuffer buf, ByteBuffer colIndicesBuf,
            ByteBuffer rowOffsetsBuf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        colIndicesBuf.order(ByteOrder.LITTLE_ENDIAN);
        rowOffsetsBuf.order(ByteOrder.LITTLE_ENDIAN);
        return getSparseBlock(DataFeatureUtils.InternalNumType.DAAL_DOUBLE, vectorIndex, vectorNum, buf,
                colIndicesBuf.asLongBuffer(), rowOffsetsBuf.asLongBuffer());
    }

    protected long getFloatSparseBlock(long vectorIndex, long vectorNum, ByteBuffer buf, ByteBuffer colIndicesBuf,
            ByteBuffer rowOffsetsBuf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        colIndicesBuf.order(ByteOrder.LITTLE_ENDIAN);
        rowOffsetsBuf.order(ByteOrder.LITTLE_ENDIAN);
        return getSparseBlock(DataFeatureUtils.InternalNumType.DAAL_SINGLE, vectorIndex, vectorNum, buf,
                colIndicesBuf.asLongBuffer(), rowOffsetsBuf.asLongBuffer());
    }

    protected long getIntSparseBlock(long vectorIndex, long vectorNum, ByteBuffer buf, ByteBuffer colIndicesBuf,
            ByteBuffer rowOffsetsBuf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        colIndicesBuf.order(ByteOrder.LITTLE_ENDIAN);
        rowOffsetsBuf.order(ByteOrder.LITTLE_ENDIAN);
        return getSparseBlock(DataFeatureUtils.InternalNumType.DAAL_INT32, vectorIndex, vectorNum, buf,
                colIndicesBuf.asLongBuffer(), rowOffsetsBuf.asLongBuffer());
    }

    private long[] colIndices;
    private long[] rowOffsets;

    private void initialize(Object data, long[] colIndices, long[] rowOffsets, long nFeatures, long nVectors) {
        this.jData = data;
        this.colIndices = colIndices;
        this.rowOffsets = rowOffsets;
        this.dataAllocatedInJava = true;
        this.cObject = initCSRNumericTable(nFeatures, nVectors);

        nJavaFeatures = nFeatures;
        nJavaVectors = nVectors;

        initDataDictionary(data.getClass().getComponentType(), nFeatures);
    }

    private void initDataDictionary(Class<?> cls, long nFeatures) {
        dict = new DataDictionary(getContext(), nFeatures, cGetCDataDictionary(cObject));
        dict.setFeature(cls, 0);
    }

    /* Creates CSR numeric table with nColumns columns and nRows rows */
    protected native long initCSRNumericTable(long nColumns, long nRows);

    @Override
    protected void onUnpack(DaalContext context) {
        if (dataAllocatedInJava) {
            initialize(this.jData, this.colIndices, this.rowOffsets, this.nJavaFeatures, this.nJavaVectors);
        } else {
            super.onUnpack(context);
        }
    }

    /* Gets index type of the C++ CSRNumericTable object */
    private native int getIndexType(long cObject);

    private native int cGetNumberOfRows(long cObject);

    private native int cGetDataSize(long cObject);

    /* Gets NIO buffer containing data of the C++ table */
    private native ByteBuffer getColIndicesBuffer(long cObject, ByteBuffer buffer);

    private native ByteBuffer getRowOffsetsBuffer(long cObject, ByteBuffer buffer);

    private native ByteBuffer getDoubleBuffer(long cObject);

    private native ByteBuffer getFloatBuffer(long cObject);

    private native ByteBuffer getLongBuffer(long cObject);
}
