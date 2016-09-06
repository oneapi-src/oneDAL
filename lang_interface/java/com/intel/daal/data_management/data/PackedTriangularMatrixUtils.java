/* file: PackedTriangularMatrixUtils.java */
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

import java.io.Serializable;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import com.intel.daal.services.DaalContext;

class PackedTriangularMatrixUtils {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    interface TriangularAccessIface {
        int getPosition(int row, int column, int nDim);
    }

    static class TriangularAccess {
        static public TriangularAccessIface getAccess(NumericTable.StorageLayout packedLayout) {
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                return triangularAccessList[0];
            } else {
                return triangularAccessList[1];
            }
        }
    }

    static class UpperTriangularAccess implements TriangularAccessIface, Serializable {
        @Override
        public int getPosition(int row, int column, int nDim) {
            if (row > column) {
                int tmp = row;
                row = column;
                column = tmp;
            }
            int rowStartOffset = ((2 * nDim - 1 * (row - 1)) * row) / 2;
            int colStartOffset = column - row;
            return rowStartOffset + colStartOffset;
        }
    }

    static class LowerTriangularAccess implements TriangularAccessIface, Serializable {
        @Override
        public int getPosition(int row, int column, int nDim) {
            if (row < column) {
                int tmp = row;
                row = column;
                column = tmp;
            }
            int rowStartOffset = ((2 + 1 * (row - 1)) * row) / 2;
            int colStartOffset = column;
            return rowStartOffset + colStartOffset;
        }
    }

    interface TriangularUpCastIface {
        void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout);
        void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout);
    }

    interface TriangularDownCastIface {
        void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout);
        void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout);
    }

    /** @private */
    static class TriangularUpCast {
        static public TriangularUpCastIface getCast(Class<?> fromCls, Class<?> toCls) {
            return triangularUpCasts[DataFeatureUtils.getTypeIndex(fromCls)][DataFeatureUtils.getTypeIndex(toCls)];
        }
    }

    /** @private */
    static class TriangularDownCast {
        static public TriangularDownCastIface getCast(Class<?> fromCls, Class<?> toCls) {
            return triangularDownCasts[DataFeatureUtils.getTypeIndex(fromCls)][DataFeatureUtils.getTypeIndex(toCls)];
        }
    }

    /** @private */
    static class TriangularUpCastDouble2Double implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (double)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (double)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularDownCastDouble2Double implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            double[] data = (double[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = buf.get (row - firstRow);
            }
        }
    }

    /** @private */
    static class TriangularUpCastDouble2Float implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (float)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, (float)data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (float)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, (float)data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularDownCastDouble2Float implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = (float)buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            float[] data = (float[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = (float)buf.get (row - firstRow);
            }
        }
    }

    /** @private */
    static class TriangularUpCastDouble2Int implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (int)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, (int)data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            double[] data = (double[])src;
            IntBuffer buf = (IntBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (int)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, (int)data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularDownCastDouble2Int implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = (int)buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            int[] data = (int[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = (int)buf.get (row - firstRow);
            }
        }
    }

    /** @private */
    static class TriangularUpCastFloat2Double implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (double)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, (double)data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (double)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, (double)data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularDownCastFloat2Double implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = (double)buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            double[] data = (double[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = (double)buf.get (row - firstRow);
            }
        }
    }

    /** @private */
    static class TriangularUpCastFloat2Float implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (float)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (float)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularDownCastFloat2Float implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            float[] data = (float[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = buf.get (row - firstRow);
            }
        }
    }

    /** @private */
    static class TriangularUpCastFloat2Int implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (int)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, (int)data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            float[] data = (float[])src;
            IntBuffer buf = (IntBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (int)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, (int)data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularDownCastFloat2Int implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = (int)buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            int[] data = (int[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = (int)buf.get (row - firstRow);
            }
        }
    }

    /** @private */
    static class TriangularUpCastInt2Double implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (double)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, (double)data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (double)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, (double)data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularDownCastInt2Double implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = (double)buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            double[] data = (double[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = (double)buf.get (row - firstRow);
            }
        }
    }

    /** @private */
    static class TriangularUpCastInt2Float implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (float)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, (float)data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (float)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, (float)data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularDownCastInt2Float implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = (float)buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            float[] data = (float[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = (float)buf.get (row - firstRow);
            }
        }
    }

    /** @private */
    static class TriangularUpCastInt2Int implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (int)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            int[] data = (int[])src;
            IntBuffer buf = (IntBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (int)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularDownCastInt2Int implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            int[] data = (int[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = buf.get (row - firstRow);
            }
        }
    }

    /** @private */
    static class TriangularUpCastLong2Double implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (double)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, (double)data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (double)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, (double)data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularUpCastLong2Float implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (float)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, (float)data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (float)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, (float)data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularUpCastLong2Int implements TriangularUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    zeroBegin = 0;
                    zeroEnd = row;
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    zeroBegin = row + 1;
                    zeroEnd = nDim;
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = zeroBegin; column < zeroEnd; column++) {
                    buf.put (row * nDim + column, (int)0);
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    buf.put (row * nDim + column, (int)data[triangularAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            dst.position(0);
            long[] data = (long[])src;
            IntBuffer buf = (IntBuffer)dst;
            int zeroBegin, zeroEnd, nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                zeroBegin = column + 1;
                zeroEnd = nDim;
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                zeroBegin = 0;
                zeroEnd = column;
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (zeroBegin < firstRow) zeroBegin = firstRow;
            if (zeroEnd > firstRow + nRows) zeroEnd = firstRow + nRows;
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = zeroBegin; row < zeroEnd; row++) {
                buf.put (row - firstRow, (int)0);
            }
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                buf.put (row - firstRow, (int)data[triangularAccess.getPosition(row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class TriangularDownCastDouble2Long implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = (long)buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            DoubleBuffer buf = (DoubleBuffer)src;
            long[] data = (long[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = (long)buf.get (row - firstRow);
            }
        }
    }

    /** @private */
    static class TriangularDownCastFloat2Long implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = (long)buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            FloatBuffer buf = (FloatBuffer)src;
            long[] data = (long[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = (long)buf.get (row - firstRow);
            }
        }
    }

    /** @private */
    static class TriangularDownCastInt2Long implements TriangularDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                int nonZeroBegin, nonZeroEnd;
                if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                    nonZeroBegin = row;
                    nonZeroEnd = nDim;
                } else {
                    nonZeroBegin = 0;
                    nonZeroEnd = row + 1;
                }
                for (int column = nonZeroBegin; column < nonZeroEnd; column++) {
                    data[triangularAccess.getPosition(firstRow + row, column, nDim)] = (long)buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, NumericTable.StorageLayout packedLayout) {
            TriangularAccessIface triangularAccess = TriangularAccess.getAccess(packedLayout);
            IntBuffer buf = (IntBuffer)src;
            long[] data = (long[])dst;
            int nonZeroBegin, nonZeroEnd;
            if (packedLayout == NumericTable.StorageLayout.upperPackedTriangularMatrix) {
                nonZeroBegin = 0;
                nonZeroEnd = column + 1;
            } else {
                nonZeroBegin = column;
                nonZeroEnd = nDim;
            }
            if (nonZeroBegin < firstRow) nonZeroBegin = firstRow;
            if (nonZeroEnd > firstRow + nRows) nonZeroEnd = firstRow + nRows;
            for (int row = nonZeroBegin; row < nonZeroEnd; row++) {
                data[triangularAccess.getPosition(row, column, nDim)] = (long)buf.get (row - firstRow);
            }
        }
    }

    static TriangularUpCastIface triangularUpCasts[][] = {
        {(TriangularUpCastIface)(new TriangularUpCastFloat2Float()), (TriangularUpCastIface)(new TriangularUpCastFloat2Double()),
         (TriangularUpCastIface)(new TriangularUpCastFloat2Int())},
        {(TriangularUpCastIface)(new TriangularUpCastDouble2Float()), (TriangularUpCastIface)(new TriangularUpCastDouble2Double()),
         (TriangularUpCastIface)(new TriangularUpCastDouble2Int())},
        {(TriangularUpCastIface)(new TriangularUpCastInt2Float()), (TriangularUpCastIface)(new TriangularUpCastInt2Double()),
         (TriangularUpCastIface)(new TriangularUpCastInt2Int())},
        {(TriangularUpCastIface)(new TriangularUpCastLong2Float()), (TriangularUpCastIface)(new TriangularUpCastLong2Double()),
         (TriangularUpCastIface)(new TriangularUpCastLong2Int())}
    };

    static TriangularDownCastIface triangularDownCasts[][] = {
        {(TriangularDownCastIface)(new TriangularDownCastFloat2Float()), (TriangularDownCastIface)(new TriangularDownCastFloat2Double()),
         (TriangularDownCastIface)(new TriangularDownCastFloat2Int()), (TriangularDownCastIface)(new TriangularDownCastFloat2Long())},
        {(TriangularDownCastIface)(new TriangularDownCastDouble2Float()), (TriangularDownCastIface)(new TriangularDownCastDouble2Double()),
         (TriangularDownCastIface)(new TriangularDownCastDouble2Int()), (TriangularDownCastIface)(new TriangularDownCastDouble2Long())},
        {(TriangularDownCastIface)(new TriangularDownCastInt2Float()), (TriangularDownCastIface)(new TriangularDownCastInt2Double()),
         (TriangularDownCastIface)(new TriangularDownCastInt2Int()), (TriangularDownCastIface)(new TriangularDownCastInt2Long())},
    };

    static TriangularAccessIface triangularAccessList[] = {
        (TriangularAccessIface)(new UpperTriangularAccess()),
        (TriangularAccessIface)(new LowerTriangularAccess())
    };
}
