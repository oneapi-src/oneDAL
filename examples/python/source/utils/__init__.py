# file: __init__.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

## <a name="DAAL-EXAMPLE-PY-UTIL"></a>

from __future__ import print_function

import sys
import numpy as np

from daal.data_management import (
    CSRNumericTable, NumericTableIface, readOnly, BlockDescriptor, packed_mask,
    DataSourceIface, FileDataSource, HomogenTensor, SubtensorDescriptor
)

CRC_REM = [
    0x00000000, 0x741B8CD6, 0xE83719AC, 0x9C2C957A, 0xA475BF8E, 0xD06E3358, 0x4C42A622, 0x38592AF4,
    0x3CF0F3CA, 0x48EB7F1C, 0xD4C7EA66, 0xA0DC66B0, 0x98854C44, 0xEC9EC092, 0x70B255E8, 0x04A9D93E,
    0x79E1E794, 0x0DFA6B42, 0x91D6FE38, 0xE5CD72EE, 0xDD94581A, 0xA98FD4CC, 0x35A341B6, 0x41B8CD60,
    0x4511145E, 0x310A9888, 0xAD260DF2, 0xD93D8124, 0xE164ABD0, 0x957F2706, 0x0953B27C, 0x7D483EAA,
    0xF3C3CF28, 0x87D843FE, 0x1BF4D684, 0x6FEF5A52, 0x57B670A6, 0x23ADFC70, 0xBF81690A, 0xCB9AE5DC,
    0xCF333CE2, 0xBB28B034, 0x2704254E, 0x531FA998, 0x6B46836C, 0x1F5D0FBA, 0x83719AC0, 0xF76A1616,
    0x8A2228BC, 0xFE39A46A, 0x62153110, 0x160EBDC6, 0x2E579732, 0x5A4C1BE4, 0xC6608E9E, 0xB27B0248,
    0xB6D2DB76, 0xC2C957A0, 0x5EE5C2DA, 0x2AFE4E0C, 0x12A764F8, 0x66BCE82E, 0xFA907D54, 0x8E8BF182,
    0x939C1286, 0xE7879E50, 0x7BAB0B2A, 0x0FB087FC, 0x37E9AD08, 0x43F221DE, 0xDFDEB4A4, 0xABC53872,
    0xAF6CE14C, 0xDB776D9A, 0x475BF8E0, 0x33407436, 0x0B195EC2, 0x7F02D214, 0xE32E476E, 0x9735CBB8,
    0xEA7DF512, 0x9E6679C4, 0x024AECBE, 0x76516068, 0x4E084A9C, 0x3A13C64A, 0xA63F5330, 0xD224DFE6,
    0xD68D06D8, 0xA2968A0E, 0x3EBA1F74, 0x4AA193A2, 0x72F8B956, 0x06E33580, 0x9ACFA0FA, 0xEED42C2C,
    0x605FDDAE, 0x14445178, 0x8868C402, 0xFC7348D4, 0xC42A6220, 0xB031EEF6, 0x2C1D7B8C, 0x5806F75A,
    0x5CAF2E64, 0x28B4A2B2, 0xB49837C8, 0xC083BB1E, 0xF8DA91EA, 0x8CC11D3C, 0x10ED8846, 0x64F60490,
    0x19BE3A3A, 0x6DA5B6EC, 0xF1892396, 0x8592AF40, 0xBDCB85B4, 0xC9D00962, 0x55FC9C18, 0x21E710CE,
    0x254EC9F0, 0x51554526, 0xCD79D05C, 0xB9625C8A, 0x813B767E, 0xF520FAA8, 0x690C6FD2, 0x1D17E304,
    0x5323A9DA, 0x2738250C, 0xBB14B076, 0xCF0F3CA0, 0xF7561654, 0x834D9A82, 0x1F610FF8, 0x6B7A832E,
    0x6FD35A10, 0x1BC8D6C6, 0x87E443BC, 0xF3FFCF6A, 0xCBA6E59E, 0xBFBD6948, 0x2391FC32, 0x578A70E4,
    0x2AC24E4E, 0x5ED9C298, 0xC2F557E2, 0xB6EEDB34, 0x8EB7F1C0, 0xFAAC7D16, 0x6680E86C, 0x129B64BA,
    0x1632BD84, 0x62293152, 0xFE05A428, 0x8A1E28FE, 0xB247020A, 0xC65C8EDC, 0x5A701BA6, 0x2E6B9770,
    0xA0E066F2, 0xD4FBEA24, 0x48D77F5E, 0x3CCCF388, 0x0495D97C, 0x708E55AA, 0xECA2C0D0, 0x98B94C06,
    0x9C109538, 0xE80B19EE, 0x74278C94, 0x003C0042, 0x38652AB6, 0x4C7EA660, 0xD052331A, 0xA449BFCC,
    0xD9018166, 0xAD1A0DB0, 0x313698CA, 0x452D141C, 0x7D743EE8, 0x096FB23E, 0x95432744, 0xE158AB92,
    0xE5F172AC, 0x91EAFE7A, 0x0DC66B00, 0x79DDE7D6, 0x4184CD22, 0x359F41F4, 0xA9B3D48E, 0xDDA85858,
    0xC0BFBB5C, 0xB4A4378A, 0x2888A2F0, 0x5C932E26, 0x64CA04D2, 0x10D18804, 0x8CFD1D7E, 0xF8E691A8,
    0xFC4F4896, 0x8854C440, 0x1478513A, 0x6063DDEC, 0x583AF718, 0x2C217BCE, 0xB00DEEB4, 0xC4166262,
    0xB95E5CC8, 0xCD45D01E, 0x51694564, 0x2572C9B2, 0x1D2BE346, 0x69306F90, 0xF51CFAEA, 0x8107763C,
    0x85AEAF02, 0xF1B523D4, 0x6D99B6AE, 0x19823A78, 0x21DB108C, 0x55C09C5A, 0xC9EC0920, 0xBDF785F6,
    0x337C7474, 0x4767F8A2, 0xDB4B6DD8, 0xAF50E10E, 0x9709CBFA, 0xE312472C, 0x7F3ED256, 0x0B255E80,
    0x0F8C87BE, 0x7B970B68, 0xE7BB9E12, 0x93A012C4, 0xABF93830, 0xDFE2B4E6, 0x43CE219C, 0x37D5AD4A,
    0x4A9D93E0, 0x3E861F36, 0xA2AA8A4C, 0xD6B1069A, 0xEEE82C6E, 0x9AF3A0B8, 0x06DF35C2, 0x72C4B914,
    0x766D602A, 0x0276ECFC, 0x9E5A7986, 0xEA41F550, 0xD218DFA4, 0xA6035372, 0x3A2FC608, 0x4E344ADE
]


def readTextFile(dataset_filename):
    data_list = []
    with open(dataset_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data_list.append(tuple([float(x) for x in line.split(',')[:-1]]))
    return np.array(data_list, dtype=np.ubyte).flatten()


def createSparseTable(dataset_filename, ntype=np.float64):

    try:
        f = open(dataset_filename, 'r')
    except Exception:
        sys.exit("Unable to open '{}'.".format(dataset_filename))

    lines = f.readlines()
    row_index_line = lines[0]
    columns_line = lines[1]
    values_line = lines[2]

    row_offsets = np.array(row_index_line.rstrip('\n,').split(','), dtype=np.uint64)
    num_vectors = len(row_offsets) - 1
    col_indices = np.array(columns_line.rstrip('\n,').split(','), dtype=np.uint64)
    num_columns = len(col_indices)
    data = np.array(values_line.rstrip('\n,').split(','), dtype=ntype)
    num_nonzeros = len(data)

    max_col = 0

    for i in range(num_columns):
        if col_indices[i] > max_col:
            max_col = col_indices[i]
    if sys.version_info[0] < 3:
        num_features = long(max_col)
    else:
        num_features = int(max_col)

    if (
        num_columns != num_nonzeros or
        num_nonzeros != int(row_offsets[num_vectors]) - 1 or
        num_features == 0 or
        num_vectors == 0
    ):
        print("Incorrect format of file")
        sys.exit()

    numeric_table = CSRNumericTable(data, col_indices, row_offsets, num_features, num_vectors)

    return numeric_table


def printAprioriItemsets(large_itemsets_table, large_itemsets_support_table,
                         itemsets_to_print=20):
    large_itemset_count = large_itemsets_support_table.getNumberOfRows()
    num_items_in_large_itemsets = large_itemsets_table.getNumberOfRows()

    large_itemsets = large_itemsets_table.getBlockOfRowsAsDouble(0, num_items_in_large_itemsets).flatten()
    large_itemsets_support_data = large_itemsets_support_table.getBlockOfRowsAsDouble(0, large_itemset_count).flatten()

    large_itemsets_array = [[] for x in range(large_itemset_count)]

    for i in range(num_items_in_large_itemsets):
        large_itemsets_array[int(large_itemsets[2 * i])].append(large_itemsets[2 * i + 1])

    support_array = [0] * large_itemset_count

    for i in range(large_itemset_count):
        support_array[int(large_itemsets_support_data[2 * i])] = large_itemsets_support_data[2 * i + 1]

    print("\nApriori example program results\n")
    print("Last " + str(itemsets_to_print) + " large itemsets: \n")
    print("Itemset\t\t\tSupport")

    if large_itemset_count > itemsets_to_print and itemsets_to_print != 0:
        i_min = large_itemset_count - itemsets_to_print
    else:
        i_min = 0

    for i in range(i_min, large_itemset_count):
        print("{", end='')
        for l in range(len(large_itemsets_array[i]) - 1):
            print("{:.0f}".format(large_itemsets_array[i][l]), end=', ')
        print("{:.0f}}}\t\t".format(large_itemsets_array[i][len(large_itemsets_array[i]) - 1]), end='')

        print("{:.0f}".format(support_array[i]))


def printAprioriRules(left_items_table, right_items_table, confidence_table,
                      num_rules_to_print=20):

    num_rules = confidence_table.getNumberOfRows()
    num_left_items = left_items_table.getNumberOfRows()
    num_right_items = right_items_table.getNumberOfRows()

    left_items = left_items_table.getBlockOfRowsAsDouble(0, num_left_items).flatten()
    right_items = right_items_table.getBlockOfRowsAsDouble(0, num_right_items).flatten()
    confidence = confidence_table.getBlockOfRowsAsDouble(0, num_rules).flatten()

    left_items_array = [[] for x in range(num_rules)]

    if num_rules == 0:
        print("\nNo association rules were found ")
        return

    for i in range(num_left_items):
        left_items_array[int(left_items[2 * i])].append(left_items[2 * i + 1])

    right_items_array = [[] for x in range(num_right_items)]

    for i in range(num_right_items):
        right_items_array[int(right_items[2 * i])].append(right_items[2 * i + 1])

    confidence_array = [0] * num_rules

    for i in range(num_rules):
        confidence_array[i] = confidence[i]

    print("\nLast {} association rules: \n".format(num_rules_to_print))
    print("Rule\t\t\t\tConfidence")
    if num_rules > num_rules_to_print and num_rules_to_print != 0:
        i_min = num_rules - num_rules_to_print
    else:
        i_min = 0

    for i in range(i_min, num_rules):
        print("{", end='')
        for l in range(len(left_items_array[i]) - 1):
            print("{:.0f}, ".format(left_items_array[i][l]), end='')
        print("{:.0f}}} => {{".format(left_items_array[i][len(left_items_array[i]) - 1]), end='')

        for l in range(len(right_items_array[i]) - 1):
            print("{:.0f}, ".format(right_items_array[i][l]), end='')
        print("{:.0f}}}\t\t".format(right_items_array[i][len(right_items_array[i]) - 1]), end='')

        print("{0:.6g}".format(confidence_array[i]))


def isFull(layout):
    layout_int = int(layout)
    if packed_mask & layout_int:
        return False
    return True


def printArray(array, num_printed_cols, num_printed_rows, num_cols, message,
               interval=10, flt64=True):
    print(message)
    flat_array = array.flatten()
    decimals = '3' if flt64 else '0'
    for i in range(num_printed_rows):
        for j in range(num_printed_cols):
            print("{:<{width}.{dec}f}".format(
                flat_array[i * num_cols + j], width=interval, dec=decimals), end=''
            )
        print()
    print()


def isUpper(layout):
    if (
        layout == NumericTableIface.upperPackedSymmetricMatrix or
        layout == NumericTableIface.upperPackedTriangularMatrix
    ):
        return True
    return False


def isLower(layout):
    if (
        layout == NumericTableIface.lowerPackedSymmetricMatrix or
        layout == NumericTableIface.lowerPackedTriangularMatrix
    ):
        return True
    return False


def printLowerArray(array, num_printed_rows, message, interval=10):
    cols = 1
    print(message)
    for i in range(num_printed_rows):
        for j in range(cols):
            print("{:<{width}.3f}".format(array[i][j], width=interval), end='')
        print()
        cols += 1
    print()


def printUpperArray(array, num_printed_cols, num_printed_rows, num_cols,
                    message, interval=10):
    print(message)
    for i in range(num_printed_rows):
        for j in range(i):
            print(' ' * 10, end='')
        for j in range(i, num_printed_cols):
            print("{:<{width}.3f}".format(array[i][j], width=interval), end='')
        print()
    print()


def printNumericTable(data_table, message='', num_printed_rows=0, num_printed_cols=0,
                      interval=10):
    num_rows = data_table.getNumberOfRows()
    num_cols = data_table.getNumberOfColumns()
    layout = data_table.getDataLayout()

    if num_printed_rows != 0:
        num_printed_rows = min(num_rows, num_printed_rows)
    else:
        num_printed_rows = num_rows

    if num_printed_cols != 0:
        num_printed_cols = min(num_cols, num_printed_cols)
    else:
        num_printed_cols = num_cols

    block = BlockDescriptor()
    if isFull(layout) or layout == NumericTableIface.csrArray:
        data_table.getBlockOfRows(0, num_rows, readOnly, block)
        printArray(block.getArray(), num_printed_cols, num_printed_rows,
                   num_cols, message, interval)
        data_table.releaseBlockOfRows(block)
    else:
        packed_table = data_table.getBlockOfRowsAsDouble(0, num_rows)

        if isLower(layout):
            printLowerArray(packed_table, num_printed_rows, message, interval)
        elif isUpper(layout):
            printUpperArray(packed_table, num_printed_cols, num_printed_rows,
                            num_cols, message, interval)


def printNumericTables(data_table_1, data_table_2, title_1='', title_2='',
                       message='', num_printed_rows=0, interval=15, flt64=True):
    num_rows_1 = data_table_1.getNumberOfRows()
    num_rows_2 = data_table_2.getNumberOfRows()
    num_cols_1 = data_table_1.getNumberOfColumns()
    num_cols_2 = data_table_2.getNumberOfColumns()

    num_rows = min(num_rows_1, num_rows_2)
    if num_printed_rows != 0:
        num_rows = min(min(num_rows_1, num_rows_2), num_printed_rows)

    block1 = BlockDescriptor()
    block2 = BlockDescriptor()
    data_table_1.getBlockOfRows(0, num_rows, readOnly, block1)
    data_table_2.getBlockOfRows(0, num_rows, readOnly, block2)

    data_float64_1 = block1.getArray()
    data_float64_2 = block2.getArray()

    data_float64_1 = data_float64_1.flatten()
    data_float64_2 = data_float64_2.flatten()

    decimals = '3' if flt64 else '0'

    print(message)
    print("{:<{width}}".format(title_1, width=(interval * num_cols_1)), end='')
    print("{:<{width}}".format(title_2, width=(interval * num_cols_2)))
    for i in range(num_rows):
        for j in range(num_cols_1):
            print("{:<{width}.{dec}f}".format(data_float64_1[i * num_cols_1 + j], width=interval, dec=decimals), end='')
        for j in range(num_cols_2):
            print("{:<{width}.0f}".format(data_float64_2[i * num_cols_2 + j], width=interval), end='')
        print()
    print()

    data_table_1.releaseBlockOfRows(block1)
    data_table_2.releaseBlockOfRows(block2)


def getCRC32(input, prevRes=0):
    from binascii import crc32
    return crc32(input, prevRes)


def copyBytes(dst, src, size):
    for i in range(size):
        dst[i] = src[i]

def printALSRatings(usersOffsetTable, itemsOffsetTable, ratings):
    nUsers = ratings.getNumberOfRows()
    nItems = ratings.getNumberOfColumns()
    ratingsData = ratings.getBlockOfRowsAsDouble(0, nUsers).flatten()
    usersOffset = usersOffsetTable.getBlockOfRowsAsInt(0, 1).flatten()[0]
    itemsOffset = itemsOffsetTable.getBlockOfRowsAsInt(0, 1).flatten()[0]

    print(" User ID, Item ID, rating")
    for i in range(nUsers):
        for j in range(nItems):
            print("{}, {}, {:.6g}".format(i + usersOffset, j + itemsOffset, ratingsData[i * nItems + j]))



def printTensor(dataTable, message="", nPrintedRows=0, nPrintedCols=0, interval=10):
    dims = dataTable.getDimensions()
    nRows = int(dims[0])

    if nPrintedRows != 0:
        nPrintedRows = min(nRows, nPrintedRows)
    else:
        nPrintedRows = nRows

    block = SubtensorDescriptor()

    dataTable.getSubtensor([], 0, nPrintedRows, readOnly, block)

    nCols = int(block.getSize() / nPrintedRows)

    if nPrintedCols != 0:
        nPrintedCols = min(nCols, nPrintedCols)
    else:
        nPrintedCols = nCols

    printArray(block.getArray(), int(nPrintedCols), int(nPrintedRows), int(nCols), message, interval)
    dataTable.releaseSubtensor(block)


def printTensors(dataTable1, dataTable2, title1="", title2="", message="", nPrintedRows=0, interval=15):
    dims1 = dataTable1.getDimensions()
    nRows1 = int(dims1[0])

    if nPrintedRows != 0:
        nPrintedRows = min(nRows1, nPrintedRows)
    else:
        nPrintedRows = nRows1

    block1 = SubtensorDescriptor()
    dataTable1.getSubtensor([], 0, nPrintedRows, readOnly, block1)
    nCols1 = int(block1.getSize() / nPrintedRows)

    dims2 = dataTable2.getDimensions()
    nRows2 = int(dims2[0])

    if nPrintedRows != 0:
        nPrintedRows = min(nRows2, nPrintedRows)
    else:
        nPrintedRows = nRows2

    block2 = SubtensorDescriptor()
    dataTable2.getSubtensor([], 0, nPrintedRows, readOnly, block2)
    nCols2 = int(block2.getSize() / nPrintedRows)

    dataType1 = block1.getArray().flatten()
    dataType2 = block2.getArray().flatten()

    print(message)
    print("{:<{width}}".format(title1, width=(interval * nCols1)), end='')
    print("{:<{width}}".format(title2, width=(interval * nCols2)))

    for i in range(nPrintedRows):
        for j in range(nCols1):
            print("{v:<{width}.0f}".format(v=dataType1[i * nCols1 + j], width=interval), end='')

        for j in range(nCols2):
            print("{:<{width}.3f}".format(dataType2[i * nCols2 + j], width=int(interval / 2)), end='')
        print()
    print()

    dataTable1.releaseSubtensor(block1)
    dataTable2.releaseSubtensor(block2)


def printTensor3d(dataTable, message="", nFirstDim=0, nSecondDim=0, interval=10):
    dims = dataTable.getDimensions()
    nRows = int(dims[0])
    nCols = int(dims[1])

    if nFirstDim != 0:
        nFirstDim = min(nRows, nFirstDim)
    else:
        nFirstDim = nRows

    if nSecondDim != 0:
        nSecondDim = min(nCols, nSecondDim)
    else:
        nSecondDim = nCols

    block = SubtensorDescriptor()

    print(message)
    for i in range(nFirstDim):
        dataTable.getSubtensor([i], 0, nSecondDim, readOnly, block)

        nThirdDim = block.getSize() / nSecondDim

        printArray(block.getArray(), int(nThirdDim), int(nSecondDim), int(nThirdDim), "", interval)

        dataTable.releaseSubtensor(block)


def readTensorFromCSV(datasetFileName, allowOneColumn=False):
    dataSource = FileDataSource(datasetFileName,
                                DataSourceIface.doAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)
    dataSource.loadDataBlock()

    nt = dataSource.getNumericTable()
    size = nt.getNumberOfRows()
    block = BlockDescriptor()
    nt.getBlockOfRows(0, size, readOnly, block)
    blockData = block.getArray().flatten()

    dims = [size]
    if nt.getNumberOfColumns() > 1 or allowOneColumn:
        dims.append(nt.getNumberOfColumns())
        size *= dims[1]

    tensorData = np.array(blockData, dtype=np.float32)

    nt.releaseBlockOfRows(block)

    tensorData.shape = dims
    tensor = HomogenTensor(tensorData, ntype=np.float32)

    return tensor
