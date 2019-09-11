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

import re
import sys

assert len(sys.argv) == 2

with open(sys.argv[1], "r") as sources:
    lines = sources.readlines()

mlc = False

for line in lines:
    line = line.rstrip()
    line = re.sub(r'#include "(.+)\.h"', r'', line)

    cmt = ""
    for (match, repl) in [(r'(?<!:)//(?P<rem>.*)', r'#\g<rem>'), # #4 '),
                          (r'/\*(?P<rem>.*)\*/', r'#\g<rem>'), # #5 \1'),
                          (r'(using namespace std)', r''),
                          (r'using namespace (.*)', r''),
                          (r'(for\s*(\S+\s+)?([^=\s]+)\s*=\s*([^;]+);\s*\w+\s*<\s*(\w+);\s*\w+\+\+)', r'for \3 in range(\4, \5):  #2.1 \1'),
                          (r'\.get\((\d+)\)(.*)', r'[\1]\2  # 15 .get(\1)'),
                          (r'(^(\s*)[\w\.]+\s*<\s*[\w\.]+\s*\>\s+)(\w+)\s+=(.*)', r'\2\3 = \4  #6.1 \1'),
                          (r'(([\w\.]+)\s*<\s*([\w\.]+)\s*\>\s+(\w+)\s+(\(.*\)))', r'\2_\3 \4\5  #6.2 \1'),
                          (r'\<\s*([\w\.]+)\s*\>(.*)', r'_\1\2  #6.3 <\1>'),
                          (r'\s*([{}])\s*', r''), # #7 \1'),
                          (r'(\s*)(services::)?(SharedPtr_)(\w+)(\s*)(\(\s*new\s+)([\w\.]+)(\(.*\))(\s*\))', r'\1\7\8 #8.1 SharedPtr<\4\5\6\7\8'),
                          (r'((services::)?SharedPtr<[^>]+>)\s+(\w+)\s*\(\s*new\s+(.*)\)', r'\3 = \4 #8.2 \1'),
                          (r'(\s*)(services::)?(SharedPtr_)(\w+)(\s*)(.*)', r'\1\4\5\6 #9 SharedPtr<\4>'),
                          # doesn't capture multi-line paramters:  (r'^(\s*)(\w+)\s+(\w+)\((.*)\)\s*;\s*', r'\1\3 = \2(\4)'),
                          (r'(int\s+)main\(.*\)', r'if __name__ == "__main__":'),
                          (r'^(\s*)([\w\.]+)\s+(\w+)\((.*[,;])\s*$', r'\1\3 = \2(\4) #10'),
                          (r'^(\s*)(?!def)([\w\.]+)\s+(\w+)\((.*)\)(?P<rem>.*)$', r'\1def \3(\4): \5 #12 \g<rem>'),
                          (r'^(\s*)(?!return)([\w\.]+)\s+(\w+);$', r'\1\3 = \2() #11 \2 \3'),
                          (r'^(\s*)((const\s+)?([\w\.]+))\s(\w+)\s*=\s*(\w+)\((.*)\)(.*)', r'\1\5 = \6(\7)\8 #12 \2'),
                          (r'^(\s*)((const\s+)?([\w\.]+))\s(?!__name__)(\w+)\s*=(.*)', r'\1\5 = \4(\6) #13 \4'),
                          (r'->|::', r'.'),
                          (r';', r''),
                          (r'true', r'True'),
                          (r'false', r'False'),
                          (r'-CPP-', r'-PY-'),
                          (r'\.cpp', r'.py'),
                          (r'C\+\+', r'Python'),
                          (r'(?<!_)float(?!(32|64))', r'np.float32'),
                          (r'(?<!_)double', r'np.float64'),
                          (r'(?<!_)(string|size_t|int|double|float)\((.+)\)(.*)', r'\2\3  #14 \1'),
                          (r'(\w+Ptr\((new\s+)?)(.+)\)', r'\3  #15 \1'),
                          (r'^(\s*)(?!return)([\.\w]+)(<\s*>)?\s+(\w+)\s*$', r'\1\4 = \2()  #16'),
                          (r'Table_(double|float,int)\((.*)\)\s*$', r'Table(\2, ntype=np.\1)  #17'),
                          (r'Distributed_(step\w+)\((.*)\)\s*$', r'Distributed(\2, step=\1)  #18 _\1'),
                          (r'=np.double', r'=np.Float64'),
                          (r'=np.float', r'=np.Float32'),
                          (r'=np.int', r'=np.Intc'),
                          (r'\(\s*,?\s*', r'('),
                          (r'std.cout << ', r'print('),
                          (r' << "', r' + "'),
                          (r' << (\S+)', r' + str(\1)'),
                          (r'(for|if)\((.*)\)', r'\1 \2  #22'),
                      ]:

        orgline = ""
        while not re.match(r'\s*#', line) and orgline != line:
            orgline = line
            lines = re.sub(match, repl, orgline).split('#')
            line = lines[0]
            for c in lines[1:]:
                cmt += "#" + c.rstrip()
    # multiline comments, one-lines should be eliminated
    if mlc:
        nl = re.sub(r'^(\s*)(.*)\*/\s*', r'#\1 \2', line)
        if nl != line:
            mlc = False
        else:
            nl = re.sub(r'^(\s*)(.*)', r'#\1 \2', line)
    else:
        nl = re.sub(r'^(\s*)/\*(.*)',  r'#\1 \2', line)
        if nl != line:
            mlc = True

    if nl == '' or mlc:
        nl = re.sub(r'^ *\#+ *\** *([\\<])', r'## \1', nl)
        nl = re.sub(r'^ *\# *\*+', r'#', nl)
#            print line, match

    print(nl + cmt)
