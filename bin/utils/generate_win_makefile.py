#===============================================================================
# Copyright 2014-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

##  Content:
##     Intel(R) DAAL examples makefiles for windows generator
##******************************************************************************

import os
import sys
import glob

def get_rules_list(dir):
    cpp_paths = glob.glob('{}/**/*.cpp'.format(dir))
    relative_cpp_paths = [ os.path.join('source', os.path.relpath(x, dir)) for x in cpp_paths ]
    exe_names =  [os.path.basename(x).replace('.cpp', '.exe') for x in cpp_paths ]
    return list(zip(exe_names, relative_cpp_paths))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: generate_win_makefile source_dir out_dir')
    else:
        source_dir, out_dir = sys.argv[1], sys.argv[2]

        rules = get_rules_list(os.path.join(source_dir, 'source'))

        examples_list_str = 'EXAMPLES_LIST = \\\n' + '{}' * len(rules)
        examples_list_str = examples_list_str.format(*[exe.replace('.exe', '+') + '\\\n' for exe, _ in rules])

        make_build_rule_body = '$(CC) $(COPTS)\$@ $** $(LOPTS)'
        make_build_rules_str = ('{}' * len(rules)).format(
            *['{0} : {1}\n\t{2}\n'.format(exe,
                                          cpp,
                                          make_build_rule_body) for exe, cpp in rules])

        make_run_rule_body = '$(RES_DIR)\$** > $(RES_DIR)\$@'
        make_run_rules_str = ('{}' * len(rules)).format(
            *['{0} : {1}\n\t{2}\n'.format(exe.replace('.exe', '.res'),
                                          exe,
                                          make_run_rule_body) for exe, _ in rules])

        with open(os.path.join(source_dir, 'makefile_win'), 'r') as makefile_win:
            template_str = makefile_win.read()

        make_str = template_str.format(examples_list=examples_list_str,
                                       make_build_rules=make_build_rules_str,
                                       make_run_rules=make_run_rules_str)

        with open(os.path.join(out_dir, 'makefile'), 'w') as makefile:
            makefile.write(make_str)
