# file: rst_examples.py
#===============================================================================
# Copyright 2014-2020 Intel Corporation
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

import os
import shutil

def create_rst(filedir, filename):
    rst_content = '.. _' + filename + ':' + '\n\n' + filename + '\n' + '#' * len(filename) + '\n' + '\n' + \
    '.. literalinclude:: ../../../examples/cpp_sycl/source/' + filedir + \
    '/' + filename + '\n' + '  ' + ':language: cpp' + '\n'

    return(rst_content)

def list_examples(path):
    examples = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.cpp' in file:
                examples.append(os.path.join(root, file))

    examples = [file.split(path)[1].split('\\')[1:] for file in examples]

    return(examples)

def write_examples(files, path):
    for file in files:
        with open(os.path.join(path, file[1].split('.')[0] + '.rst'), 'w') as rst_file:
            rst_file.write(create_rst(file[0], file[1]))

def run(script_dir):
    example_path = os.path.join(script_dir, '..', 'examples\\cpp_sycl\\source')
    rst_path = os.path.join(script_dir, 'source\\examples_sycl')

    if os.path.exists(rst_path):
        shutil.rmtree(rst_path)
    os.makedirs(rst_path)

    files = list_examples(example_path)
    write_examples(files, rst_path)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run(script_dir)
