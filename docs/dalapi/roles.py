# file: roles.py
#===============================================================================
# Copyright 2019-2020 Intel Corporation
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

import re
from docutils import nodes
from sphinx import roles

_term_ref_re = re.compile(r'(.+)<(.+)>', flags=re.DOTALL)
def capterm_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    xref_role = roles.XRefRole(innernodeclass=nodes.inline,
                               warn_dangling=True)
    term_match = _term_ref_re.match(text)
    if term_match:
        txt, ref = term_match.group(1), term_match.group(2)
    else:
        txt, ref = text, text
    fixed_term = f'{txt.strip()} <{ref.strip().capitalize()}>'
    return xref_role('std:term', rawtext, fixed_term, lineno, inliner, options, content)


_term_txt_ref_re = re.compile(r'(.*)<(.+)>(.*)', flags=re.DOTALL)
def txtref_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    xref_role = roles.XRefRole(lowercase=True,
                               innernodeclass=nodes.inline,
                               warn_dangling=True)
    def extract_ref_words(ref):
        return [item for sub in ref.split('-') for item in sub.split('_')]

    def make_term_text(words, ref, suffix=''):
        txt = ' '.join(words).strip()
        txt = txt[0].lower() + txt[1:]
        return f"{txt}{suffix} <{ref.strip()}>"

    term_match = _term_txt_ref_re.match(text)
    if term_match:
        alias, ref, suffix = term_match.group(1), term_match.group(2), term_match.group(3)
        if len(alias) > 0 and len(suffix) == 0:
            fixed_term = text
        elif len(alias) == 0 and len(suffix) > 0:
            words = extract_ref_words(ref)
            fixed_term = make_term_text(words, ref, suffix)
        else:
            raise RuntimeError('Unexpected role syntax: ' + rawtext)
    else:
        ref, words = text, extract_ref_words(text)
        fixed_term = make_term_text(words, ref)

    return xref_role('std:ref', rawtext, fixed_term, lineno, inliner, options, content)
