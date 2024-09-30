# oneDAL Design Document (RFC)

> Please follow the document style requirements listed
> [here](../README.md#document-style).

## Introduction

Short description of the idea proposed with explained motivation. The
motivation could be:
- Widespread usage of the proposed feature in different frameworks or
  applications. Provide references to support the claim.
- Improved users experience for API changes and extensions. Code snippets to
  showcase the benefits would be nice here.
- Performance improvements with the data, if available.
- Improved engineering practices.

Introduction may also include any additional information that sheds light on
the proposal, such as history of the matter, links to relevant issues and
discussions, etc.

## Proposal

A full and detailed description of the proposal, with highlighted consequences.

Depending on the kind of the proposal, the description should cover:

- The expected performance benefit. This usually best presented as a profiling
  information from a workload showing that a particular operation takes
  significant percentage of the total time and thus is a good optimization
  candidate.

- The definition of algorithm including interface
  and semantics. It is OK to have sketches for the interface, but the
  semantics should be fairly well defined.

A proposal should include the alternatives that were considered with listed
pros and cons. The alternatives should be clearly separated to make possible
discussions clear.

Pay close attention to the following aspects of the library:
- API and ABI backwards compatibility. The library follows semantic versioning
  so if any of those interfaces are to be broken we need to state that
  explicitly.
- Performance implications, as this is one of the main goals of the library.
- Dependencies and support matrix: does the proposal brings any new
  dependencies or affects the supported configurations, including 
  legal/license implication

Some other common subsections here are:
- Discussion: some people like to list all the options first (as separate
  subsections), and then have a dedicated section with the discussion/
- Listing of the proposed API and examples of its usage.
- Testing aspects.
- Execution plan if approved, aka next steps.

## Open Questions

List here the significant uncertainties that are relevant to the proposal.

However, sometimes the answers on these questions won't be found even if the
proposal is accepted, say, by leaving the current implementation as is.
