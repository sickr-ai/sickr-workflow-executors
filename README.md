# SICKR workflow executors

This private repository is the release source for SICKR-maintained workflow
executors. The orchestrator pins this repository as a submodule and bundles the
package into its private runtime artifact.

The executor catalog is data-driven. Every implementation must have exactly one
versioned entry in `manifests/builtin-executors.v1.json`; registration
conformance fails when an implementation or manifest exists without the other.
Organization and team executors use the same manifest fields and execution
input/output envelopes demonstrated by `sickr-workflow-extension-template`.

Updates are additive contract versions. A released workflow remains pinned to
the version it was published with. Removing an executor from the current
library never rewrites an existing workflow binding.

