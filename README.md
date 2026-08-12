# SICKR workflow executors

This private repository is the release source for SICKR-maintained workflow
executors. The orchestrator pins this repository as a squashed Git subtree and bundles the
package into its private runtime artifact.
The current implementation library targets the `sickr==0.1.0` host SDK; that
compatibility is declared by the `host` project extra rather than relying on an
undocumented neighboring checkout.

The executor catalog is data-driven. Every implementation must have exactly one
versioned entry in `manifests/builtin-executors.v1.json`; registration
conformance fails when an implementation or manifest exists without the other.
Organization and team executors use the same manifest fields and execution
input/output envelopes demonstrated by `sickr-workflow-extension-template`.
Bundled executors use trusted in-process transport, while organization and team
executors use isolated scripts; both cross the same versioned JSON input/output
adapter and the same manifest conformance gate.

Updates are additive contract versions. A released workflow remains pinned to
the version it was published with. Removing an executor from the current
library never rewrites an existing workflow binding.
