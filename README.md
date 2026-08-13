# SICKR workflow executors

This public repository is the release source for SICKR-maintained workflow
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

## Public-library validation phase

Organizations can import the pinned public manifest into their executor
catalog. Those organization-owned installations are deliberately marked as
public-library mirrors. They are visible and lifecycle-managed, but cannot be
attached to a workflow until the isolated runtime cutover. This validates
repository discovery, contract comparison, organization ownership, immutable
version pinning and retirement without changing production execution.

The later isolated-runtime cutover will replace that mirror transport with
repository entrypoints under the same manifest and installation identities.
Until then, the catalog must not represent these mirrors as isolated customer
scripts.
