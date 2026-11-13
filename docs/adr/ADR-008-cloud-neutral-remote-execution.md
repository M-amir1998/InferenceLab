# ADR-008 — Cloud-neutral remote execution

**Status:** accepted · **Date:** 2026-11-13

## Context

GPU capacity will be rented from whichever provider has availability and a good price at
the time. Provider SDKs, stored credentials and automatic provisioning would add lock-in
and the risk of accidental spend.

## Decision

* The only abstraction is a `RemoteHost` reached over **SSH** with the user's existing SSH
  setup; configs hold at most a key *path*, never key material.
* InferenceLab never creates, resizes or terminates machines. Campaigns stop and remove
  their containers and temporary configs, and leave the machine running.
* Paid execution requires `--execute`; `remote plan` prints the exact command sequence;
  every run has a duration cap; cost estimates are upper bounds from recorded prices.
* Engines run from pinned images with ports published on `127.0.0.1`; secrets come from an
  env file the user creates on the host.
* Provider-specific helpers may be added later as optional documentation or plugins.

## Consequences

* Works with any provider (or an on-prem box) that offers Docker and the NVIDIA container
  runtime; no accidental spend.
* The user provisions and terminates machines manually.
