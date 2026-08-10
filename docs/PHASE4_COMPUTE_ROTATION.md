# Phase 4 compute rotation

The project-wide CPU quota is 12 vCPUs. Central GKE uses 4, Edge GKE uses 6,
and Jenkins uses 2, so the private traffic-generator VM cannot run at the same
time as Jenkins without increasing quota.

The approved alternative keeps the quota unchanged and runs exactly one of
Jenkins or the traffic generator:

```text
CI mode:    Jenkins RUNNING     traffic generator TERMINATED
Demo mode:  Jenkins TERMINATED  traffic generator RUNNING
```

Central and Edge GKE are not stopped by this rotation.

## Commands

Inspect state:

```bash
scripts/switch_demo_compute.sh status
```

Enter live-detection demo mode:

```bash
scripts/switch_demo_compute.sh demo
```

Return to CI/CD mode:

```bash
scripts/switch_demo_compute.sh ci
```

The script preflights both explicit instance names, stops one instance and
waits for `TERMINATED` before starting the other. It never changes GKE or the
CPU quota.

## Operational boundary

- Finish or cancel Jenkins builds before entering demo mode.
- Do not merge application code while Jenkins is stopped. GitHub webhook
  delivery/retry is not treated as a durable build queue; after returning to CI
  mode, verify the expected commit was built.
- Finish the active traffic profile before returning to CI mode. Stopping the
  VM terminates its agent, Zeek capture, and any undelivered observation queue.
- Stopping a VM removes its compute charge, but its persistent disk remains
  allocated and billable.
- Terraform owns both instances. Rotation changes runtime power state only and
  must not be replaced by editing Terraform resource addresses or state.
