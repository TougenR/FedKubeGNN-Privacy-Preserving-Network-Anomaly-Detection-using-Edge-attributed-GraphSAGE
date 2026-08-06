locals {
  clusters = {
    central = {
      name                     = "fedkube-central"
      subnetwork               = google_compute_subnetwork.central.self_link
      pods_range               = "central-pods"
      services_range           = "central-services"
      master_cidr              = "172.16.0.0/28"
      machine_type             = var.central_machine_type
      workload_service_account = google_service_account.accounts["gke_central"].email
    }
    edge_01 = {
      name                     = "fedkube-edge-01"
      subnetwork               = google_compute_subnetwork.edge.self_link
      pods_range               = "edge-01-pods"
      services_range           = "edge-01-services"
      master_cidr              = "172.16.0.16/28"
      machine_type             = var.edge_machine_type
      workload_service_account = google_service_account.accounts["gke_edge"].email
    }
  }
}

resource "google_container_cluster" "fedkube" {
  for_each = local.clusters

  name       = each.value.name
  location   = var.zone
  network    = google_compute_network.fedkube.self_link
  subnetwork = each.value.subnetwork

  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false
  enable_shielded_nodes    = true
  networking_mode          = "VPC_NATIVE"

  release_channel { channel = "REGULAR" }

  ip_allocation_policy {
    cluster_secondary_range_name  = each.value.pods_range
    services_secondary_range_name = each.value.services_range
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = each.value.master_cidr
  }

  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = concat(var.admin_source_ranges, ["${google_compute_address.nat.address}/32"])
      content {
        cidr_block   = cidr_blocks.value
        display_name = "operator-${cidr_blocks.key}"
      }
    }
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    http_load_balancing { disabled = false }
    horizontal_pod_autoscaling { disabled = false }
    gce_persistent_disk_csi_driver_config { enabled = true }
  }

  maintenance_policy {
    recurring_window {
      start_time = "2026-07-29T18:00:00Z"
      end_time   = "2026-07-29T22:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA"
    }
  }

  resource_labels = var.labels
  depends_on      = [google_project_service.required]
}

resource "google_container_node_pool" "primary" {
  for_each = local.clusters

  name       = "primary"
  location   = var.zone
  cluster    = google_container_cluster.fedkube[each.key].name
  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # The Free Trial global CPU quota is fully allocated by the steady-state
  # topology. Replace the old node before creating its upgrade replacement.
  upgrade_settings {
    max_surge       = 0
    max_unavailable = 1
    strategy        = "SURGE"
  }

  node_config {
    machine_type    = each.value.machine_type
    disk_type       = each.key == "central" ? "pd-standard" : "pd-balanced"
    disk_size_gb    = each.key == "central" ? 100 : 150
    image_type      = "COS_CONTAINERD"
    service_account = each.value.workload_service_account
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels          = merge(var.labels, { topology = each.key })
    metadata        = { disable-legacy-endpoints = "true" }
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
    workload_metadata_config { mode = "GKE_METADATA" }
  }
}
