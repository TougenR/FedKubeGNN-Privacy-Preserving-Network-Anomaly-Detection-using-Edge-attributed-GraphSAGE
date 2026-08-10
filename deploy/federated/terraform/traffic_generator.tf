locals {
  traffic_generator_labels = merge(var.labels, {
    phase     = "phase4"
    component = "traffic-generator"
  })
}

resource "google_compute_address" "traffic_generator" {
  count        = var.traffic_generator_enabled ? 1 : 0
  name         = "fedkube-traffic-generator"
  region       = var.region
  address_type = "INTERNAL"
  subnetwork   = google_compute_subnetwork.central.id
  address      = var.traffic_generator_ip
}

# These addresses are intentionally reserved without instances. Traffic leaves
# the generator through the VPC router, receives no TCP response, and is
# observed locally by Zeek without targeting an external or third-party host.
resource "google_compute_address" "traffic_blackhole" {
  for_each = var.traffic_generator_enabled ? {
    "01" = "10.20.0.20"
    "02" = "10.20.0.21"
    "03" = "10.20.0.22"
  } : {}

  name         = "fedkube-traffic-blackhole-${each.key}"
  region       = var.region
  address_type = "INTERNAL"
  subnetwork   = google_compute_subnetwork.edge.id
  address      = each.value
}

resource "google_compute_firewall" "traffic_generator_iap_ssh" {
  count         = var.traffic_generator_enabled ? 1 : 0
  name          = "fedkube-traffic-generator-iap-ssh"
  network       = google_compute_network.fedkube.name
  direction     = "INGRESS"
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["fedkube-traffic-generator"]
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_firewall" "traffic_generator_control" {
  count         = var.traffic_generator_enabled ? 1 : 0
  name          = "fedkube-traffic-generator-control"
  network       = google_compute_network.fedkube.name
  direction     = "INGRESS"
  source_ranges = ["10.40.0.0/16"]
  target_tags   = ["fedkube-traffic-generator"]
  allow {
    protocol = "tcp"
    ports    = ["8091"]
  }
}

resource "google_compute_firewall" "traffic_generator_private_egress" {
  count     = var.traffic_generator_enabled ? 1 : 0
  name      = "fedkube-traffic-generator-private-egress"
  network   = google_compute_network.fedkube.name
  direction = "EGRESS"
  priority  = 900
  destination_ranges = [
    "10.10.0.5/32",
    "10.20.0.20/32",
    "10.20.0.21/32",
    "10.20.0.22/32",
    "10.20.0.30/32",
    "10.20.0.31/32",
  ]
  target_tags = ["fedkube-traffic-generator"]
  allow {
    protocol = "tcp"
  }
  allow {
    protocol = "udp"
  }
  allow {
    protocol = "icmp"
  }
}

resource "google_compute_firewall" "traffic_generator_https_egress" {
  count              = var.traffic_generator_enabled ? 1 : 0
  name               = "fedkube-traffic-generator-https-egress"
  network            = google_compute_network.fedkube.name
  direction          = "EGRESS"
  priority           = 900
  destination_ranges = ["0.0.0.0/0"]
  target_tags        = ["fedkube-traffic-generator"]
  allow {
    protocol = "tcp"
    ports    = ["443"]
  }
}

# Keep access to the GCE metadata/DNS endpoint explicit before the catch-all
# egress deny. The VM uses metadata identity to read its Secret Manager token.
resource "google_compute_firewall" "traffic_generator_metadata_egress" {
  count              = var.traffic_generator_enabled ? 1 : 0
  name               = "fedkube-traffic-generator-metadata-egress"
  network            = google_compute_network.fedkube.name
  direction          = "EGRESS"
  priority           = 890
  destination_ranges = ["169.254.169.254/32"]
  target_tags        = ["fedkube-traffic-generator"]
  allow {
    protocol = "tcp"
    ports    = ["53", "80", "443"]
  }
  allow {
    protocol = "udp"
    ports    = ["53"]
  }
}

resource "google_compute_firewall" "traffic_generator_deny_other_egress" {
  count              = var.traffic_generator_enabled ? 1 : 0
  name               = "fedkube-traffic-generator-deny-other-egress"
  network            = google_compute_network.fedkube.name
  direction          = "EGRESS"
  priority           = 1000
  destination_ranges = ["0.0.0.0/0"]
  target_tags        = ["fedkube-traffic-generator"]
  deny {
    protocol = "all"
  }
}

resource "google_compute_instance" "traffic_generator" {
  count        = var.traffic_generator_enabled ? 1 : 0
  name         = "fedkube-traffic-generator"
  zone         = var.zone
  machine_type = var.traffic_generator_machine_type
  tags         = ["fedkube-traffic-generator"]
  labels       = local.traffic_generator_labels

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2404-lts-amd64"
      size  = 20
      type  = "pd-standard"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.central.id
    network_ip = google_compute_address.traffic_generator[0].address
  }

  service_account {
    email  = google_service_account.accounts["traffic_generator"].email
    scopes = ["cloud-platform"]
  }

  metadata = {
    block-project-ssh-keys = "true"
    enable-oslogin         = "TRUE"
    serial-port-enable     = "FALSE"
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  can_ip_forward            = false
  allow_stopping_for_update = true
  deletion_protection       = false
  lifecycle {
    prevent_destroy = false
  }

  depends_on = [
    google_compute_firewall.traffic_generator_control,
    google_compute_firewall.traffic_generator_iap_ssh,
    google_secret_manager_secret_iam_member.traffic_generator_token,
  ]
}
