data "http" "github_meta" {
  url             = "https://api.github.com/meta"
  request_headers = { Accept = "application/vnd.github+json" }
}

locals {
  # Jenkins has an external IPv4 address. GCP does not allow IPv4 and IPv6
  # source ranges in the same firewall rule, so IPv6 hook ranges cannot reach
  # this endpoint and must not be included in its IPv4 ingress rule.
  github_hook_ipv4_ranges = [
    for cidr in jsondecode(data.http.github_meta.response_body).hooks : cidr
    if !strcontains(cidr, ":")
  ]
}

resource "google_compute_address" "jenkins" {
  name       = "fedkube-jenkins"
  region     = var.region
  depends_on = [google_project_service.required["compute.googleapis.com"]]
}

resource "google_compute_firewall" "jenkins_webhook" {
  name          = "fedkube-jenkins-github-webhook"
  network       = google_compute_network.fedkube.name
  source_ranges = local.github_hook_ipv4_ranges
  target_tags   = ["fedkube-jenkins"]
  allow {
    protocol = "tcp"
    ports    = ["80"]
  }
}

resource "google_compute_firewall" "jenkins_ssh" {
  name          = "fedkube-jenkins-ssh"
  network       = google_compute_network.fedkube.name
  source_ranges = distinct(concat(var.admin_source_ranges, ["35.235.240.0/20"]))
  target_tags   = ["fedkube-jenkins"]
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_instance" "jenkins" {
  name         = "fedkube-jenkins"
  zone         = var.zone
  machine_type = var.jenkins_machine_type
  tags         = ["fedkube-jenkins"]
  labels       = var.labels

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2404-lts-amd64"
      size  = 50
      type  = "pd-balanced"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.central.id
    access_config { nat_ip = google_compute_address.jenkins.address }
  }

  service_account {
    email  = google_service_account.accounts["jenkins"].email
    scopes = ["cloud-platform"]
  }

  metadata = {
    block-project-ssh-keys = "false"
    enable-oslogin         = "TRUE"
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  lifecycle { prevent_destroy = false }
  depends_on = [google_project_service.required]
}
