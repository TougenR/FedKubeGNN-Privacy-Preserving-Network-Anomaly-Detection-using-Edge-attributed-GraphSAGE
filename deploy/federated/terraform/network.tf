resource "google_compute_network" "fedkube" {
  name                    = "fedkube-vpc"
  auto_create_subnetworks = false
  routing_mode            = "GLOBAL"
  depends_on              = [google_project_service.required]
}

resource "google_compute_subnetwork" "central" {
  name                     = "fedkube-central"
  region                   = var.region
  network                  = google_compute_network.fedkube.id
  ip_cidr_range            = "10.10.0.0/20"
  private_ip_google_access = true
  secondary_ip_range {
    range_name    = "central-pods"
    ip_cidr_range = "10.40.0.0/16"
  }
  secondary_ip_range {
    range_name    = "central-services"
    ip_cidr_range = "10.50.0.0/20"
  }
}

resource "google_compute_subnetwork" "edge" {
  name                     = "fedkube-edge-01"
  region                   = var.region
  network                  = google_compute_network.fedkube.id
  ip_cidr_range            = "10.20.0.0/20"
  private_ip_google_access = true
  secondary_ip_range {
    range_name    = "edge-01-pods"
    ip_cidr_range = "10.60.0.0/16"
  }
  secondary_ip_range {
    range_name    = "edge-01-services"
    ip_cidr_range = "10.70.0.0/20"
  }
}

resource "google_compute_router" "fedkube" {
  name    = "fedkube-router"
  region  = var.region
  network = google_compute_network.fedkube.id
}

resource "google_compute_address" "nat" {
  name       = "fedkube-nat"
  region     = var.region
  depends_on = [google_project_service.required["compute.googleapis.com"]]
}

resource "google_compute_router_nat" "fedkube" {
  name                                = "fedkube-nat"
  router                              = google_compute_router.fedkube.name
  region                              = var.region
  nat_ip_allocate_option              = "MANUAL_ONLY"
  nat_ips                             = [google_compute_address.nat.self_link]
  source_subnetwork_ip_ranges_to_nat  = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  enable_endpoint_independent_mapping = true
}

resource "google_compute_address" "flower_internal" {
  name         = "fedkube-flower-internal"
  region       = var.region
  address_type = "INTERNAL"
  subnetwork   = google_compute_subnetwork.central.id
  address      = "10.10.0.10"
}

resource "google_compute_address" "elastic_internal" {
  name         = "fedkube-elastic-internal"
  region       = var.region
  address_type = "INTERNAL"
  subnetwork   = google_compute_subnetwork.central.id
  address      = "10.10.0.11"
}
