resource "google_storage_bucket" "training" {
  name                        = "${var.project_id}-fedkube-training"
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  versioning { enabled = true }
  depends_on = [google_project_service.required]
}

resource "google_storage_bucket" "models" {
  name                        = "${var.project_id}-fedkube-model-artifacts"
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  versioning { enabled = true }
  depends_on = [google_project_service.required]
}

locals {
  secret_ids = toset(concat([
    "fedkube-flower-ca",
    "fedkube-flower-cert",
    "fedkube-flower-key",
    "fedkube-elastic-password",
    ], var.traffic_generator_enabled ? [
    "fedkube-traffic-agent-token",
    "fedkube-traffic-observation-token",
  ] : []))
}

resource "google_secret_manager_secret" "runtime" {
  for_each  = local.secret_ids
  secret_id = each.value
  replication {
    auto {}
  }
  labels     = var.labels
  depends_on = [google_project_service.required]
}
