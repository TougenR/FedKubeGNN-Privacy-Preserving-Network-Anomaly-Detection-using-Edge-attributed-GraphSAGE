locals {
  service_accounts = merge({
    gke_central      = "fedkube-gke-central"
    gke_edge         = "fedkube-gke-edge-01"
    workload_central = "fedkube-central"
    workload_edge    = "fedkube-edge-01"
    external_central = "external-secrets-central"
    external_edge    = "external-secrets-edge-01"
    jenkins          = "fedkube-jenkins"
    }, var.traffic_generator_enabled ? {
    traffic_generator = "fedkube-traffic-generator"
  } : {})
}

locals {
  traffic_generator_secret_access = var.traffic_generator_enabled ? {
    "agent-external-central" = {
      account = "external_central"
      secret  = "fedkube-traffic-agent-token"
    }
    "agent-generator" = {
      account = "traffic_generator"
      secret  = "fedkube-traffic-agent-token"
    }
    "observation-external-central" = {
      account = "external_central"
      secret  = "fedkube-traffic-observation-token"
    }
    "observation-generator" = {
      account = "traffic_generator"
      secret  = "fedkube-traffic-observation-token"
    }
  } : {}
}

resource "google_secret_manager_secret_iam_member" "traffic_generator_token" {
  for_each = local.traffic_generator_secret_access

  project   = var.project_id
  secret_id = google_secret_manager_secret.runtime[each.value.secret].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.accounts[each.value.account].email}"
}

resource "google_service_account" "accounts" {
  for_each     = local.service_accounts
  account_id   = each.value
  display_name = "FedKube ${each.key}"
  depends_on   = [google_project_service.required]
}

locals {
  gke_central = google_service_account.accounts["gke_central"]
  gke_edge    = google_service_account.accounts["gke_edge"]
}

resource "google_project_iam_member" "gke_node_roles" {
  for_each = {
    "central-logging"     = { account = local.gke_central.email, role = "roles/logging.logWriter" }
    "central-monitoring"  = { account = local.gke_central.email, role = "roles/monitoring.metricWriter" }
    "central-metadata"    = { account = local.gke_central.email, role = "roles/stackdriver.resourceMetadata.writer" }
    "central-gke-default" = { account = local.gke_central.email, role = "roles/container.defaultNodeServiceAccount" }
    "edge-logging"        = { account = local.gke_edge.email, role = "roles/logging.logWriter" }
    "edge-monitoring"     = { account = local.gke_edge.email, role = "roles/monitoring.metricWriter" }
    "edge-metadata"       = { account = local.gke_edge.email, role = "roles/stackdriver.resourceMetadata.writer" }
    "edge-gke-default"    = { account = local.gke_edge.email, role = "roles/container.defaultNodeServiceAccount" }
  }
  project = var.project_id
  role    = each.value.role
  member  = "serviceAccount:${each.value.account}"
}

resource "google_storage_bucket_iam_member" "training_readers" {
  for_each = toset(["workload_central", "workload_edge"])
  bucket   = google_storage_bucket.training.name
  role     = "roles/storage.objectViewer"
  member   = "serviceAccount:${google_service_account.accounts[each.value].email}"
}

resource "google_storage_bucket_iam_member" "model_writer" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.accounts["workload_central"].email}"
}

locals {
  workload_identity_bindings = {
    central = {
      gsa       = "workload_central"
      namespace = "fedkube"
      ksa       = "fedkube-central-workload"
    }
    edge = {
      gsa       = "workload_edge"
      namespace = "fedkube"
      ksa       = "fedkube-edge-01-workload"
    }
    external_central = {
      gsa       = "external_central"
      namespace = "external-secrets"
      ksa       = "external-secrets-central"
    }
    external_edge = {
      gsa       = "external_edge"
      namespace = "external-secrets"
      ksa       = "external-secrets-edge-01"
    }
  }
}

resource "google_service_account_iam_member" "workload_identity" {
  for_each           = local.workload_identity_bindings
  service_account_id = google_service_account.accounts[each.value.gsa].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${each.value.namespace}/${each.value.ksa}]"
  depends_on         = [google_container_cluster.fedkube]
}

resource "google_secret_manager_secret_iam_member" "external_secret_access" {
  for_each = {
    central_ca      = { secret = "fedkube-flower-ca", account = "external_central" }
    central_cert    = { secret = "fedkube-flower-cert", account = "external_central" }
    central_key     = { secret = "fedkube-flower-key", account = "external_central" }
    central_elastic = { secret = "fedkube-elastic-password", account = "external_central" }
    edge_ca         = { secret = "fedkube-flower-ca", account = "external_edge" }
    edge_elastic    = { secret = "fedkube-elastic-password", account = "external_edge" }
  }
  project   = var.project_id
  secret_id = google_secret_manager_secret.runtime[each.value.secret].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.accounts[each.value.account].email}"
}
