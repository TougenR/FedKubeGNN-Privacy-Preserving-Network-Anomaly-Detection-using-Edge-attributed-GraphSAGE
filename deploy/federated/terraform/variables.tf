variable "project_id" {
  type    = string
  default = "fedlearning-20260729-hn"
}

variable "project_number" {
  type    = string
  default = "421305342389"
}

variable "region" {
  type    = string
  default = "asia-southeast1"
}

variable "zone" {
  type    = string
  default = "asia-southeast1-b"
}

variable "billing_account_id" {
  type        = string
  description = "Billing account ID without the billingAccounts/ prefix."
  default     = "01CBDA-776DA4-325E05"
}

variable "budget_amount" {
  type    = number
  default = 7800000
}

variable "budget_currency_code" {
  type    = string
  default = "VND"
}

variable "central_machine_type" {
  type    = string
  default = "e2-standard-4"
}

variable "edge_machine_type" {
  type    = string
  default = "e2-custom-6-24576"
}

variable "jenkins_machine_type" {
  type    = string
  default = "e2-standard-2"
}

variable "admin_source_ranges" {
  type        = list(string)
  description = "CIDRs allowed to SSH to Jenkins. Use the operator's current /32."
  validation {
    condition     = length(var.admin_source_ranges) > 0 && !contains(var.admin_source_ranges, "0.0.0.0/0")
    error_message = "admin_source_ranges must be non-empty and may not contain 0.0.0.0/0."
  }
}

variable "labels" {
  type = map(string)
  default = {
    project     = "fedkube"
    phase       = "phase3"
    environment = "demo"
  }
}
