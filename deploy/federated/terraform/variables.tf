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

variable "traffic_generator_enabled" {
  type        = bool
  description = "Create the private Phase 4 traffic-generator VM and its scoped controls."
  default     = true
}

variable "traffic_generator_machine_type" {
  type        = string
  description = "Smallest shape with enough memory for Zeek, shipper, and traffic agent."
  default     = "e2-small"
}

variable "traffic_generator_ip" {
  type        = string
  description = "Fixed RFC1918 address in the existing Central subnet."
  default     = "10.10.0.20"
  validation {
    condition     = can(cidrhost("${var.traffic_generator_ip}/32", 0)) && can(regex("^10\\.10\\.([0-9]|1[0-5])\\.([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$", var.traffic_generator_ip))
    error_message = "traffic_generator_ip must be an IPv4 address in the 10.10.0.0/20 Central subnet."
  }
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
