terraform {
  backend "gcs" {
    bucket = "fedlearning-20260729-hn-fedkube-tfstate"
    prefix = "phase3/platform"
  }
}
