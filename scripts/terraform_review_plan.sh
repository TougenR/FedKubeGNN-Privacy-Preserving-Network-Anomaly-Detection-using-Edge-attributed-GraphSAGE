#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
source_dir="${repo_root}/infra/terraform"
review_dir="$(mktemp -d)"
output_dir="${repo_root}/artifacts/phase3/terraform"
admin_cidr="${1:-}"

if [[ -z "${admin_cidr}" ]]; then
  admin_cidr="$(curl --fail --silent --show-error https://api.ipify.org)/32"
fi
if [[ ! "${admin_cidr}" =~ ^[0-9a-fA-F:.]+/[0-9]{1,3}$ ]]; then
  echo "Usage: $0 [operator CIDR, for example 203.0.113.10/32]" >&2
  exit 2
fi

for tf_file in "${source_dir}"/*.tf; do
  if [[ "$(basename "${tf_file}")" != "backend.tf" ]]; then
    cp "${tf_file}" "${review_dir}/"
  fi
done
cp "${source_dir}/.terraform.lock.hcl" "${review_dir}/"

terraform -chdir="${review_dir}" init -backend=false -input=false >/dev/null
terraform -chdir="${review_dir}" plan \
  -refresh=false -lock=false -input=false \
  -var "admin_source_ranges=[\"${admin_cidr}\"]" \
  -out=phase3.tfplan

mkdir -p "${output_dir}"
terraform -chdir="${review_dir}" show -no-color phase3.tfplan \
  >"${output_dir}/phase3-plan.txt"
echo "Review-only plan: ${output_dir}/phase3-plan.txt"
echo "Temporary binary plan (not for apply): ${review_dir}/phase3.tfplan"
