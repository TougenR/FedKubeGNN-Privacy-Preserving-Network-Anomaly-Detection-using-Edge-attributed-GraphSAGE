# Phase 3: GKE, GitOps, Six-Scenario Training, And Kibana

Phase 3 deploys the Phase 2 benchmark without changing its data or evaluation
contract. The initial platform uses one Central and one Edge GKE Standard
cluster; adding Edge clusters changes placement values, not the Flower app.

## Architecture

```text
 Developer / PR
      |
      +--> GitHub Actions: tests, lint, Trivy, Helm/Terraform/Ansible validation
      |
      `-- merge/push main --> GitHub webhook --> Jenkins VM (e2-standard-2)
                                                   |
                                                   +-- test/build/scan
                                                   +-- push hieunguyen595/fedkube-gnn:<git-sha>
                                                   `-- commit image@sha256 to environments/
                                                                       |
                                                                       v
                                                        Argo CD (Central GKE)
                                                         /               \
                                                        v                 v
 Custom VPC 10.0.0.0/8                    CENTRAL GKE                 EDGE-01 GKE
 +-----------------------------------+  subnet 10.10.0.0/20      subnet 10.20.0.0/20
 |                                   |  pods 10.40.0.0/16        pods 10.60.0.0/16
 |  GCS Training Data                |  svc  10.50.0.0/20        svc  10.70.0.0/20
 |      |                            |       |                          |
 |      +--> Central metadata PVC ---+--> SuperLink + ServerApp         |
 |      |                            |       ^                          |
 |      `----------------------------+--> Dataset Sync Job --> Edge PVC |
 |                                        /    /    /    /    /    /   |
 |                                       v    v    v    v    v    v    |
 |          10.10.0.10:443 <--- Internal NGINX <--- TLS updates/metrics |
 |                   |                    ^                             |
 |                   `-------- SuperLink |  six SuperNode/ClientApps ---'
 |                                        | 34-1 Mirai
 |                                        | 1-1  Hide-and-Seek
 |                                        | 3-1  Muhstik
 |                                        | 9-1  Linux.Hajime
 |                                        | 36-1 Okiru
 |                                        ` 39-1 IRCBot
 |
 |  ServerApp: 30 rounds x 5 local epochs, full participation
 |             FedAvg, then FedProx(mu=0.01)
 |      |
 |      `--> Global checkpoint/model --> GCS Model Artifacts
 |
 |  Central + Edge container logs --> Filebeat --> 10.10.0.11:9200
 |                                              --> Elasticsearch (30 GiB, 7d)
 |                                              --> Kibana (ClusterIP only)
 +--------------------------------------------------------------------------+

 Separate protected bucket: GCS Terraform State (versioned remote backend)
```

## Fixed deployment values

| Setting | Value |
| --- | --- |
| GCP project | `fedlearning-20260729-hn` (`421305342389`) |
| Region / zone | `asia-southeast1` / `asia-southeast1-b` |
| Central | `fedkube-central`, one `e2-standard-4` node |
| Edge | `fedkube-edge-01`, one `e2-custom-6-24576` node |
| Jenkins | `fedkube-jenkins`, one `e2-standard-2` VM |
| Training bucket | `fedlearning-20260729-hn-fedkube-training` |
| Model bucket | `fedlearning-20260729-hn-fedkube-model-artifacts` |
| Terraform state bucket | `fedlearning-20260729-hn-fedkube-tfstate` |
| Docker image | `hieunguyen595/fedkube-gnn@sha256:<digest>` |
| Budget alert | VND 7,800,000 at 50%, 90%, and 100% |

Elasticsearch and Kibana are an MVP, not an HA logging service. Kibana stays
internal; an operator reaches it after authentication with a local port-forward:

```bash
kubectl -n fedkube port-forward service/fedkube-logs-kb-http 5601:5601
```

Filebeat indices use `fedkube-*`; the `fedkube-7d` ILM policy deletes them after
seven days. Raw IPs, flow features, labels per flow, model tensors, and secrets
remain forbidden by the Phase 2 observability contract.

## Deployment authority

- GitHub Actions proves proposed code and configuration.
- Jenkins produces an application image and changes only the desired digest in
  Git. An environment-only Jenkins commit exits before build to prevent loops.
- Argo CD is the only ongoing deployer. It reconciles chart plus environment
  values and reports `Synced/Healthy` for both clusters.
- Terraform owns cloud resources, IAM, network, buckets, Secret Manager
  containers, clusters, and the Jenkins VM. Terraform never deploys workloads.
- Ansible owns Jenkins host packages, JCasC, plugins, and webhook proxy.

The one-time Argo bootstrap installs Argo CD, registers Edge-01, and submits the
operator/ApplicationSet definitions. No ongoing Jenkins stage contains `helm`,
`kubectl`, or `terraform apply`.
