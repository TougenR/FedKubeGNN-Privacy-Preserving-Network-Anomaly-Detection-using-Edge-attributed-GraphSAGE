{{- define "fedkube.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "fedkube.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name (include "fedkube.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "fedkube.labels" -}}
app.kubernetes.io/name: {{ include "fedkube.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: fedkube
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
{{- end }}

{{- define "fedkube.serviceAccountName" -}}
{{- default (include "fedkube.fullname" .) .Values.serviceAccount.name }}
{{- end }}

{{- define "fedkube.appImage" -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest }}
{{- end }}
