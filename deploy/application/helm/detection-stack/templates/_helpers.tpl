{{- define "detection-stack.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "detection-stack.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "detection-stack.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "detection-stack.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "detection-stack.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- required "serviceAccount.name is required when create=false" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "detection-stack.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- end -}}
{{- end -}}
