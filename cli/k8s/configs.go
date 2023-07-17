package k8s

var webhook string = `---
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: {{ .App }}.{{ .Namespace }}.appscope.io
  labels:
    app: {{ .App }}
webhooks:
  - name: {{ .App }}.{{ .Namespace }}.appscope.io
    sideEffects: None
    admissionReviewVersions: ["v1", "v1beta1"]
    matchPolicy: Equivalent
    failurePolicy: Fail
    clientConfig:
      service:
        name: {{ .App }}
        namespace: {{ .Namespace }}
        path: "/mutate"
    rules:
      - operations: [ "CREATE" ]
        apiGroups: [""]
        apiVersions: ["v1"]
        resources: ["pods"]
        scope: "*"
    namespaceSelector:
      matchLabels:
        scope: enabled
`

var csr string = `---
apiVersion: batch/v1
kind: Job
metadata:
  name: webhook-cert-setup
  namespace: {{ .Namespace }}
spec:
  template:
    spec:
      serviceAccountName: webhook-cert-sa
      containers:
      - name: webhook-cert-setup
        # This is a minimal kubectl image based on Alpine Linux that signs certificates using the k8s extension api server
        image: cribl/k8s-webhook-cert-manager:1.0.1
        command: ["./generate_certificate.sh"]
        args:
          - "--service"
          - "{{ .App }}"
          - "--webhook"
          - "{{ .App }}.{{ .Namespace }}.appscope.io"
          - "--secret"
          - "{{ .App }}-secret"
          - "--namespace"
          - "{{ .Namespace }}"
          - "--signer-name"
          - "{{ .SignerName }}"
      restartPolicy: OnFailure
  backoffLimit: 3
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: webhook-cert-cluster-role
rules:
  - apiGroups: ["admissionregistration.k8s.io"]
    resources: ["mutatingwebhookconfigurations"]
    verbs: ["get", "create", "patch"]
  - apiGroups: ["certificates.k8s.io"]
    resources: ["certificatesigningrequests"]
    verbs: ["create", "get", "delete"]
  - apiGroups: ["certificates.k8s.io"]
    resources: ["certificatesigningrequests/approval"]
    verbs: ["update"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["create", "get", "patch"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get"]
  - apiGroups: ["certificates.k8s.io"]
    resources: ["signers"]
    resourceNames: ["{{ .SignerName }}"]
    verbs: ["approve"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: webhook-cert-cluster-role-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: webhook-cert-cluster-role
subjects:
  - kind: ServiceAccount
    name: webhook-cert-sa
    namespace: {{ .Namespace }}
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: webhook-cert-sa
  namespace: {{ .Namespace }}
`

var webhookDeployment string = `---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: scope-cluster-role
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    resourceNames: ["scope"]
    verbs: ["get", "patch", "put", "update"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["create"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: scope-cluster-role-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: scope-cluster-role
subjects:
  - kind: ServiceAccount
    name: scope-cert-sa
    namespace: {{ .Namespace }}
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: scope-cert-sa
  namespace: {{ .Namespace }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .App }}
  namespace: {{ .Namespace }}
  labels:
    app: {{ .App }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {{ .App }}
  strategy: {}
  template:
    metadata:
      creationTimestamp: null
      labels:
        app: {{ .App }}
{{- if not .ExporterDisable }}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "{{ .ExporterPromPort }}"
{{- end }}
    spec:
      serviceAccountName: scope-cert-sa
      containers:
        - name: {{ .App }}
          image: cribl/scope:{{ .Version }}
          command: ["/bin/bash"]
          args:
          - "-c"
          - "/usr/local/bin/scope k8s --server{{if .Debug}} --debug{{end}}{{if gt (len .MetricDest) 0}} --metricdest {{ .MetricDest }}{{end}}{{if and (gt (len .MetricFormat) 0) (eq (len .CriblDest) 0)}} --metricformat {{ .MetricFormat }} --metricprefix {{ .MetricPrefix }}{{end}}{{if gt (len .EventDest) 0}} --eventdest {{ .EventDest }}{{end}}{{if gt (len .CriblDest) 0}} --cribldest {{ .CriblDest}}{{end}} || sleep 1000"
          imagePullPolicy: IfNotPresent
          volumeMounts:
            - name: certs
              mountPath: /etc/certs
              readOnly: true
          ports:
            - containerPort: {{ .Port }}
              protocol: TCP
{{- if not .ExporterDisable }}
        - name: {{ .App }}-stats-exporter
          image: prom/statsd-exporter:v0.24.0
          args:
          {{- if eq .ExporterStatsDProtocol "tcp" }}
          - --statsd.listen-tcp=:{{ .ExporterStatsDPort }}
          {{- else if eq .ExporterStatsDProtocol "udp" }}
          - --statsd.listen-udp=:{{ .ExporterStatsDPort }}
          {{- end }}
          - --web.listen-address=:{{ .ExporterPromPort }}
          - --statsd.mapping-config=/tmp/mapping.conf
          imagePullPolicy: Always
          volumeMounts:
            - name: {{ .App }}-stats-exporter-mapping-config-file
              mountPath: /tmp
          ports:
            {{- if eq .ExporterStatsDProtocol "tcp" }}
            - containerPort: {{ .ExporterStatsDPort }}
              protocol: TCP
            {{- else if eq .ExporterStatsDProtocol "udp" }}
            - containerPort: {{ .ExporterStatsDPort }}
              protocol: UDP
            {{- end }}
            - containerPort: {{ .ExporterPromPort }}
              protocol: TCP
{{- end }}
      volumes:
        - name: certs
          secret:
            secretName: {{ .App }}-secret
{{- if not .ExporterDisable }}
        - name: {{ .App }}-stats-exporter-mapping-config-file
          configMap:
            name: {{ .App }}-stats-exporter-mapping-config
---
apiVersion: v1
kind: Service
metadata:
  name: {{ .App }}-stats-exporter
  namespace: {{ .Namespace }}
spec:
  type: ClusterIP
  ports:
    {{- if eq .ExporterStatsDProtocol "tcp" }}
    - name: {{ .ExporterStatsDPort }}-stats-exporter-statsd-listen
      protocol: TCP
      port: {{ .ExporterStatsDPort }}
      targetPort: {{ .ExporterStatsDPort }}
    {{- else if eq .ExporterStatsDProtocol "udp" }}
    - name: {{ .ExporterStatsDPort }}-stats-exporter-statsd-listen
      protocol: UDP
      port: {{ .ExporterStatsDPort }}
      targetPort: {{ .ExporterStatsDPort }}
    {{- end }}
    - name: {{ .ExporterPromPort }}-stats-exporter-prometheus-http
      protocol: TCP
      port: {{ .ExporterPromPort }}
      targetPort: {{ .ExporterPromPort }}
  selector:
    app: {{ .App }}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .App }}-stats-exporter-mapping-config
  namespace: {{ .Namespace }}
data:
  mapping.conf: |-
    mappings:
    - match: "http.duration.server"
      help: "Total duration of http response"
      observer_type: histogram
      histogram_options:
        buckets: [ 0.01, 0.025, 0.05, 0.1 ]
        native_histogram_bucket_factor: 1.1
        native_histogram_max_buckets: 256
      name: "appscope_http_duration_server"
{{- end }}
---
apiVersion: v1
kind: Service
metadata:
  name: {{ .App }}
  namespace: {{ .Namespace }}
spec:
  type: ClusterIP
  ports:
    - name: {{ .Port }}-tcp
      protocol: TCP
      port: 443
      targetPort: {{ .Port }}
  selector:
    app: {{ .App }}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .App }}
  namespace: {{ .Namespace }}
data:
  scope.yml: |
{{ .ScopeConfigYaml | toString | indent 4 }}
`

// - "k8s"
// - "--server"
// - "--metricdest"
// - "{{ .MetricDest }}"
// - "--metricformat"
// - "{{ .MetricFormat }}"
// - "--metricprefix"
// - "{{ .MetricPrefix }}"
// - "--eventdest"
// - "{{ .EventDest }}"
