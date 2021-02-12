package k8s

var webhook string = `---
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: {{ .App }}
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
    NamespaceSelector:
      matchLabels:
        scope: enabled
`

var csr string = `---
apiVersion: batch/v1
kind: Job
metadata:
  name: webhook-cert-setup
spec:
  template:
    spec:
      serviceAccountName: webhook-cert-sa
      containers:
      - name: webhook-cert-setup
        # This is a minimal kubectl image based on Alpine Linux that signs certificates using the k8s extension api server
        image: quay.io/didil/k8s-webhook-cert-manager:0.13.19-1-a
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
    namespace: default
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: webhook-cert-sa
`

var webhookDeployment string = `---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .App }}
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
    spec:
      containers:
        - name: {{ .App }}
          image: criblio/scope:{{ .Version }}
          imagePullPolicy: IfNotPresent
          volumeMounts:
            - name: certs
              mountPath: /etc/certs
              readOnly: true
          ports:
            - containerPort: 4443
              protocol: TCP
      volumes:
        - name: certs
          secret:
            secretName: {{ .App }}-secret 
---
apiVersion: v1
kind: Service
metadata:
  name: {{ .App }}
spec:
  type: ClusterIP
  ports:
    - name: 4443-tcp
      protocol: TCP
      port: 443
      targetPort: 4443
  selector:
    app: {{ .App }} 
`
