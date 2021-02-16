package k8s

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHandleMutate(t *testing.T) {
	app := App{&Options{}}

	mux := http.NewServeMux()
	mux.HandleFunc("/mutate", app.HandleMutate)
	s := httptest.NewServer(mux)
	defer s.Close()

	admissionReviewJSON := `{
		"kind": "AdmissionReview",
		"apiVersion": "admission.k8s.io/v1",
		"request": {
		  "uid": "922a00bc-ba06-494e-bb48-b0928658f9ce",
		  "kind": {
			"group": "",
			"version": "v1",
			"kind": "Pod"
		  },
		  "resource": {
			"group": "",
			"version": "v1",
			"resource": "pods"
		  },
		  "requestKind": {
			"group": "",
			"version": "v1",
			"kind": "Pod"
		  },
		  "requestResource": {
			"group": "",
			"version": "v1",
			"resource": "pods"
		  },
		  "name": "busybox",
		  "namespace": "default",
		  "operation": "CREATE",
		  "userInfo": {
			"username": "kubernetes-admin",
			"groups": [
			  "system:masters",
			  "system:authenticated"
			]
		  },
		  "object": {
			"kind": "Pod",
			"apiVersion": "v1",
			"metadata": {
			  "name": "busybox",
			  "namespace": "default",
			  "labels": {
				"app": "busybox"
			  },
			  "annotations": {
			  },
			  "managedFields": [
				{
				  "manager": "kubectl",
				  "operation": "Update",
				  "apiVersion": "v1",
				  "time": "2020-10-01T15:39:15Z",
				  "fieldsType": "FieldsV1",
				  "fieldsV1": {
					"f:metadata": {
					  "f:annotations": {
						".": {},
						"f:kubectl.kubernetes.io/last-applied-configuration": {}
					  },
					  "f:labels": {
						".": {},
						"f:app": {}
					  }
					},
					"f:spec": {
					  "f:containers": {
						"k:{\"name\":\"busybox\"}": {
						  ".": {},
						  "f:args": {},
						  "f:image": {},
						  "f:imagePullPolicy": {},
						  "f:name": {},
						  "f:resources": {},
						  "f:terminationMessagePath": {},
						  "f:terminationMessagePolicy": {}
						}
					  },
					  "f:dnsPolicy": {},
					  "f:enableServiceLinks": {},
					  "f:restartPolicy": {},
					  "f:schedulerName": {},
					  "f:securityContext": {},
					  "f:terminationGracePeriodSeconds": {}
					}
				  }
				}
			  ]
			},
			"spec": {
			  "volumes": [
				{
				  "name": "default-token-k5llm",
				  "secret": {
					"secretName": "default-token-k5llm"
				  }
				}
			  ],
			  "containers": [
				{
				  "name": "busybox",
				  "image": "busybox",
				  "args": [
					"sleep",
					"3600"
				  ],
				  "resources": {},
				  "volumeMounts": [
					{
					  "name": "default-token-k5llm",
					  "readOnly": true,
					  "mountPath": "/var/run/secrets/kubernetes.io/serviceaccount"
					}
				  ],
				  "terminationMessagePath": "/dev/termination-log",
				  "terminationMessagePolicy": "File",
				  "imagePullPolicy": "Always"
				}
			  ],
			  "restartPolicy": "Always",
			  "terminationGracePeriodSeconds": 5,
			  "dnsPolicy": "ClusterFirst",
			  "serviceAccountName": "default",
			  "serviceAccount": "default",
			  "securityContext": {},
			  "schedulerName": "default-scheduler",
			  "tolerations": [
				{
				  "key": "node.kubernetes.io/not-ready",
				  "operator": "Exists",
				  "effect": "NoExecute",
				  "tolerationSeconds": 300
				},
				{
				  "key": "node.kubernetes.io/unreachable",
				  "operator": "Exists",
				  "effect": "NoExecute",
				  "tolerationSeconds": 300
				}
			  ],
			  "priority": 0,
			  "enableServiceLinks": true,
			  "preemptionPolicy": "PreemptLowerPriority"
			},
			"status": {}
		  },
		  "dryRun": false,
		  "options": {
			"kind": "CreateOptions",
			"apiVersion": "meta.k8s.io/v1"
		  }
		}
	  }`
	var b bytes.Buffer

	_, err := b.WriteString(admissionReviewJSON)
	assert.NoError(t, err)

	req, err := http.NewRequest(http.MethodPost, s.URL+"/mutate", &b)
	assert.NoError(t, err)

	req.Header.Add("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	assert.NoError(t, err)

	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, "application/json", resp.Header.Get("Content-Type"))

	respBody, err := ioutil.ReadAll(resp.Body)
	assert.NoError(t, err)

	assert.Equal(t, `{"kind":"AdmissionReview","apiVersion":"admission.k8s.io/v1","response":{"uid":"922a00bc-ba06-494e-bb48-b0928658f9ce","allowed":true,"patch":"W3sib3AiOiJyZXBsYWNlIiwicGF0aCI6Ii9tZXRhZGF0YS9sYWJlbHMiLCJ2YWx1ZSI6eyJhcHAiOiJidXN5Ym94IiwiaW8uYXBwc2NvcGUvc2NvcGVkIjoidHJ1ZSJ9fSx7Im9wIjoicmVwbGFjZSIsInBhdGgiOiIvc3BlYy9pbml0Q29udGFpbmVycyIsInZhbHVlIjpbeyJuYW1lIjoic2NvcGUiLCJpbWFnZSI6ImNyaWJsL3Njb3BlOiIsImNvbW1hbmQiOlsiL3Vzci9sb2NhbC9iaW4vc2NvcGUiLCJleGNyZXRlIiwiLS1tZXRyaWNkZXN0IiwiIiwiLS1tZXRyaWNmb3JtYXQiLCIiLCItLWV2ZW50ZGVzdCIsIiIsIi9zY29wZSJdLCJyZXNvdXJjZXMiOnt9LCJ2b2x1bWVNb3VudHMiOlt7Im5hbWUiOiJzY29wZSIsIm1vdW50UGF0aCI6Ii9zY29wZSJ9XX1dfSx7Im9wIjoicmVwbGFjZSIsInBhdGgiOiIvc3BlYy9jb250YWluZXJzIiwidmFsdWUiOlt7Im5hbWUiOiJidXN5Ym94IiwiaW1hZ2UiOiJidXN5Ym94IiwiYXJncyI6WyJzbGVlcCIsIjM2MDAiXSwiZW52IjpbeyJuYW1lIjoiTERfUFJFTE9BRCIsInZhbHVlIjoiL3Njb3BlL2xpYnNjb3BlLnNvIn0seyJuYW1lIjoiU0NPUEVfQ09ORl9QQVRIIiwidmFsdWUiOiIvc2NvcGUvc2NvcGUueW1sIn0seyJuYW1lIjoiU0NPUEVfRVhFQ19QQVRIIiwidmFsdWUiOiIvc2NvcGUvbGRzY29wZSJ9LHsibmFtZSI6IlNDT1BFX1RBR19ub2RlX25hbWUiLCJ2YWx1ZUZyb20iOnsiZmllbGRSZWYiOnsiZmllbGRQYXRoIjoic3BlYy5ub2RlTmFtZSJ9fX0seyJuYW1lIjoiU0NPUEVfVEFHX25hbWVzcGFjZSIsInZhbHVlRnJvbSI6eyJmaWVsZFJlZiI6eyJmaWVsZFBhdGgiOiJtZXRhZGF0YS5uYW1lc3BhY2UifX19XSwicmVzb3VyY2VzIjp7fSwidm9sdW1lTW91bnRzIjpbeyJuYW1lIjoiZGVmYXVsdC10b2tlbi1rNWxsbSIsInJlYWRPbmx5Ijp0cnVlLCJtb3VudFBhdGgiOiIvdmFyL3J1bi9zZWNyZXRzL2t1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQifSx7Im5hbWUiOiJzY29wZSIsIm1vdW50UGF0aCI6Ii9zY29wZSJ9XSwidGVybWluYXRpb25NZXNzYWdlUGF0aCI6Ii9kZXYvdGVybWluYXRpb24tbG9nIiwidGVybWluYXRpb25NZXNzYWdlUG9saWN5IjoiRmlsZSIsImltYWdlUHVsbFBvbGljeSI6IkFsd2F5cyJ9XX0seyJvcCI6InJlcGxhY2UiLCJwYXRoIjoiL3NwZWMvdm9sdW1lcyIsInZhbHVlIjpbeyJuYW1lIjoiZGVmYXVsdC10b2tlbi1rNWxsbSIsInNlY3JldCI6eyJzZWNyZXROYW1lIjoiZGVmYXVsdC10b2tlbi1rNWxsbSJ9fSx7Im5hbWUiOiJzY29wZSJ9XX1d","patchType":"JSONPatch"}}`, string(respBody))
}
