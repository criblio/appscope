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

	assert.Equal(t, `{"kind":"AdmissionReview","apiVersion":"admission.k8s.io/v1","response":{"uid":"922a00bc-ba06-494e-bb48-b0928658f9ce","allowed":true,"patch":"W3sib3AiOiJyZXBsYWNlIiwicGF0aCI6Ii9tZXRhZGF0YS9sYWJlbHMiLCJ2YWx1ZSI6eyJhcHAiOiJidXN5Ym94IiwiYXBwc2NvcGUuZGV2L3Njb3BlIjoidHJ1ZSJ9fSx7Im9wIjoicmVwbGFjZSIsInBhdGgiOiIvc3BlYy9pbml0Q29udGFpbmVycyIsInZhbHVlIjpbeyJuYW1lIjoic2NvcGUiLCJpbWFnZSI6ImNyaWJsL3Njb3BlOiIsImNvbW1hbmQiOlsiL3Vzci9sb2NhbC9iaW4vc2NvcGUiLCJleGVjcmV0ZSIsIi0tbWV0cmljZGVzdCIsIiIsIi0tbWV0cmljZm9ybWF0IiwiIiwiLS1ldmVudGRlc3QiLCIiLCIvc2NvcGUiXSwicmVzb3VyY2VzIjp7fSwidm9sdW1lTW91bnRzIjpbeyJuYW1lIjoic2NvcGUiLCJtb3VudFBhdGgiOiIvc2NvcGUifV19XX0seyJvcCI6InJlcGxhY2UiLCJwYXRoIjoiL3NwZWMvY29udGFpbmVycyIsInZhbHVlIjpbeyJuYW1lIjoiYnVzeWJveCIsImltYWdlIjoiYnVzeWJveCIsImFyZ3MiOlsic2xlZXAiLCIzNjAwIl0sImVudiI6W3sibmFtZSI6IkxEX1BSRUxPQUQiLCJ2YWx1ZSI6Ii9zY29wZS9saWJzY29wZS5zbyJ9LHsibmFtZSI6IlNDT1BFX0NPTkZfUEFUSCIsInZhbHVlIjoiL3Njb3BlL3Njb3BlLnltbCJ9LHsibmFtZSI6IlNDT1BFX0VYRUNfUEFUSCIsInZhbHVlIjoiL3Njb3BlL2xkc2NvcGUifSx7Im5hbWUiOiJTQ09QRV9UQUdfbm9kZV9uYW1lIiwidmFsdWVGcm9tIjp7ImZpZWxkUmVmIjp7ImZpZWxkUGF0aCI6InNwZWMubm9kZU5hbWUifX19LHsibmFtZSI6IlNDT1BFX1RBR19uYW1lc3BhY2UiLCJ2YWx1ZUZyb20iOnsiZmllbGRSZWYiOnsiZmllbGRQYXRoIjoibWV0YWRhdGEubmFtZXNwYWNlIn19fV0sInJlc291cmNlcyI6e30sInZvbHVtZU1vdW50cyI6W3sibmFtZSI6ImRlZmF1bHQtdG9rZW4tazVsbG0iLCJyZWFkT25seSI6dHJ1ZSwibW91bnRQYXRoIjoiL3Zhci9ydW4vc2VjcmV0cy9rdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50In0seyJuYW1lIjoic2NvcGUiLCJtb3VudFBhdGgiOiIvc2NvcGUifSx7Im5hbWUiOiJzY29wZS1jb25mIiwibW91bnRQYXRoIjoiL3Njb3BlL3Njb3BlLnltbCIsInN1YlBhdGgiOiJzY29wZS55bWwifV0sInRlcm1pbmF0aW9uTWVzc2FnZVBhdGgiOiIvZGV2L3Rlcm1pbmF0aW9uLWxvZyIsInRlcm1pbmF0aW9uTWVzc2FnZVBvbGljeSI6IkZpbGUiLCJpbWFnZVB1bGxQb2xpY3kiOiJBbHdheXMifV19LHsib3AiOiJyZXBsYWNlIiwicGF0aCI6Ii9zcGVjL3ZvbHVtZXMiLCJ2YWx1ZSI6W3sibmFtZSI6ImRlZmF1bHQtdG9rZW4tazVsbG0iLCJzZWNyZXQiOnsic2VjcmV0TmFtZSI6ImRlZmF1bHQtdG9rZW4tazVsbG0ifX0seyJuYW1lIjoic2NvcGUifSx7Im5hbWUiOiJzY29wZS1jb25mIiwiY29uZmlnTWFwIjp7Im5hbWUiOiJzY29wZSJ9fV19XQ==","patchType":"JSONPatch"}}`, string(respBody))
}
