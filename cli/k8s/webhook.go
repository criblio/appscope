package k8s

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"

	"github.com/criblio/scope/internal"
	"github.com/rs/zerolog/log"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// App contains configuration for a webhook server
type App struct {
	*Options
}

// HandleMutate handles the mutate endpoint
func (app *App) HandleMutate(w http.ResponseWriter, r *http.Request) {
	log.Debug().Str("method", r.Method).Str("proto", r.Proto).Str("remoteaddr", r.RemoteAddr).Str("uri", r.RequestURI).Int("length", int(r.ContentLength)).Msg("request received")
	admissionReview := &admissionv1.AdmissionReview{}

	err := json.NewDecoder(r.Body).Decode(admissionReview)
	if err != nil {
		app.HandleError(w, r, fmt.Errorf("invalid JSON input"))
		return
	}

	pod := &corev1.Pod{}
	if err := json.Unmarshal(admissionReview.Request.Object.Raw, pod); err != nil {
		app.HandleError(w, r, fmt.Errorf("unmarshal to pod: %v", err))
		return
	}

	shouldModify := true

	if _, ok := pod.ObjectMeta.Annotations["appscope.dev/disable"]; ok {
		shouldModify = false
	}

	ver := strings.Split(internal.GetVersion(), "-")

	patch := []JSONPatchEntry{}
	if shouldModify {
		log.Debug().Interface("pod", pod).Msgf("modifying pod")
		// creates the in-cluster config
		config, err := rest.InClusterConfig()
		if err != nil {
			// Only throw an error if we're not in a test
			// OR if we are in a test, and the error is NOT ErrNotInACluster
			if !strings.HasSuffix(os.Args[0], ".test") || err != rest.ErrNotInCluster {
				app.HandleError(w, r, err)
				return
			}
		} else {
			// creates the clientset
			clientset, err := kubernetes.NewForConfig(config)
			if err != nil {
				app.HandleError(w, r, err)
				return
			}

			// Get current namespace from serviceaccount
			namespace, err := ioutil.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
			if err != nil {
				app.HandleError(w, r, err)
				return
			}

			// Get reference configmap in scope's namespace
			cm, err := clientset.CoreV1().ConfigMaps(string(namespace)).Get(context.TODO(), "scope", metav1.GetOptions{})
			if err != nil {
				app.HandleError(w, r, err)
				return
			}

			// If the scope configmap does exists in our namespace, create it from template in scope namespace
			if cmlocal, err := clientset.CoreV1().ConfigMaps(admissionReview.Request.Namespace).Get(context.TODO(), "scope", metav1.GetOptions{}); errors.IsNotFound(err) {
				log.Debug().Interface("cm", cm).Str("namespace", admissionReview.Request.Namespace).Msgf("creating configmap")
				cm.SetResourceVersion("")
				cm.SetNamespace(admissionReview.Request.Namespace)
				_, err := clientset.CoreV1().ConfigMaps(admissionReview.Request.Namespace).Create(context.TODO(), cm, metav1.CreateOptions{})
				if err != nil {
					app.HandleError(w, r, err)
					return
				}
			} else {
				// We do exist, so update the data with current configuration from our scope namespace configmap
				cmlocal.Data = cm.Data
				log.Debug().Interface("cm", cm).Str("namespace", admissionReview.Request.Namespace).Msgf("updating configmap")
				_, err := clientset.CoreV1().ConfigMaps(admissionReview.Request.Namespace).Update(context.TODO(), cmlocal, metav1.UpdateOptions{})
				if err != nil {
					app.HandleError(w, r, err)
					return
				}
			}
		}

		cmd := []string{
			"/usr/local/bin/scope",
			"excrete",
		}
		if len(app.CriblDest) > 0 {
			cmd = append(cmd,
				"--cribldest",
				app.CriblDest,
			)
		} else {
			cmd = append(cmd,
				"--metricdest",
				app.MetricDest,
				"--metricformat",
				app.MetricFormat,
				"--eventdest",
				app.EventDest,
			)
		}
		cmd = append(cmd, "/scope")
		// Scope initcontainer will output the scope binary and scope library in the scope volume
		pod.Spec.InitContainers = append(pod.Spec.InitContainers, corev1.Container{
			Name:    "scope",
			Image:   fmt.Sprintf("cribl/scope:%s", internal.GetVersion()),
			Command: cmd,
			VolumeMounts: []corev1.VolumeMount{{
				Name:      "scope",
				MountPath: "/scope",
			}},
		})

		// Create scope-conf volume
		// assumed to be emptyDir
		pod.Spec.Volumes = append(pod.Spec.Volumes, corev1.Volume{
			Name: "scope",
		}, corev1.Volume{
			Name: "scope-conf",
			VolumeSource: corev1.VolumeSource{
				ConfigMap: &corev1.ConfigMapVolumeSource{
					LocalObjectReference: corev1.LocalObjectReference{
						Name: "scope",
					},
				},
			},
		})

		// add volume mount to all containers in the pod
		for i := 0; i < len(pod.Spec.Containers); i++ {
			pod.Spec.Containers[i].VolumeMounts = append(pod.Spec.Containers[i].VolumeMounts, corev1.VolumeMount{
				Name:      "scope",
				MountPath: "/scope",
			}, corev1.VolumeMount{
				Name:      "scope-conf",
				MountPath: "/scope/scope.yml",
				SubPath:   "scope.yml",
			})
			if len(app.CriblDest) > 0 {
				pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, corev1.EnvVar{
					Name:  "SCOPE_CRIBL",
					Value: app.CriblDest,
				})
			}
			// Add environment variables to configure scope
			pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, corev1.EnvVar{
				Name:  "LD_PRELOAD",
				Value: "/scope/libscope.so",
			})
			pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, corev1.EnvVar{
				Name:  "SCOPE_CONF_PATH",
				Value: "/scope/scope.yml",
			})
			pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, corev1.EnvVar{
				Name:  "SCOPE_EXEC_PATH",
				Value: "/scope/ldscope",
			})
			pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, corev1.EnvVar{
				Name:  "LD_LIBRARY_PATH",
				Value: fmt.Sprintf("/tmp/libscope-%s", ver[0]),
			})
			// Get some metadata pushed into scope from the K8S downward API
			pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, corev1.EnvVar{
				Name: "SCOPE_TAG_node_name",
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{
						FieldPath: "spec.nodeName",
					},
				},
			})
			pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, corev1.EnvVar{
				Name: "SCOPE_TAG_namespace",
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{
						FieldPath: "metadata.namespace",
					},
				},
			})
			// Add tags from k8s metadata
			for k, v := range pod.ObjectMeta.Labels {
				if strings.HasPrefix(k, "app.kubernetes.io") {
					parts := strings.Split(k, "/")
					if len(parts) > 1 {
						pod.Spec.Containers[i].Env = append(pod.Spec.Containers[i].Env, corev1.EnvVar{
							Name:  fmt.Sprintf("SCOPE_TAG_%s", strings.ToLower(parts[1])),
							Value: v,
						})
					}
				}
			}
		}

		pod.ObjectMeta.Labels["appscope.dev/scope"] = "true"

		initContainersBytes, err := json.Marshal(&pod.Spec.InitContainers)
		if err != nil {
			app.HandleError(w, r, fmt.Errorf("error marshaling initContainers: %v", err))
			return
		}

		containersBytes, err := json.Marshal(&pod.Spec.Containers)
		if err != nil {
			app.HandleError(w, r, fmt.Errorf("error marshaling initContainers: %v", err))
			return
		}

		volumesBytes, err := json.Marshal(&pod.Spec.Volumes)
		if err != nil {
			app.HandleError(w, r, fmt.Errorf("marshall volumes: %v", err))
			return
		}

		labelsBytes, err := json.Marshal(&pod.ObjectMeta.Labels)
		if err != nil {
			app.HandleError(w, r, fmt.Errorf("marshal labels: %v", err))
			return
		}

		// build json patch
		patch = []JSONPatchEntry{
			{
				Op:    "replace",
				Path:  "/metadata/labels",
				Value: labelsBytes,
			},
			{
				Op:    "replace",
				Path:  "/spec/initContainers",
				Value: initContainersBytes,
			},
			{
				Op:    "replace",
				Path:  "/spec/containers",
				Value: containersBytes,
			},
			{
				Op:    "replace",
				Path:  "/spec/volumes",
				Value: volumesBytes,
			},
		}
	}

	patchBytes, err := json.Marshal(&patch)
	if err != nil {
		app.HandleError(w, r, fmt.Errorf("marshall jsonpatch: %v", err))
		return
	}
	log.Debug().RawJSON("patch", patchBytes).Msgf("patch")

	patchType := admissionv1.PatchTypeJSONPatch

	// build admission response
	admissionResponse := &admissionv1.AdmissionResponse{
		UID:       admissionReview.Request.UID,
		Allowed:   true,
		Patch:     patchBytes,
		PatchType: &patchType,
	}

	respAdmissionReview := &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			Kind:       "AdmissionReview",
			APIVersion: "admission.k8s.io/v1",
		},
		Response: admissionResponse,
	}

	w.Header().Set("Content-Type", "application/json")
	outb, err := json.Marshal(respAdmissionReview)
	if err != nil {
		app.HandleError(w, r, fmt.Errorf("json encoding error: %v", err))
		return
	}

	_, err = w.Write(outb)
	if err != nil {
		app.HandleError(w, r, fmt.Errorf("write error: %v", err))
		return
	}
	log.Info().Bool("modified", shouldModify).Msgf("patch returned")
}

// JSONPatchEntry represents a single JSON patch entry
type JSONPatchEntry struct {
	Op    string          `json:"op"`
	Path  string          `json:"path"`
	Value json.RawMessage `json:"value,omitempty"`
}
