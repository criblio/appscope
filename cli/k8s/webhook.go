package k8s

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/criblio/scope/internal"
	"github.com/rs/zerolog/log"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// App contains configuration for a webhook server
type App struct {
	*Options
}

// HandleMutate handles the mutate endpoint
func (app *App) HandleMutate(w http.ResponseWriter, r *http.Request) {
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

	patch := []JSONPatchEntry{}
	if shouldModify {
		log.Debug().Interface("pod", pod).Msgf("modifying pod")
		pod.Spec.InitContainers = append(pod.Spec.InitContainers, corev1.Container{
			Name:  "scope",
			Image: fmt.Sprintf("cribl/scope:%s", internal.GetVersion()),
			Command: []string{"/usr/local/bin/scope",
				"excrete",
				"--metricdest",
				app.MetricDest,
				"--metricformat",
				app.MetricFormat,
				"--eventdest",
				app.EventDest,
				"/scope",
			},
			VolumeMounts: []corev1.VolumeMount{{
				Name:      "scope",
				MountPath: "/scope",
			}},
		})

		// assumed to be emptyDir
		pod.Spec.Volumes = append(pod.Spec.Volumes, corev1.Volume{
			Name: "scope",
		})

		// add volume mount to all containers in the pod
		for i := 0; i < len(pod.Spec.Containers); i++ {
			pod.Spec.Containers[i].VolumeMounts = append(pod.Spec.Containers[i].VolumeMounts, corev1.VolumeMount{
				Name:      "scope",
				MountPath: "/scope",
			})
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
