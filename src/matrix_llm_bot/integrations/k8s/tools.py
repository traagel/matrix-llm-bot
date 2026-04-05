from __future__ import annotations

K8S_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "k8s_check_health",
            "description": "Check if a specific service/deployment is healthy and running in the k3s cluster.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "Name of the service to check"},
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_get_pods",
            "description": "List pods and their status. Optionally filter by namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Namespace to filter by. Empty = all."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_get_logs",
            "description": "Get recent logs from a pod.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Namespace the pod is in"},
                    "pod_name": {"type": "string", "description": "Full or partial pod name"},
                    "tail_lines": {"type": "integer", "description": "Number of log lines (default 30)"},
                },
                "required": ["namespace", "pod_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_list_namespaces",
            "description": "List all namespaces in the cluster.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_get_deployments",
            "description": "List deployments and replica status. Optionally filter by namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Namespace to filter by. Empty = all."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_get_services",
            "description": "List Kubernetes services and their ports. Optionally filter by namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Namespace to filter by. Empty = all."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_cluster_map",
            "description": "Get a full map of the cluster: namespaces, deployments with images/versions, services.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
