from __future__ import annotations

from kubernetes_asyncio import client


async def cluster_map(api: client.ApiClient) -> str:
    """Full cluster snapshot: namespaces, deployments with images, services."""
    v1 = client.CoreV1Api(api_client=api)
    apps = client.AppsV1Api(api_client=api)

    ns_list = await v1.list_namespace()
    namespaces = [ns.metadata.name for ns in ns_list.items if not ns.metadata.name.startswith("kube-")]

    lines: list[str] = []
    for ns in sorted(namespaces):
        deps = await apps.list_namespaced_deployment(ns)
        svcs = await v1.list_namespaced_service(ns)
        if not deps.items and not svcs.items:
            continue
        lines.append(f"\n## {ns}")
        for d in deps.items:
            ready = d.status.ready_replicas or 0
            desired = d.spec.replicas or 0
            images = ", ".join(c.image for c in d.spec.template.spec.containers)
            status = "UP" if ready == desired and desired > 0 else "DOWN"
            lines.append(f"  deploy/{d.metadata.name}: {status} ({ready}/{desired}) image={images}")
        for s in svcs.items:
            if s.metadata.name == "kubernetes":
                continue
            ports = ", ".join(str(p.port) for p in (s.spec.ports or []))
            lines.append(f"  svc/{s.metadata.name}: ports=[{ports}]")

    return "\n".join(lines)


async def list_namespaces(api: client.ApiClient) -> str:
    v1 = client.CoreV1Api(api_client=api)
    ns_list = await v1.list_namespace()
    return "\n".join(f"- {ns.metadata.name}" for ns in ns_list.items)


async def get_pods(api: client.ApiClient, namespace: str = "") -> str:
    v1 = client.CoreV1Api(api_client=api)
    pods = await (v1.list_namespaced_pod(namespace) if namespace else v1.list_pod_for_all_namespaces())
    lines = []
    for p in pods.items:
        cs = p.status.container_statuses or []
        ready = sum(1 for c in cs if c.ready)
        restarts = sum(c.restart_count for c in cs)
        lines.append(
            f"{p.metadata.namespace}/{p.metadata.name}: {p.status.phase} ({ready}/{len(cs)} ready, {restarts} restarts)"
        )
    return "\n".join(lines) if lines else "No pods found"


async def get_deployments(api: client.ApiClient, namespace: str = "") -> str:
    apps = client.AppsV1Api(api_client=api)
    deps = await (
        apps.list_namespaced_deployment(namespace) if namespace else apps.list_deployment_for_all_namespaces()
    )
    lines = []
    for d in deps.items:
        ready = d.status.ready_replicas or 0
        desired = d.spec.replicas or 0
        lines.append(f"{d.metadata.namespace}/{d.metadata.name}: {ready}/{desired} ready")
    return "\n".join(lines) if lines else "No deployments found"


async def get_logs(api: client.ApiClient, namespace: str, pod_name: str, tail_lines: int = 30) -> str:
    v1 = client.CoreV1Api(api_client=api)
    pods = await v1.list_namespaced_pod(namespace)
    match = next((p.metadata.name for p in pods.items if pod_name in p.metadata.name), None)
    if not match:
        return f"No pod matching '{pod_name}' found in namespace '{namespace}'"
    logs = await v1.read_namespaced_pod_log(match, namespace, tail_lines=tail_lines)
    return logs if logs else "No logs available"


async def get_services(api: client.ApiClient, namespace: str = "") -> str:
    v1 = client.CoreV1Api(api_client=api)
    svcs = await (v1.list_namespaced_service(namespace) if namespace else v1.list_service_for_all_namespaces())
    lines = []
    for s in svcs.items:
        ports = ", ".join(str(p.port) for p in (s.spec.ports or []))
        lines.append(f"{s.metadata.namespace}/{s.metadata.name}: {s.spec.type} ports=[{ports}]")
    return "\n".join(lines) if lines else "No services found"


async def service_status(api: client.ApiClient, service_name: str) -> str:
    """Deployment + pod detail for a named service."""
    v1 = client.CoreV1Api(api_client=api)
    apps = client.AppsV1Api(api_client=api)

    deps = await apps.list_deployment_for_all_namespaces()
    pods = await v1.list_pod_for_all_namespaces()

    matched_deps = [d for d in deps.items if service_name.lower() in d.metadata.name.lower()]
    matched_pods = [p for p in pods.items if service_name.lower() in p.metadata.name.lower()]

    if not matched_deps and not matched_pods:
        return f"No deployment or pods found matching '{service_name}'"

    lines: list[str] = []
    for d in matched_deps:
        ready = d.status.ready_replicas or 0
        desired = d.spec.replicas or 0
        health = "HEALTHY" if ready == desired and desired > 0 else "UNHEALTHY"
        lines.append(f"deploy/{d.metadata.namespace}/{d.metadata.name}: {health} ({ready}/{desired} replicas)")
        for img in (c.image for c in d.spec.template.spec.containers):
            lines.append(f"  image: {img}")

    for p in matched_pods:
        cs = p.status.container_statuses or []
        ready = sum(1 for c in cs if c.ready)
        restarts = sum(c.restart_count for c in cs)
        lines.append(
            f"pod/{p.metadata.namespace}/{p.metadata.name}: {p.status.phase or 'Unknown'}"
            f" ({ready}/{len(cs)} ready, {restarts} restarts)"
        )

    return "\n".join(lines)


async def service_health(api: client.ApiClient, service_name: str = "") -> str:
    """Pod-level health summary. Empty service_name = whole cluster (excluding kube-*)."""
    v1 = client.CoreV1Api(api_client=api)
    pods = await v1.list_pod_for_all_namespaces()

    if service_name:
        matched = [p for p in pods.items if service_name.lower() in p.metadata.name.lower()]
        if not matched:
            return f"No pods found matching '{service_name}'"
        label = service_name
    else:
        matched = [p for p in pods.items if not (p.metadata.namespace or "").startswith("kube-")]
        if not matched:
            return "No pods found"
        label = "cluster"

    unhealthy = []
    healthy_count = 0
    for p in matched:
        cs = p.status.container_statuses or []
        all_ready = bool(cs) and all(c.ready for c in cs)
        restarts = sum(c.restart_count for c in cs)
        phase = p.status.phase or "Unknown"
        if phase == "Running" and all_ready:
            healthy_count += 1
        else:
            reason = next(
                (
                    (c.state.waiting and c.state.waiting.reason)
                    or (c.state.terminated and c.state.terminated.reason)
                    or ""
                    for c in cs
                    if not c.ready
                ),
                "",
            )
            detail = phase
            if reason:
                detail += f"/{reason}"
            if restarts:
                detail += f", {restarts} restarts"
            unhealthy.append(f"  UNHEALTHY: {p.metadata.namespace}/{p.metadata.name} ({detail})")

    lines = [f"{label}: {healthy_count}/{len(matched)} pods healthy"]
    lines.extend(unhealthy)
    return "\n".join(lines)


async def service_version(api: client.ApiClient, service_name: str) -> str:
    """Image tags for all deployments matching service_name."""
    apps = client.AppsV1Api(api_client=api)
    deps = await apps.list_deployment_for_all_namespaces()
    lines = []
    for d in deps.items:
        if service_name.lower() in d.metadata.name.lower():
            images = ", ".join(c.image for c in d.spec.template.spec.containers)
            lines.append(f"{d.metadata.namespace}/{d.metadata.name}: {images}")
    return "\n".join(lines) if lines else f"No deployment matching '{service_name}' found"


async def check_service_health(api: client.ApiClient, service_name: str) -> str:
    """Deployment replica health for a named service."""
    apps = client.AppsV1Api(api_client=api)
    deps = await apps.list_deployment_for_all_namespaces()
    results = []
    for d in deps.items:
        if service_name.lower() in d.metadata.name.lower():
            ready = d.status.ready_replicas or 0
            desired = d.spec.replicas or 0
            status = "HEALTHY" if ready == desired and desired > 0 else "UNHEALTHY"
            results.append(f"{d.metadata.namespace}/{d.metadata.name}: {status} ({ready}/{desired} replicas ready)")
    return "\n".join(results) if results else f"No deployment matching '{service_name}' found"
