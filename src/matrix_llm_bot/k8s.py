from __future__ import annotations

from kubernetes_asyncio import client, config


class K8sClient:
    def __init__(self) -> None:
        self._api_client: client.ApiClient | None = None

    async def _ensure_client(self) -> None:
        if self._api_client is None:
            config.load_incluster_config()
            self._api_client = client.ApiClient()

    async def aclose(self) -> None:
        if self._api_client is not None:
            await self._api_client.close()

    async def cluster_map(self) -> str:
        """Build a full map of the cluster: namespaces, deployments with images, services."""
        await self._ensure_client()
        v1 = client.CoreV1Api(api_client=self._api_client)
        apps = client.AppsV1Api(api_client=self._api_client)

        ns_list = await v1.list_namespace()
        namespaces = [
            ns.metadata.name for ns in ns_list.items
            if not ns.metadata.name.startswith("kube-")
        ]

        lines: list[str] = []
        for ns in sorted(namespaces):
            deps = await apps.list_namespaced_deployment(ns)
            svcs = await v1.list_namespaced_service(ns)
            if not deps.items and not svcs.items:
                continue
            lines.append(f"\n## {ns}")
            for d in deps.items:
                name = d.metadata.name
                ready = d.status.ready_replicas or 0
                desired = d.spec.replicas or 0
                images = [c.image for c in d.spec.template.spec.containers]
                image_str = ", ".join(images)
                status = "UP" if ready == desired and desired > 0 else "DOWN"
                lines.append(f"  deploy/{name}: {status} ({ready}/{desired}) image={image_str}")
            for s in svcs.items:
                if s.metadata.name == "kubernetes":
                    continue
                ports = ", ".join(str(p.port) for p in (s.spec.ports or []))
                lines.append(f"  svc/{s.metadata.name}: ports=[{ports}]")

        return "\n".join(lines)

    async def list_namespaces(self) -> str:
        await self._ensure_client()
        v1 = client.CoreV1Api(api_client=self._api_client)
        ns_list = await v1.list_namespace()
        return "\n".join(f"- {ns.metadata.name}" for ns in ns_list.items)

    async def get_pods(self, namespace: str = "") -> str:
        await self._ensure_client()
        v1 = client.CoreV1Api(api_client=self._api_client)
        pods = await (
            v1.list_namespaced_pod(namespace) if namespace
            else v1.list_pod_for_all_namespaces()
        )
        lines = []
        for p in pods.items:
            ns = p.metadata.namespace
            name = p.metadata.name
            phase = p.status.phase
            ready = sum(1 for c in (p.status.container_statuses or []) if c.ready)
            total = len(p.status.container_statuses or [])
            restarts = sum(c.restart_count for c in (p.status.container_statuses or []))
            lines.append(f"{ns}/{name}: {phase} ({ready}/{total} ready, {restarts} restarts)")
        return "\n".join(lines) if lines else "No pods found"

    async def get_deployments(self, namespace: str = "") -> str:
        await self._ensure_client()
        apps = client.AppsV1Api(api_client=self._api_client)
        deps = await (
            apps.list_namespaced_deployment(namespace) if namespace
            else apps.list_deployment_for_all_namespaces()
        )
        lines = []
        for d in deps.items:
            ns = d.metadata.namespace
            name = d.metadata.name
            ready = d.status.ready_replicas or 0
            desired = d.spec.replicas or 0
            lines.append(f"{ns}/{name}: {ready}/{desired} ready")
        return "\n".join(lines) if lines else "No deployments found"

    async def get_logs(self, namespace: str, pod_name: str, tail_lines: int = 30) -> str:
        await self._ensure_client()
        v1 = client.CoreV1Api(api_client=self._api_client)
        pods = await v1.list_namespaced_pod(namespace)
        match = next(
            (p.metadata.name for p in pods.items if pod_name in p.metadata.name),
            None,
        )
        if not match:
            return f"No pod matching '{pod_name}' found in namespace '{namespace}'"
        logs = await v1.read_namespaced_pod_log(match, namespace, tail_lines=tail_lines)
        return logs if logs else "No logs available"

    async def get_services(self, namespace: str = "") -> str:
        await self._ensure_client()
        v1 = client.CoreV1Api(api_client=self._api_client)
        svcs = await (
            v1.list_namespaced_service(namespace) if namespace
            else v1.list_service_for_all_namespaces()
        )
        lines = []
        for s in svcs.items:
            ns = s.metadata.namespace
            name = s.metadata.name
            stype = s.spec.type
            ports = ", ".join(str(p.port) for p in (s.spec.ports or []))
            lines.append(f"{ns}/{name}: {stype} ports=[{ports}]")
        return "\n".join(lines) if lines else "No services found"

    async def service_status(self, service_name: str) -> str:
        """Deterministic status report for a named service across all namespaces."""
        await self._ensure_client()
        v1 = client.CoreV1Api(api_client=self._api_client)
        apps = client.AppsV1Api(api_client=self._api_client)

        deps = await apps.list_deployment_for_all_namespaces()
        pods = await v1.list_pod_for_all_namespaces()

        matched_deps = [
            d for d in deps.items
            if service_name.lower() in d.metadata.name.lower()
        ]
        matched_pods = [
            p for p in pods.items
            if service_name.lower() in p.metadata.name.lower()
        ]

        if not matched_deps and not matched_pods:
            return f"No deployment or pods found matching '{service_name}'"

        lines: list[str] = []
        for d in matched_deps:
            ns = d.metadata.namespace
            name = d.metadata.name
            ready = d.status.ready_replicas or 0
            desired = d.spec.replicas or 0
            health = "HEALTHY" if ready == desired and desired > 0 else "UNHEALTHY"
            images = [c.image for c in d.spec.template.spec.containers]
            lines.append(f"deploy/{ns}/{name}: {health} ({ready}/{desired} replicas)")
            for img in images:
                lines.append(f"  image: {img}")

        for p in matched_pods:
            ns = p.metadata.namespace
            name = p.metadata.name
            phase = p.status.phase or "Unknown"
            cs = p.status.container_statuses or []
            ready = sum(1 for c in cs if c.ready)
            total = len(cs)
            restarts = sum(c.restart_count for c in cs)
            lines.append(f"pod/{ns}/{name}: {phase} ({ready}/{total} ready, {restarts} restarts)")

        return "\n".join(lines)

    async def service_health(self, service_name: str = "") -> str:
        """Pod-level health summary: healthy/total count, names of unhealthy pods.
        If service_name is empty, reports on all pods in the cluster."""
        await self._ensure_client()
        v1 = client.CoreV1Api(api_client=self._api_client)
        pods = await v1.list_pod_for_all_namespaces()

        if service_name:
            matched = [
                p for p in pods.items
                if service_name.lower() in p.metadata.name.lower()
            ]
            if not matched:
                return f"No pods found matching '{service_name}'"
            label = service_name
        else:
            matched = [
                p for p in pods.items
                if not (p.metadata.namespace or "").startswith("kube-")
            ]
            if not matched:
                return "No pods found"
            label = "cluster"

        healthy = []
        unhealthy = []
        for p in matched:
            ns = p.metadata.namespace
            name = p.metadata.name
            phase = p.status.phase or "Unknown"
            cs = p.status.container_statuses or []
            all_ready = bool(cs) and all(c.ready for c in cs)
            restarts = sum(c.restart_count for c in cs)
            # Collect reason for unhealthy pods
            reason = ""
            for c in cs:
                if not c.ready:
                    state = c.state
                    if state.waiting and state.waiting.reason:
                        reason = state.waiting.reason
                    elif state.terminated and state.terminated.reason:
                        reason = state.terminated.reason
            if phase == "Running" and all_ready:
                healthy.append(f"{ns}/{name}")
            else:
                detail = f"{phase}"
                if reason:
                    detail += f"/{reason}"
                if restarts:
                    detail += f", {restarts} restarts"
                unhealthy.append(f"  UNHEALTHY: {ns}/{name} ({detail})")

        total = len(matched)
        h_count = len(healthy)
        lines = [f"{label}: {h_count}/{total} pods healthy"]
        lines.extend(unhealthy)
        return "\n".join(lines)

    async def service_version(self, service_name: str) -> str:
        """Return image tags for all deployments matching service_name across all namespaces."""
        await self._ensure_client()
        apps = client.AppsV1Api(api_client=self._api_client)
        deps = await apps.list_deployment_for_all_namespaces()
        lines = []
        for d in deps.items:
            if service_name.lower() in d.metadata.name.lower():
                ns = d.metadata.namespace
                name = d.metadata.name
                images = [c.image for c in d.spec.template.spec.containers]
                lines.append(f"{ns}/{name}: {', '.join(images)}")
        return "\n".join(lines) if lines else f"No deployment matching '{service_name}' found"

    async def check_service_health(self, service_name: str) -> str:
        await self._ensure_client()
        apps = client.AppsV1Api(api_client=self._api_client)
        deps = await apps.list_deployment_for_all_namespaces()
        results = []
        for d in deps.items:
            if service_name.lower() in d.metadata.name.lower():
                ns = d.metadata.namespace
                name = d.metadata.name
                ready = d.status.ready_replicas or 0
                desired = d.spec.replicas or 0
                status = "HEALTHY" if ready == desired and desired > 0 else "UNHEALTHY"
                results.append(f"{ns}/{name}: {status} ({ready}/{desired} replicas ready)")
        if not results:
            return f"No deployment matching '{service_name}' found"
        return "\n".join(results)
