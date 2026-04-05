from __future__ import annotations

from kubernetes_asyncio import client, config

from . import queries


class K8sClient:
    def __init__(self) -> None:
        self._api: client.ApiClient | None = None

    async def _ensure(self) -> client.ApiClient:
        if self._api is None:
            config.load_incluster_config()
            self._api = client.ApiClient()
        return self._api

    async def aclose(self) -> None:
        if self._api is not None:
            await self._api.close()

    # --- cluster-wide ---

    async def cluster_map(self) -> str:
        return await queries.cluster_map(await self._ensure())

    async def list_namespaces(self) -> str:
        return await queries.list_namespaces(await self._ensure())

    async def get_pods(self, namespace: str = "") -> str:
        return await queries.get_pods(await self._ensure(), namespace)

    async def get_deployments(self, namespace: str = "") -> str:
        return await queries.get_deployments(await self._ensure(), namespace)

    async def get_services(self, namespace: str = "") -> str:
        return await queries.get_services(await self._ensure(), namespace)

    async def get_logs(self, namespace: str, pod_name: str, tail_lines: int = 30) -> str:
        return await queries.get_logs(await self._ensure(), namespace, pod_name, tail_lines)

    # --- service-level ---

    async def service_status(self, service_name: str) -> str:
        return await queries.service_status(await self._ensure(), service_name)

    async def service_health(self, service_name: str = "") -> str:
        return await queries.service_health(await self._ensure(), service_name)

    async def service_version(self, service_name: str) -> str:
        return await queries.service_version(await self._ensure(), service_name)

    async def check_service_health(self, service_name: str) -> str:
        return await queries.check_service_health(await self._ensure(), service_name)

    async def dispatch_tool(self, name: str, args: dict) -> str:
        """Execute a k8s tool call by name, as requested by the LLM."""
        if name == "k8s_check_health":
            return await self.check_service_health(args["service_name"])
        if name == "k8s_get_pods":
            return await self.get_pods(args.get("namespace", ""))
        if name == "k8s_get_logs":
            return await self.get_logs(args["namespace"], args["pod_name"], args.get("tail_lines", 30))
        if name == "k8s_list_namespaces":
            return await self.list_namespaces()
        if name == "k8s_get_deployments":
            return await self.get_deployments(args.get("namespace", ""))
        if name == "k8s_get_services":
            return await self.get_services(args.get("namespace", ""))
        if name == "k8s_cluster_map":
            return await self.cluster_map()
        return f"Unknown k8s tool: {name}"
