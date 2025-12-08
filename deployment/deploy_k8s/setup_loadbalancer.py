import pulumi
import pulumi_gcp as gcp
from pulumi import StackReference, ResourceOptions, Output
import pulumi_kubernetes as k8s


def setup_loadbalancer(
    namespace, k8s_provider, orchestrator_service, frontend_service, app_name
):
    # Nginx Ingress Controller using Helm and Create Ingress Resource
    nginx_helm = k8s.helm.v3.Release(
        "nginx-f5",
        chart="nginx-ingress",
        version="2.3.1",  # pick a current stable version from F5 docs/releases
        namespace=namespace.metadata.name,
        repository_opts=k8s.helm.v3.RepositoryOptsArgs(
            repo="https://helm.nginx.com/stable"
        ),
        values={
            "controller": {
                # Service exposed as a LoadBalancer with your static IP
                "service": {
                    "type": "LoadBalancer",
                },
                # Resource requests/limits (similar to what you had)
                "resources": {
                    "requests": {
                        "memory": "128Mi",
                        "cpu": "100m",
                    },
                    "limits": {
                        "memory": "256Mi",
                        "cpu": "200m",
                    },
                },
                "replicaCount": 1,
                # IngressClass config – default class name is "nginx" in this chart
                "ingressClass": {
                    "name": "nginx",
                    "create": True,
                    "setAsDefaultIngress": True,
                },
            },
        },
        opts=ResourceOptions(provider=k8s_provider),
    )

    # Get the service created by Helm to extract the LoadBalancer IP
    nginx_service = k8s.core.v1.Service.get(
        "nginx-ingress-service",
        pulumi.Output.concat(
            nginx_helm.status.namespace,
            "/",
            nginx_helm.status.name,
            "-nginx-ingress-controller",  # often resolves to <release-name>-ingress-nginx-controller
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[nginx_helm]),
    )
    ip_address = nginx_service.status.load_balancer.ingress[0].ip
    host = ip_address.apply(lambda ip: f"{ip}.sslip.io")

    ingress = k8s.networking.v1.Ingress(
        f"{app_name}-ingress",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name=f"{app_name}-ingress",
            namespace=namespace.metadata.name,
            annotations={
                # cert-manager integration
                "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                # F5 NIC supports this variant for SSL redirect
                # "ingress.kubernetes.io/ssl-redirect": "true",
                "ingress.kubernetes.io/ssl-redirect": "false",
                # Rewrite /api/* to /* to match nginx.conf behavior
                # F5 NGINX Ingress uses nginx.org/rewrites annotation
                # Format: "serviceName=<svc-name> rewrite=<path>"
                "nginx.org/rewrites": "serviceName=orchestrator rewrite=/",
            },
        ),
        spec=k8s.networking.v1.IngressSpecArgs(
            ingress_class_name="nginx",  # Use nginx
            rules=[
                k8s.networking.v1.IngressRuleArgs(
                    host=host,
                    http=k8s.networking.v1.HTTPIngressRuleValueArgs(
                        paths=[
                            # Orchestrator/API Gateway service
                            # Path /api/ matches all requests starting with /api/
                            # The rewrite annotation will strip /api and forward to orchestrator
                            k8s.networking.v1.HTTPIngressPathArgs(
                                path="/api/",
                                path_type="Prefix",
                                backend=k8s.networking.v1.IngressBackendArgs(
                                    service=k8s.networking.v1.IngressServiceBackendArgs(
                                        name=orchestrator_service.metadata["name"],
                                        port=k8s.networking.v1.ServiceBackendPortArgs(
                                            number=8000
                                        ),
                                    )
                                ),
                            ),
                            # Frontend
                            k8s.networking.v1.HTTPIngressPathArgs(
                                path="/",
                                path_type="Prefix",
                                backend=k8s.networking.v1.IngressBackendArgs(
                                    service=k8s.networking.v1.IngressServiceBackendArgs(
                                        name=frontend_service.metadata["name"],
                                        port=k8s.networking.v1.ServiceBackendPortArgs(
                                            number=3000
                                        ),
                                    )
                                ),
                            ),
                        ]
                    ),
                )
            ],
        ),
        opts=ResourceOptions(
            provider=k8s_provider,
            depends_on=[nginx_helm],
        ),
    )

    return ip_address, ingress, host
