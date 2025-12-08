import pulumi
import pulumi_gcp as gcp
from pulumi import StackReference, ResourceOptions, Output
import pulumi_kubernetes as k8s


def setup_containers(project, namespace, k8s_provider, ksa_name, app_name):
    # Get image references from deploy_images stack
    # For local backend, use: "organization/project/stack"
    images_stack = pulumi.StackReference("organization/deploy-images/dev")
    # Get the image tags (these are arrays, so we take the first element)
    api_service_tag = images_stack.get_output("dnd-master-api-service-tags")
    frontend_tag = images_stack.get_output("dnd-master-frontend-react-tags")
    orchestrator_tag = images_stack.get_output("dnd-master-orchestrator-tags")
    rule_agent_tag = images_stack.get_output("dnd-master-rule-agent-tags")

    # General persistent storage for application data (5Gi)
    persistent_pvc = k8s.core.v1.PersistentVolumeClaim(
        "persistent-pvc",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="persistent-pvc",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.PersistentVolumeClaimSpecArgs(
            access_modes=["ReadWriteOnce"],  # Single pod read/write access
            resources=k8s.core.v1.VolumeResourceRequirementsArgs(
                requests={"storage": "5Gi"},  # Request 5GB of persistent storage
            ),
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace]),
    )

    # Dedicated storage for ChromaDB vector database (10Gi)
    chromadb_pvc = k8s.core.v1.PersistentVolumeClaim(
        "chromadb-pvc",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="chromadb-pvc",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.PersistentVolumeClaimSpecArgs(
            access_modes=["ReadWriteOnce"],  # Single pod read/write access
            resources=k8s.core.v1.VolumeResourceRequirementsArgs(
                requests={"storage": "10Gi"},  # Request 10GB for vector embeddings
            ),
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace]),
    )

    # --- Frontend Deployment ---
    # Creates pods running the frontend container on port 3000
    # ram 1.7 gb
    frontend_deployment = k8s.apps.v1.Deployment(
        "frontend",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="frontend",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "frontend"},  # Select pods with this label
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "frontend"},  # Label assigned to pods
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="frontend",
                            image=frontend_tag.apply(
                                lambda tags: tags[0]
                            ),  # Container image (placeholder - needs to be filled)
                            image_pull_policy="IfNotPresent",  # Use cached image if available
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=3000,  # Frontend app listens on port 3000
                                    protocol="TCP",
                                )
                            ],
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="NODE_ENV",
                                    value="production",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="NEXT_PUBLIC_API_BASE_URL",
                                    value="/api",
                                ),
                            ],
                            resources=k8s.core.v1.ResourceRequirementsArgs(
                                requests={"cpu": "250m", "memory": "2Gi"},
                                limits={"cpu": "500m", "memory": "3Gi"},
                            ),
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace]),
    )

    frontend_service = k8s.core.v1.Service(
        "frontend-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="frontend",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",  # Internal only - not exposed outside cluster
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=3000,  # Service port
                    target_port=3000,  # Container port to forward to
                    protocol="TCP",
                )
            ],
            selector={"run": "frontend"},  # Route traffic to pods with this label
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider, depends_on=[frontend_deployment]
        ),
    )

    # vector-db deployment
    vector_db_deployment = k8s.apps.v1.Deployment(
        "vector-db",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="vector-db",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "vector-db"},
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "vector-db"},
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="vector-db",
                            image="chromadb/chroma",
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=8000,
                                    protocol="TCP",
                                )
                            ],
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="IS_PERSISTENT", value="TRUE"
                                ),  # Enable data persistence
                                k8s.core.v1.EnvVarArgs(
                                    name="ANONYMIZED_TELEMETRY", value="FALSE"
                                ),  # Disable telemetry
                            ],
                            volume_mounts=[
                                k8s.core.v1.VolumeMountArgs(
                                    name="chromadb-storage",
                                    mount_path="/chroma/chroma",
                                ),
                            ],
                            resources=k8s.core.v1.ResourceRequirementsArgs(
                                requests={"cpu": "100m", "memory": "128Mi"},
                                limits={"cpu": "200m", "memory": "256Mi"},
                            ),
                        ),
                    ],
                    volumes=[
                        k8s.core.v1.VolumeArgs(
                            name="chromadb-storage",
                            persistent_volume_claim=k8s.core.v1.PersistentVolumeClaimVolumeSourceArgs(
                                claim_name=chromadb_pvc.metadata.name,  # Mount the 10Gi PVC
                            ),
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider, depends_on=[namespace, chromadb_pvc]
        ),
    )

    # vector-db Service
    vector_db_service = k8s.core.v1.Service(
        "vector-db-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="vector-db",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",  # Internal only
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=8000,
                    target_port=8000,
                    protocol="TCP",
                )
            ],
            selector={"run": "vector-db"},
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider, depends_on=[vector_db_deployment]
        ),
    )

    # api_service Deployment
    api_deployment = k8s.apps.v1.Deployment(
        "api",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="api",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "api"},
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "api"},
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    service_account_name=ksa_name,  # Use KSA for Workload Identity (GCP access)
                    security_context=k8s.core.v1.PodSecurityContextArgs(
                        fs_group=1000,
                    ),
                    volumes=[
                        k8s.core.v1.VolumeArgs(
                            name="persistent-vol",
                            persistent_volume_claim=k8s.core.v1.PersistentVolumeClaimVolumeSourceArgs(
                                claim_name=persistent_pvc.metadata.name,  # Temporary storage (lost on restart)
                            ),
                        )
                    ],
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="api",
                            image=api_service_tag.apply(
                                lambda tags: tags[0]
                            ),  # API container image (placeholder - needs to be filled)
                            image_pull_policy="IfNotPresent",
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=9000,  # API server port
                                    protocol="TCP",
                                )
                            ],
                            volume_mounts=[
                                k8s.core.v1.VolumeMountArgs(
                                    name="persistent-vol",
                                    mount_path="/persistent",  # Temporary file storage
                                )
                            ],
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="GCP_PROJECT",
                                    value=project,
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="GCP_LOCATION",
                                    value="us-central1",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="DEV",
                                    value="0",
                                ),
                            ],
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider, depends_on=[vector_db_service]
        ),
    )

    # api_service Service
    api_service = k8s.core.v1.Service(
        "api-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="api",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",  # Internal only
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=9000,
                    target_port=9000,
                    protocol="TCP",
                )
            ],
            selector={"run": "api"},
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[api_deployment]),
    )

    # Rule Agent Deployment
    rule_agent_deployment = k8s.apps.v1.Deployment(
        "rule-agent",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="rule-agent",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "rule-agent"},
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "rule-agent"},
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    service_account_name=ksa_name,
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="rule-agent",
                            image=rule_agent_tag.apply(lambda tags: tags[0]),
                            image_pull_policy="IfNotPresent",
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=9002,
                                    protocol="TCP",
                                )
                            ],
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="GCP_PROJECT",
                                    value=project,
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="GCP_LOCATION",
                                    value="us-central1",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="CHROMADB_HOST",
                                    value="vector-db",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="CHROMADB_PORT",
                                    value="8000",
                                ),
                            ],
                            command=["/bin/bash"],
                            args=["-c", "uv run uvicorn app:app --host 0.0.0.0 --port 9002"],
                            resources=k8s.core.v1.ResourceRequirementsArgs(
                                requests={"cpu": "250m", "memory": "512Mi"},
                                limits={"cpu": "500m", "memory": "1Gi"},
                            ),
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider, depends_on=[vector_db_service]
        ),
    )

    # Rule Agent Service
    rule_agent_service = k8s.core.v1.Service(
        "rule-agent-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="rule-agent",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=9002,
                    target_port=9002,
                    protocol="TCP",
                )
            ],
            selector={"run": "rule-agent"},
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[rule_agent_deployment]),
    )

    # Orchestrator Deployment
    orchestrator_deployment = k8s.apps.v1.Deployment(
        "orchestrator",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="orchestrator",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.apps.v1.DeploymentSpecArgs(
            selector=k8s.meta.v1.LabelSelectorArgs(
                match_labels={"run": "orchestrator"},
            ),
            template=k8s.core.v1.PodTemplateSpecArgs(
                metadata=k8s.meta.v1.ObjectMetaArgs(
                    labels={"run": "orchestrator"},
                ),
                spec=k8s.core.v1.PodSpecArgs(
                    service_account_name=ksa_name,
                    containers=[
                        k8s.core.v1.ContainerArgs(
                            name="orchestrator",
                            image=orchestrator_tag.apply(lambda tags: tags[0]),
                            image_pull_policy="IfNotPresent",
                            ports=[
                                k8s.core.v1.ContainerPortArgs(
                                    container_port=8000,
                                    protocol="TCP",
                                )
                            ],
                            env=[
                                k8s.core.v1.EnvVarArgs(
                                    name="GCP_PROJECT",
                                    value=project,
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="GCP_LOCATION",
                                    value="us-central1",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="NARRATOR_ENDPOINT_ID",
                                    value="5165249441082376192",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="RULE_AGENT_URL",
                                    value="http://rule-agent:9002",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="COMBAT_AGENT_URL",
                                    value="http://api:9000",
                                ),
                                k8s.core.v1.EnvVarArgs(
                                    name="NARRATOR_AGENT_URL",
                                    value="http://narrator-agent:9001",
                                ),
                            ],
                            resources=k8s.core.v1.ResourceRequirementsArgs(
                                requests={"cpu": "250m", "memory": "512Mi"},
                                limits={"cpu": "500m", "memory": "1Gi"},
                            ),
                        ),
                    ],
                ),
            ),
        ),
        opts=pulumi.ResourceOptions(
            provider=k8s_provider, depends_on=[rule_agent_service, api_service]
        ),
    )

    # Orchestrator Service
    orchestrator_service = k8s.core.v1.Service(
        "orchestrator-service",
        metadata=k8s.meta.v1.ObjectMetaArgs(
            name="orchestrator",
            namespace=namespace.metadata.name,
        ),
        spec=k8s.core.v1.ServiceSpecArgs(
            type="ClusterIP",
            ports=[
                k8s.core.v1.ServicePortArgs(
                    port=8000,
                    target_port=8000,
                    protocol="TCP",
                )
            ],
            selector={"run": "orchestrator"},
        ),
        opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[orchestrator_deployment]),
    )

    return frontend_service, orchestrator_service
