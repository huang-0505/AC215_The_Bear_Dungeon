import os
import pulumi
import pulumi_docker_build as docker_build
from pulumi_gcp import artifactregistry
from pulumi import CustomTimeouts
import datetime

# 🔧 Get project info
project = pulumi.Config("gcp").require("project")
location = os.environ["GCP_REGION"]

# 🕒 Timestamp for tagging
timestamp_tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
repository_name = "dnd-master-repository"
registry_url = f"{location}-docker.pkg.dev/{project}/{repository_name}"

# Create Artifact Registry repository for Docker images
docker_repository = artifactregistry.Repository(
    "dnd-master-repository",
    repository_id=repository_name,
    location=location,
    format="DOCKER",
    description="Docker repository for DnD Master application images",
)

# 🔐 Reference to existing Artifact Registry repository (dnd-app-repository)
# Note: Run this command before using pulumi up to configure authentication:
# gcloud auth configure-docker us-central1-docker.pkg.dev
app_repository_name = "dnd-app-repository"
app_repository_location = "us-central1"
app_registry_url = f"{app_repository_location}-docker.pkg.dev/{project}/{app_repository_name}"

# Export the app repository details for reference
pulumi.export("app-repository-name", app_repository_name)
pulumi.export("app-repository-location", app_repository_location)
pulumi.export("app-registry-url", app_registry_url)

# Docker Build + Push -> API Service (Backend)
image_config = {
    "image_name": "dnd-master-api-service",
    "context_path": "/project-root/src/backend",
    "dockerfile": "Dockerfile"
}
api_service_image = docker_build.Image(
    f"build-{image_config["image_name"]}",
    tags=[pulumi.Output.concat(registry_url, "/", image_config["image_name"], ":", timestamp_tag)],
    context=docker_build.BuildContextArgs(location=image_config["context_path"]),
    dockerfile={"location": f"{image_config["context_path"]}/{image_config["dockerfile"]}"},
    platforms=[docker_build.Platform.LINUX_AMD64],
    push=True,
    opts=pulumi.ResourceOptions(
        custom_timeouts=CustomTimeouts(create="30m"),
        retain_on_delete=True,
        depends_on=[docker_repository]
    )
)
# Export references to stack
pulumi.export("dnd-master-api-service-ref", api_service_image.ref)
pulumi.export("dnd-master-api-service-tags", api_service_image.tags)

# Docker Build + Push -> Frontend
image_config = {
    "image_name": "dnd-master-frontend-react",
    "context_path": "/project-root/src/frontend",
    "dockerfile": "Dockerfile"
}
frontend_image = docker_build.Image(
    f"build-{image_config["image_name"]}",
    tags=[pulumi.Output.concat(registry_url, "/", image_config["image_name"], ":", timestamp_tag)],
    context=docker_build.BuildContextArgs(location=image_config["context_path"]),
    dockerfile={"location": f"{image_config["context_path"]}/{image_config["dockerfile"]}"},
    platforms=[docker_build.Platform.LINUX_AMD64],
    push=True,
    opts=pulumi.ResourceOptions(
        custom_timeouts=CustomTimeouts(create="30m"),
        retain_on_delete=True,
        depends_on=[docker_repository]
    )
)
pulumi.export("dnd-master-frontend-react-ref", frontend_image.ref)
pulumi.export("dnd-master-frontend-react-tags", frontend_image.tags)

# Docker Build + Push -> Orchestrator (API Gateway)
image_config = {
    "image_name": "dnd-master-orchestrator",
    "context_path": "/project-root/src/orchestrator",
    "dockerfile": "Dockerfile"
}
orchestrator_image = docker_build.Image(
    f"build-{image_config["image_name"]}",
    tags=[pulumi.Output.concat(registry_url, "/", image_config["image_name"], ":", timestamp_tag)],
    context=docker_build.BuildContextArgs(location=image_config["context_path"]),
    dockerfile={"location": f"{image_config["context_path"]}/{image_config["dockerfile"]}"},
    platforms=[docker_build.Platform.LINUX_AMD64],
    push=True,
    opts=pulumi.ResourceOptions(
        custom_timeouts=CustomTimeouts(create="30m"),
        retain_on_delete=True,
        depends_on=[docker_repository]
    )
)
# Export references to stack
pulumi.export("dnd-master-orchestrator-ref", orchestrator_image.ref)
pulumi.export("dnd-master-orchestrator-tags", orchestrator_image.tags)

# Docker Build + Push -> Rule Agent
image_config = {
    "image_name": "dnd-master-rule-agent",
    "context_path": "/project-root/src/rule_agent",
    "dockerfile": "Dockerfile"
}
rule_agent_image = docker_build.Image(
    f"build-{image_config["image_name"]}",
    tags=[pulumi.Output.concat(registry_url, "/", image_config["image_name"], ":", timestamp_tag)],
    context=docker_build.BuildContextArgs(location=image_config["context_path"]),
    dockerfile={"location": f"{image_config["context_path"]}/{image_config["dockerfile"]}"},
    platforms=[docker_build.Platform.LINUX_AMD64],
    push=True,
    opts=pulumi.ResourceOptions(
        custom_timeouts=CustomTimeouts(create="30m"),
        retain_on_delete=True,
        depends_on=[docker_repository]
    )
)
# Export references to stack
pulumi.export("dnd-master-rule-agent-ref", rule_agent_image.ref)
pulumi.export("dnd-master-rule-agent-tags", rule_agent_image.tags)