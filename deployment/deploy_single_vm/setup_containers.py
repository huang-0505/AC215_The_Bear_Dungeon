import pulumi
from pulumi import ResourceOptions
from pulumi_command import remote
import pulumi_docker as docker


def setup_containers(connection, configure_docker, project, instance_ip, ssh_user):
    """
    Setup and deploy all application containers:
    - Copy GCP secrets to the VM
    - Create Docker network
    - Create persistent directories
    - Deploy frontend container
    - Deploy vector DB (ChromaDB) container
    - Load vector DB data
    - Deploy API service container

    Args:
        connection: SSH connection configuration
        configure_docker: The Docker configuration command (dependency)
        project: GCP project ID

    Returns:
        remote.Command: The last container deployment command (for dependency chaining)
    """
    # Get image references from deploy_images stack
    images_stack = pulumi.StackReference("organization/deploy-images/dev")
    # Get the image tags (these are arrays, so we take the first element)
    api_service_tag = images_stack.get_output("dnd-master-api-service-tags")
    frontend_tag = images_stack.get_output("dnd-master-frontend-react-tags")
    orchestrator_tag = images_stack.get_output("dnd-master-orchestrator-tags")
    rule_agent_tag = images_stack.get_output("dnd-master-rule-agent-tags")
    # Note: Not using vector-db-cli loader - matching docker-compose.yml setup

    # Setup GCP secrets for containers
    copy_secrets = remote.Command(
        "copy-gcp-secrets",
        connection=connection,
        create="""
            sudo mkdir -p /srv/secrets
            sudo chmod 0755 /srv/secrets
        """,
        opts=ResourceOptions(depends_on=[configure_docker]),
    )

    upload_service_account = remote.CopyToRemote(
        "upload-service-account-key",
        connection=connection,
        source=pulumi.FileAsset("/secrets/gcp-service.json"),
        remote_path="/tmp/gcp-service.json",
        opts=ResourceOptions(depends_on=[copy_secrets]),
    )

    move_secrets = remote.Command(
        "move-secrets-to-srv",
        connection=connection,
        create="""
            sudo mv /tmp/gcp-service.json /srv/secrets/gcp-service.json
            sudo chmod 0644 /srv/secrets/gcp-service.json
            sudo chown root:root /srv/secrets/gcp-service.json

            # Activate service account for gcloud (for current user)
            gcloud auth activate-service-account --key-file /srv/secrets/gcp-service.json
            gcloud auth configure-docker us-docker.pkg.dev --quiet
            gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

            # Also activate for root user (Docker may run as root)
            sudo gcloud auth activate-service-account --key-file /srv/secrets/gcp-service.json
            sudo gcloud auth configure-docker us-docker.pkg.dev --quiet
            sudo gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

            # Configure Docker credential helper to use gcloud
            mkdir -p ~/.docker
            cat > ~/.docker/config.json << 'EOF'
{
  "credHelpers": {
    "us-docker.pkg.dev": "gcloud",
    "us-central1-docker.pkg.dev": "gcloud",
    "gcr.io": "gcloud"
  }
}
EOF

            # Also configure for root
            sudo mkdir -p /root/.docker
            sudo bash -c 'cat > /root/.docker/config.json << '\''EOF'\''
{
  "credHelpers": {
    "us-docker.pkg.dev": "gcloud",
    "us-central1-docker.pkg.dev": "gcloud",
    "gcr.io": "gcloud"
  }
}
EOF'
        """,
        opts=ResourceOptions(depends_on=[upload_service_account]),
    )

    # Verify authentication works
    verify_auth = remote.Command(
        "verify-docker-auth",
        connection=connection,
        create="""
            # Test authentication by listing repositories
            echo "Testing authentication to Artifact Registry..."
            gcloud auth print-access-token > /tmp/gcloud-token.txt
            echo "Access token obtained successfully"

            # Verify docker credential helper is installed
            which docker-credential-gcloud || echo "Warning: docker-credential-gcloud not found in PATH"

            # Test docker login
            echo "Testing docker authentication..."
            cat /tmp/gcloud-token.txt | docker login -u oauth2accesstoken --password-stdin https://us-docker.pkg.dev
            cat /tmp/gcloud-token.txt | docker login -u oauth2accesstoken --password-stdin https://us-central1-docker.pkg.dev
            rm /tmp/gcloud-token.txt

            echo "Docker authentication successful!"
        """,
        opts=ResourceOptions(depends_on=[move_secrets]),
    )

    # Create directories on persistent disk
    create_dirs = remote.Command(
        "create-persistent-directories",
        connection=connection,
        create="""
            sudo mkdir -p /mnt/disk-1/persistent
            sudo mkdir -p /mnt/disk-1/chromadb
            sudo chmod 0777 /mnt/disk-1/persistent
            sudo chmod 0777 /mnt/disk-1/chromadb
        """,
        opts=ResourceOptions(depends_on=[verify_auth]),
    )

    # Set up Docker provider with SSH credentials for remote access
    # Note: Authentication is handled by gcloud credential helper configured above
    docker_provider = docker.Provider(
        "docker-provider",
        host=instance_ip.apply(lambda ip: f"ssh://{ssh_user}@{ip}"),
        # SSH options to handle key-based authentication and suppress host checking
        ssh_opts=[
            "-i",
            "/secrets/ssh-key-deployment",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ],
        # Registry auth is handled by gcloud credential helper on the VM
        # No need to specify registry_auth here
        opts=ResourceOptions(depends_on=[create_dirs]),
    )

    # Create Docker network
    docker_network = docker.Network(
        "docker-network",
        name="appnetwork",
        driver="bridge",
        opts=ResourceOptions(provider=docker_provider),
    )

    # Deploy containers
    # Frontend (expose only, no port mapping - accessed via Nginx)
    deploy_frontend = docker.Container(
        "deploy-frontend-container",
        image=frontend_tag.apply(lambda tags: tags[0]),
        name="frontend",
        networks_advanced=[
            docker.ContainerNetworksAdvancedArgs(
                name=docker_network.name,
            ),
        ],
        opts=ResourceOptions(
            provider=docker_provider,
            depends_on=[docker_network, create_dirs],
        ),
    )

    # Vector DB
    deploy_vector_db = docker.Container(
        "deploy-vector-db-container",
        image="chromadb/chroma:latest",
        name="vector-db",
        restart="always",
        # Map container port to host port
        ports=[
            docker.ContainerPortArgs(
                internal=8000,  # Container port
                external=8000,  # Host port
            )
        ],
        # Environment variables for the container
        envs=[
            "IS_PERSISTENT=TRUE",
            "ANONYMIZED_TELEMETRY=FALSE",
        ],
        # Mount persistent volume for ChromaDB data
        volumes=[
            docker.ContainerVolumeArgs(
                host_path="/mnt/disk-1/chromadb",
                container_path="/chroma/chroma",
                read_only=False,
            )
        ],
        # Connect to the app network for inter-container communication
        networks_advanced=[
            docker.ContainerNetworksAdvancedArgs(
                name=docker_network.name,
            ),
        ],
        opts=ResourceOptions(
            provider=docker_provider,
            depends_on=[docker_network],
        ),
    )

    # Note: Skipping vector-db data loading to match docker-compose.yml
    # Assume ChromaDB data is pre-loaded or not required for initial deployment

    # Rule Agent (RAG-based D&D rules validation)
    # expose only, no port mapping
    deploy_rule_agent = docker.Container(
        "deploy-rule-agent-container",
        image=rule_agent_tag.apply(lambda tags: tags[0]),
        name="rule-agent",
        restart="always",
        envs=[
            "GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-service.json",
            f"GCP_PROJECT={project}",
            "GCP_LOCATION=us-central1",
            "CHROMADB_HOST=vector-db",
            "CHROMADB_PORT=8000",
        ],
        volumes=[
            docker.ContainerVolumeArgs(
                host_path="/srv/secrets",
                container_path="/secrets",
                read_only=False,
            ),
        ],
        networks_advanced=[
            docker.ContainerNetworksAdvancedArgs(
                name=docker_network.name,
            )
        ],
        command=["/bin/bash", "-c", "uv run uvicorn app:app --host 0.0.0.0 --port 9002"],
        opts=ResourceOptions(
            provider=docker_provider,
            depends_on=[deploy_vector_db],
        ),
    )

    # Combat Agent (Backend/API Service)
    # expose only, no port mapping
    deploy_api_service = docker.Container(
        "deploy-api-service-container",
        image=api_service_tag.apply(lambda tags: tags[0]),
        name="combat-agent",
        restart="always",
        envs=[
            "GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-service.json",
            f"GCP_PROJECT={project}",
            "GCP_LOCATION=us-central1",
            "DEV=0",
        ],
        volumes=[
            docker.ContainerVolumeArgs(
                host_path="/srv/secrets",
                container_path="/secrets",
                read_only=False,
            ),
        ],
        networks_advanced=[
            docker.ContainerNetworksAdvancedArgs(
                name=docker_network.name,
            )
        ],
        opts=ResourceOptions(
            provider=docker_provider,
            depends_on=[docker_network],
        ),
    )

    # Orchestrator (API Gateway)
    # expose only, no port mapping - accessed via Nginx
    deploy_orchestrator = docker.Container(
        "deploy-orchestrator-container",
        image=orchestrator_tag.apply(lambda tags: tags[0]),
        name="api-gateway",
        restart="always",
        envs=[
            "GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-service.json",
            f"GCP_PROJECT={project}",
            "GCP_LOCATION=us-central1",
            "RULE_AGENT_URL=http://rule-agent:9002",
            "COMBAT_AGENT_URL=http://combat-agent:9000",
        ],
        volumes=[
            docker.ContainerVolumeArgs(
                host_path="/srv/secrets",
                container_path="/secrets",
                read_only=False,
            ),
        ],
        networks_advanced=[
            docker.ContainerNetworksAdvancedArgs(
                name=docker_network.name,
            )
        ],
        opts=ResourceOptions(
            provider=docker_provider,
            depends_on=[deploy_vector_db, deploy_rule_agent, deploy_api_service],
        ),
    )

    return docker_provider, docker_network
