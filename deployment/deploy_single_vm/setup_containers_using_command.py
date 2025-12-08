import pulumi
from pulumi import ResourceOptions
from pulumi_command import remote


def setup_containers(connection, configure_docker, project):
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
    vector_db_cli_tag = images_stack.get_output("dnd-master-vector-db-cli-tags")

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
            gcloud auth activate-service-account --key-file /srv/secrets/gcp-service.json
            gcloud auth configure-docker us-docker.pkg.dev --quiet
        """,
        opts=ResourceOptions(depends_on=[upload_service_account]),
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
        opts=ResourceOptions(depends_on=[move_secrets]),
    )

    # Create Docker network
    create_network = remote.Command(
        "create-docker-network",
        connection=connection,
        create="""
            docker network create appnetwork || true
        """,
        opts=ResourceOptions(depends_on=[create_dirs]),
    )

    # Deploy containers
    # Frontend
    deploy_frontend = remote.Command(
        "deploy-frontend-container",
        connection=connection,
        create=frontend_tag.apply(
            lambda tags: f"""
                docker pull {tags[0]}
                docker stop frontend || true
                docker rm frontend || true
                docker run -d \
                    --name frontend \
                    --network appnetwork \
                    -p 3000:3000 \
                    --restart always \
                    {tags[0]}
            """
        ),
        opts=ResourceOptions(depends_on=[create_network]),
    )

    # Vector DB
    deploy_vector_db = remote.Command(
        "deploy-vector-db-container",
        connection=connection,
        create="""
            docker pull chromadb/chroma:latest
            docker stop vector-db || true
            docker rm vector-db || true
            docker run -d \
                --name vector-db \
                --network appnetwork \
                -p 8000:8000 \
                -e IS_PERSISTENT=TRUE \
                -e ANONYMIZED_TELEMETRY=FALSE \
                -v /mnt/disk-1/chromadb:/chroma/chroma \
                --restart always \
                chromadb/chroma
        """,
        opts=ResourceOptions(depends_on=[create_network]),
    )

    # Load vector DB data
    load_vector_db = remote.Command(
        "load-vector-db-data",
        connection=connection,
        create=vector_db_cli_tag.apply(
            lambda tags: f"""
                docker pull {tags[0]}
                docker run --rm \
                    -e GCP_PROJECT="{project}" \
                    -e CHROMADB_HOST="vector-db" \
                    -e CHROMADB_PORT="8000" \
                    -e GOOGLE_APPLICATION_CREDENTIALS="/secrets/gcp-service.json" \
                    -v /srv/secrets:/secrets \
                    --network appnetwork \
                    {tags[0]} \
                    cli.py --download --load --chunk_type recursive-split
            """
        ),
        opts=ResourceOptions(depends_on=[deploy_vector_db]),
    )

    # API Service
    deploy_api_service = remote.Command(
        "deploy-api-service-container",
        connection=connection,
        create=api_service_tag.apply(
            lambda tags: f"""
                docker pull {tags[0]}
                docker stop api-service || true
                docker rm api-service || true
                docker run -d \
                    --name api-service \
                    --network appnetwork \
                    -p 9000:9000 \
                    -e GOOGLE_APPLICATION_CREDENTIALS="/secrets/gcp-service.json" \
                    -e GCP_PROJECT="{project}" \
                    -e GCS_BUCKET_NAME="dnd-master-models" \
                    -e CHROMADB_HOST="vector-db" \
                    -e CHROMADB_PORT="8000" \
                    -v /mnt/disk-1/persistent:/persistent \
                    -v /srv/secrets:/secrets \
                    --restart always \
                    {tags[0]}
            """
        ),
        opts=ResourceOptions(depends_on=[load_vector_db]),
    )

    return deploy_api_service
