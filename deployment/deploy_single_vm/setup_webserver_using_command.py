import hashlib
import pulumi
from pulumi import ResourceOptions
from pulumi_command import remote


def file_checksum(path: str) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        path: Path to the file

    Returns:
        str: Hexadecimal checksum string
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def setup_webserver(connection, deploy_api_service):
    """
    Setup and deploy Nginx webserver:
    - Create nginx configuration directory
    - Upload nginx configuration file with checksum tracking
    - Deploy nginx container
    - Restart nginx to ensure configuration is loaded

    Args:
        connection: SSH connection configuration
        deploy_api_service: The API service deployment command (dependency)

    Returns:
        remote.Command: The nginx restart command (for dependency chaining)
    """
    # Create nginx config directory
    create_nginx_conf_dir = remote.Command(
        "create-nginx-conf-directory",
        connection=connection,
        create="""
            sudo mkdir -p /conf/nginx
            sudo chmod 0755 /conf/nginx
        """,
        opts=ResourceOptions(depends_on=[deploy_api_service]),
    )

    # Shared asset for nginx config
    nginx_conf_asset = pulumi.FileAsset("../nginx-conf/nginx/nginx.conf")

    # Compute checksum for nginx config
    checksum = file_checksum("../nginx-conf/nginx/nginx.conf")

    # Copy nginx configuration file triggered if checksum changes
    upload_nginx_conf = remote.CopyToRemote(
        "upload-nginx-conf",
        connection=connection,
        source=nginx_conf_asset,
        remote_path="/tmp/nginx.conf",
        triggers=[nginx_conf_asset, checksum],
        opts=ResourceOptions(depends_on=[create_nginx_conf_dir]),
    )

    # Move nginx config to final location
    move_nginx_conf = remote.Command(
        "move-nginx-conf",
        connection=connection,
        create="""
            sudo mv /tmp/nginx.conf /conf/nginx/nginx.conf
            sudo chmod 0644 /conf/nginx/nginx.conf
            sudo chown root:root /conf/nginx/nginx.conf
        """,
        triggers=[nginx_conf_asset, checksum],
        opts=ResourceOptions(depends_on=[upload_nginx_conf]),
    )

    # Deploy nginx container
    deploy_nginx = remote.Command(
        "deploy-nginx-container",
        connection=connection,
        create="""
            docker pull nginx:stable
            docker stop nginx || true
            docker rm nginx || true
            docker run -d \
                --name nginx \
                --network appnetwork \
                -p 80:80 \
                -p 443:443 \
                -v /conf/nginx/nginx.conf:/etc/nginx/nginx.conf:ro \
                --restart always \
                nginx:stable
        """,
        triggers=[nginx_conf_asset],
        opts=ResourceOptions(depends_on=[move_nginx_conf]),
    )

    # Restart nginx to ensure config is loaded
    restart_nginx = remote.Command(
        "restart-nginx-container",
        connection=connection,
        create="""
            docker container restart nginx
        """,
        triggers=[nginx_conf_asset, checksum],
        opts=ResourceOptions(depends_on=[deploy_nginx, upload_nginx_conf]),
    )

    return restart_nginx
