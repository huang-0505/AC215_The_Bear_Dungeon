import os
import pulumi
from create_instance import create_instance
from provision_instance import provision_instance
from setup_containers import setup_containers
from setup_webserver import setup_webserver

# Get project info and configuration
gcp_config = pulumi.Config("gcp")
project = gcp_config.require("project")
ssh_user = pulumi.Config("security").require("ssh_user")
zone = os.environ["GCP_ZONE"]

# Create GCP infrastructure
instance, instance_ip, connection, persistent_disk, network = create_instance()

# Provision the instance with system-level setup
configure_docker = provision_instance(connection, instance, ssh_user)

# Setup and deploy all application containers
docker_provider, docker_network = setup_containers(
    connection, configure_docker, project, instance_ip, ssh_user
)

# Setup and deploy Nginx webserver
restart_nginx = setup_webserver(connection, docker_provider, docker_network)

# Export references to stack
pulumi.export("instance_name", instance.name)
pulumi.export("instance_ip", instance_ip)
pulumi.export("zone", zone)
pulumi.export("persistent_disk_name", persistent_disk.name)
pulumi.export("ssh_user", ssh_user)
pulumi.export("ssh_command", instance_ip.apply(lambda ip: f"ssh {ssh_user}@{ip}"))
