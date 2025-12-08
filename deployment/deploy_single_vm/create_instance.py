import os
import pulumi
import pulumi_gcp as gcp
from pulumi import ResourceOptions
from pulumi_command import remote


def load_ssh_key_pair():
    """Load SSH keys for remote access"""
    with open("/secrets/ssh-key-deployment", "r") as private_key_file:
        private_key = private_key_file.read()
    with open("/secrets/ssh-key-deployment.pub", "r") as public_key_file:
        public_key = public_key_file.read()
    return private_key, public_key


def create_instance():
    """
    Create GCP infrastructure including network, firewall rules, disk, and VM instance.

    Returns:
        tuple: (instance, instance_ip, connection, persistent_disk, network)
    """
    # Get project info and configuration
    gcp_config = pulumi.Config("gcp")
    project = gcp_config.require("project")
    ssh_user = pulumi.Config("security").require("ssh_user")
    gcp_service_account_email = pulumi.Config("security").require(
        "gcp_service_account_email"
    )
    location = os.environ["GCP_REGION"]
    zone = os.environ["GCP_ZONE"]

    # Configuration variables
    persistent_disk_name = "dnd-master-demo-disk"
    persistent_disk_size = 50
    machine_instance_name = "dnd-master-demo"
    machine_type = "n2d-standard-2"
    machine_disk_size = 50

    # Load SSH keys
    private_key, ssh_public_key = load_ssh_key_pair()

    # Create a new network with auto-created subnetworks
    network = gcp.compute.Network(
        "dnd-master-network",
        auto_create_subnetworks=True,
    )

    # Create firewall rule for HTTP traffic
    firewall_http = gcp.compute.Firewall(
        "allow-http",
        network=network.self_link,
        allows=[
            gcp.compute.FirewallAllowArgs(
                protocol="tcp",
                ports=["80"],
            )
        ],
        source_ranges=["0.0.0.0/0"],
        target_tags=["http-server"],
    )

    # Create firewall rule for HTTPS traffic
    firewall_https = gcp.compute.Firewall(
        "allow-https",
        network=network.self_link,
        allows=[
            gcp.compute.FirewallAllowArgs(
                protocol="tcp",
                ports=["443"],
            )
        ],
        source_ranges=["0.0.0.0/0"],
        target_tags=["https-server"],
    )

    # Create firewall rule for SSH
    firewall_ssh = gcp.compute.Firewall(
        "allow-ssh",
        network=network.self_link,
        allows=[
            gcp.compute.FirewallAllowArgs(
                protocol="tcp",
                ports=["22"],
            )
        ],
        source_ranges=["0.0.0.0/0"],
    )

    # Create persistent disk for data storage
    persistent_disk = gcp.compute.Disk(
        persistent_disk_name,
        zone=zone,
        size=persistent_disk_size,
        type="pd-standard",
    )

    # Resolve the latest Ubuntu 22.04 LTS image from the public Ubuntu project
    ubuntu_image = gcp.compute.get_image(
        family="ubuntu-2204-lts",
        project="ubuntu-os-cloud",
    )

    # Create the VM instance
    instance = gcp.compute.Instance(
        machine_instance_name,
        machine_type=machine_type,
        zone=zone,
        boot_disk=gcp.compute.InstanceBootDiskArgs(
            initialize_params=gcp.compute.InstanceBootDiskInitializeParamsArgs(
                image=ubuntu_image.self_link,
                size=machine_disk_size,
            ),
        ),
        # Attach persistent disk
        attached_disks=[
            gcp.compute.InstanceAttachedDiskArgs(
                source=persistent_disk.id,
            )
        ],
        network_interfaces=[
            gcp.compute.InstanceNetworkInterfaceArgs(
                network=network.id,
                # Let GCP auto-assign an ephemeral IP
                access_configs=[
                    gcp.compute.InstanceNetworkInterfaceAccessConfigArgs(
                        network_tier="STANDARD"
                    )
                ],
            )
        ],
        metadata={"ssh-keys": f"{ssh_user}:{ssh_public_key}"},
        tags=["http-server", "https-server"],
        service_account=gcp.compute.InstanceServiceAccountArgs(
            email=gcp_service_account_email,
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
            ],
        ),
        opts=ResourceOptions(
            depends_on=[firewall_http, firewall_https, firewall_ssh, persistent_disk]
        ),
    )

    # Get the VM's public IP address (dynamically assigned)
    instance_ip = instance.network_interfaces.apply(
        lambda interfaces: interfaces[0].access_configs[0].nat_ip
    )

    # Create SSH connection configuration for remote commands
    connection = remote.ConnectionArgs(
        host=instance_ip,
        user=ssh_user,
        private_key=private_key,
    )

    return instance, instance_ip, connection, persistent_disk, network
