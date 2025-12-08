from pulumi import ResourceOptions
from pulumi_command import remote


def provision_instance(connection, instance, ssh_user):
    """
    Provision the VM instance with system-level setup:
    - Format and mount persistent disk
    - Disable unattended upgrades
    - Update system packages
    - Install base dependencies
    - Install Docker and configure it
    - Install Python packages for Docker management

    Args:
        connection: SSH connection configuration
        instance: The VM instance resource
        ssh_user: SSH user for the instance

    Returns:
        remote.Command: The last provisioning command (for dependency chaining)
    """
    # Provision VM: Format and mount persistent disk
    format_disk = remote.Command(
        "format-persistent-disk",
        connection=connection,
        create="""
            # Format disk only if it doesn't have a filesystem
            sudo sh -c 'blkid -o value -s TYPE /dev/disk/by-id/google-persistent-disk-1 || mkfs.ext4 /dev/disk/by-id/google-persistent-disk-1'
        """,
        opts=ResourceOptions(depends_on=[instance]),
    )

    create_mount_dir = remote.Command(
        "create-mount-directory",
        connection=connection,
        create="""
            sudo mkdir -p /mnt/disk-1
            sudo chown root:root /mnt/disk-1
            sudo chmod 0755 /mnt/disk-1
        """,
        opts=ResourceOptions(depends_on=[format_disk]),
    )

    mount_disk = remote.Command(
        "mount-persistent-disk",
        connection=connection,
        create="""
            # Add to fstab if not already present
            if ! grep -q '/mnt/disk-1' /etc/fstab; then
                echo '/dev/disk/by-id/google-persistent-disk-1 /mnt/disk-1 ext4 discard,defaults 0 2' | sudo tee -a /etc/fstab
            fi
            # Mount the disk
            sudo mount -a
        """,
        opts=ResourceOptions(depends_on=[create_mount_dir]),
    )

    # Provision VM: Disable unattended upgrades and update system
    disable_unattended_upgrades = remote.Command(
        "disable-unattended-upgrades",
        connection=connection,
        create="""
            sudo systemctl disable --now apt-daily.timer || true
            sudo systemctl disable --now apt-daily-upgrade.timer || true
            sudo systemctl daemon-reload
            sudo systemd-run --property="After=apt-daily.service apt-daily-upgrade.service" --wait /bin/true
        """,
        opts=ResourceOptions(depends_on=[mount_disk]),
    )

    # Provision VM: Update apt and install dependencies
    update_system = remote.Command(
        "update-system",
        connection=connection,
        create="""
            sudo apt-get update
            sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
        """,
        opts=ResourceOptions(depends_on=[disable_unattended_upgrades]),
    )

    install_dependencies = remote.Command(
        "install-dependencies",
        connection=connection,
        create="""
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
                apt-transport-https \
                ca-certificates \
                curl \
                gnupg-agent \
                software-properties-common \
                python3-setuptools \
                python3-pip \
                lsb-release
        """,
        opts=ResourceOptions(depends_on=[update_system]),
    )

    # Provision VM: Install Docker
    install_docker = remote.Command(
        "install-docker",
        connection=connection,
        create="""
            # Get distributor ID and release
            DISTRO=$(lsb_release -is | tr '[:upper:]' '[:lower:]')
            RELEASE=$(lsb_release -cs)

            # Add Docker GPG key
            curl -fsSL https://download.docker.com/linux/$DISTRO/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

            # Add Docker repository
            echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/$DISTRO $RELEASE stable" | \
                sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

            # Install Docker
            sudo apt-get update
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io
        """,
        opts=ResourceOptions(depends_on=[install_dependencies]),
    )

    # Provision VM: Install Python packages for Docker management
    install_pip_packages = remote.Command(
        "install-pip-packages",
        connection=connection,
        create="""
            sudo pip3 install requests docker
        """,
        opts=ResourceOptions(depends_on=[install_docker]),
    )

    # Provision VM: Configure Docker
    configure_docker = remote.Command(
        "configure-docker",
        connection=connection,
        create=f"""
            # Create docker group
            sudo groupadd docker || true

            # Add current user to docker group
            sudo usermod -aG docker {ssh_user}

            # Start and enable Docker service first
            sudo systemctl start docker
            sudo systemctl enable docker

            # Docker socket permissions
            sudo chmod 666 /var/run/docker.sock

            # Install docker-credential-gcr (gcloud credential helper for Docker)
            # This is already included in gcloud SDK, just need to ensure it's in PATH
            gcloud components install docker-credential-gcr --quiet || true

            # Authenticate Docker with GCP (using instance service account)
            # The VM already has a service account attached, so gcloud will use that automatically
            # Configure for both user and root
            gcloud auth configure-docker --quiet
            gcloud auth configure-docker us-docker.pkg.dev --quiet
            gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

            # Also configure for root user (since Pulumi Docker provider may run as root)
            sudo gcloud auth configure-docker --quiet
            sudo gcloud auth configure-docker us-docker.pkg.dev --quiet
            sudo gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

            # Verify gcloud is using the instance service account
            echo "Current gcloud account:"
            gcloud auth list

            # Test access to Artifact Registry
            echo "Testing access to Artifact Registry..."
            gcloud artifacts repositories list --location=us-central1 || echo "Note: List may fail if repository is in different project"
        """,
        opts=ResourceOptions(depends_on=[install_pip_packages]),
    )

    return configure_docker
