"""
Sample Docker API responses for testing.

This module provides sample Docker API responses that can be used for testing
Docker-related functionality without requiring an actual Docker installation.
"""

# Sample container list response
sample_container_list = [
    {
        "Id": "container1",
        "Names": ["/mcp-filesystem"],
        "Image": "mcp/filesystem:latest",
        "ImageID": "sha256:abc123",
        "Command": "/bin/sh -c 'npm start'",
        "Created": 1625097600,
        "State": {
            "Status": "running",
            "Running": True,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "Dead": False,
            "Pid": 1234,
            "ExitCode": 0,
            "Error": "",
            "StartedAt": "2023-01-01T00:00:00Z",
            "FinishedAt": "0001-01-01T00:00:00Z"
        },
        "Ports": [
            {
                "IP": "0.0.0.0",
                "PrivatePort": 8080,
                "PublicPort": 8080,
                "Type": "tcp"
            }
        ],
        "Labels": {
            "com.example.mcp": "filesystem",
            "com.example.version": "1.0"
        },
        "Status": "Up 2 hours",
        "HostConfig": {
            "NetworkMode": "default"
        },
        "NetworkSettings": {
            "Networks": {
                "bridge": {
                    "IPAddress": "172.17.0.2",
                    "Gateway": "172.17.0.1",
                    "IPPrefixLen": 16,
                    "MacAddress": "02:42:ac:11:00:02"
                }
            }
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/var/run/docker.sock",
                "Destination": "/var/run/docker.sock",
                "Mode": "rw",
                "RW": True,
                "Propagation": "rprivate"
            }
        ]
    },
    {
        "Id": "container2",
        "Names": ["/mcp-github"],
        "Image": "mcp/github:latest",
        "ImageID": "sha256:def456",
        "Command": "/bin/sh -c 'npm start'",
        "Created": 1625097500,
        "State": {
            "Status": "running",
            "Running": True,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "Dead": False,
            "Pid": 5678,
            "ExitCode": 0,
            "Error": "",
            "StartedAt": "2023-01-01T00:00:00Z",
            "FinishedAt": "0001-01-01T00:00:00Z"
        },
        "Ports": [
            {
                "IP": "0.0.0.0",
                "PrivatePort": 8081,
                "PublicPort": 8081,
                "Type": "tcp"
            }
        ],
        "Labels": {
            "com.example.mcp": "github",
            "com.example.version": "1.0"
        },
        "Status": "Up 1 hour",
        "HostConfig": {
            "NetworkMode": "default"
        },
        "NetworkSettings": {
            "Networks": {
                "bridge": {
                    "IPAddress": "172.17.0.3",
                    "Gateway": "172.17.0.1",
                    "IPPrefixLen": 16,
                    "MacAddress": "02:42:ac:11:00:03"
                }
            }
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/var/run/docker.sock",
                "Destination": "/var/run/docker.sock",
                "Mode": "rw",
                "RW": True,
                "Propagation": "rprivate"
            }
        ]
    },
    {
        "Id": "container3",
        "Names": ["/mcp-postgres"],
        "Image": "mcp/postgres:latest",
        "ImageID": "sha256:ghi789",
        "Command": "/bin/sh -c 'npm start'",
        "Created": 1625097400,
        "State": {
            "Status": "exited",
            "Running": False,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "Dead": False,
            "Pid": 0,
            "ExitCode": 0,
            "Error": "",
            "StartedAt": "2023-01-01T00:00:00Z",
            "FinishedAt": "2023-01-01T01:00:00Z"
        },
        "Ports": [],
        "Labels": {
            "com.example.mcp": "postgres",
            "com.example.version": "1.0"
        },
        "Status": "Exited (0) 1 hour ago",
        "HostConfig": {
            "NetworkMode": "default"
        },
        "NetworkSettings": {
            "Networks": {
                "bridge": {
                    "IPAddress": "",
                    "Gateway": "",
                    "IPPrefixLen": 0,
                    "MacAddress": ""
                }
            }
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/var/run/docker.sock",
                "Destination": "/var/run/docker.sock",
                "Mode": "rw",
                "RW": True,
                "Propagation": "rprivate"
            }
        ]
    }
]

# Sample container inspect response
sample_container_inspect = {
    "Id": "container1",
    "Created": "2023-01-01T00:00:00Z",
    "Path": "/bin/sh",
    "Args": ["-c", "npm start"],
    "State": {
        "Status": "running",
        "Running": True,
        "Paused": False,
        "Restarting": False,
        "OOMKilled": False,
        "Dead": False,
        "Pid": 1234,
        "ExitCode": 0,
        "Error": "",
        "StartedAt": "2023-01-01T00:00:00Z",
        "FinishedAt": "0001-01-01T00:00:00Z"
    },
    "Image": "sha256:abc123",
    "ResolvConfPath": "/var/lib/docker/containers/container1/resolv.conf",
    "HostnamePath": "/var/lib/docker/containers/container1/hostname",
    "HostsPath": "/var/lib/docker/containers/container1/hosts",
    "LogPath": "/var/lib/docker/containers/container1/container1-json.log",
    "Name": "/mcp-filesystem",
    "RestartCount": 0,
    "Driver": "overlay2",
    "Platform": "linux",
    "MountLabel": "",
    "ProcessLabel": "",
    "AppArmorProfile": "",
    "ExecIDs": None,
    "HostConfig": {
        "Binds": ["/var/run/docker.sock:/var/run/docker.sock:rw"],
        "ContainerIDFile": "",
        "LogConfig": {
            "Type": "json-file",
            "Config": {}
        },
        "NetworkMode": "default",
        "PortBindings": {
            "8080/tcp": [
                {
                    "HostIp": "",
                    "HostPort": "8080"
                }
            ]
        },
        "RestartPolicy": {
            "Name": "no",
            "MaximumRetryCount": 0
        },
        "AutoRemove": False,
        "VolumeDriver": "",
        "VolumesFrom": None,
        "CapAdd": None,
        "CapDrop": None,
        "CgroupnsMode": "host",
        "Dns": [],
        "DnsOptions": [],
        "DnsSearch": [],
        "ExtraHosts": None,
        "GroupAdd": None,
        "IpcMode": "private",
        "Cgroup": "",
        "Links": None,
        "OomScoreAdj": 0,
        "PidMode": "",
        "Privileged": False,
        "PublishAllPorts": False,
        "ReadonlyRootfs": False,
        "SecurityOpt": None,
        "UTSMode": "",
        "UsernsMode": "",
        "ShmSize": 67108864,
        "Runtime": "runc",
        "ConsoleSize": [0, 0],
        "Isolation": "",
        "CpuShares": 0,
        "Memory": 0,
        "NanoCpus": 0,
        "CgroupParent": "",
        "BlkioWeight": 0,
        "BlkioWeightDevice": [],
        "BlkioDeviceReadBps": None,
        "BlkioDeviceWriteBps": None,
        "BlkioDeviceReadIOps": None,
        "BlkioDeviceWriteIOps": None,
        "CpuPeriod": 0,
        "CpuQuota": 0,
        "CpuRealtimePeriod": 0,
        "CpuRealtimeRuntime": 0,
        "CpusetCpus": "",
        "CpusetMems": "",
        "Devices": [],
        "DeviceCgroupRules": None,
        "DeviceRequests": None,
        "KernelMemory": 0,
        "KernelMemoryTCP": 0,
        "MemoryReservation": 0,
        "MemorySwap": 0,
        "MemorySwappiness": None,
        "OomKillDisable": False,
        "PidsLimit": None,
        "Ulimits": None,
        "CpuCount": 0,
        "CpuPercent": 0,
        "IOMaximumIOps": 0,
        "IOMaximumBandwidth": 0,
        "MaskedPaths": [
            "/proc/asound",
            "/proc/acpi",
            "/proc/kcore",
            "/proc/keys",
            "/proc/latency_stats",
            "/proc/timer_list",
            "/proc/timer_stats",
            "/proc/sched_debug",
            "/proc/scsi",
            "/sys/firmware"
        ],
        "ReadonlyPaths": [
            "/proc/bus",
            "/proc/fs",
            "/proc/irq",
            "/proc/sys",
            "/proc/sysrq-trigger"
        ]
    },
    "GraphDriver": {
        "Data": {
            "LowerDir": "/var/lib/docker/overlay2/abc123/diff",
            "MergedDir": "/var/lib/docker/overlay2/abc123/merged",
            "UpperDir": "/var/lib/docker/overlay2/abc123/diff",
            "WorkDir": "/var/lib/docker/overlay2/abc123/work"
        },
        "Name": "overlay2"
    },
    "Mounts": [
        {
            "Type": "bind",
            "Source": "/var/run/docker.sock",
            "Destination": "/var/run/docker.sock",
            "Mode": "rw",
            "RW": True,
            "Propagation": "rprivate"
        }
    ],
    "Config": {
        "Hostname": "container1",
        "Domainname": "",
        "User": "",
        "AttachStdin": False,
        "AttachStdout": False,
        "AttachStderr": False,
        "ExposedPorts": {
            "8080/tcp": {}
        },
        "Tty": False,
        "OpenStdin": False,
        "StdinOnce": False,
        "Env": [
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "NODE_VERSION=14.17.3",
            "YARN_VERSION=1.22.5"
        ],
        "Cmd": ["/bin/sh", "-c", "npm start"],
        "Image": "mcp/filesystem:latest",
        "Volumes": None,
        "WorkingDir": "/app",
        "Entrypoint": None,
        "OnBuild": None,
        "Labels": {
            "com.example.mcp": "filesystem",
            "com.example.version": "1.0"
        }
    },
    "NetworkSettings": {
        "Bridge": "",
        "SandboxID": "abc123",
        "HairpinMode": False,
        "LinkLocalIPv6Address": "",
        "LinkLocalIPv6PrefixLen": 0,
        "Ports": {
            "8080/tcp": [
                {
                    "HostIp": "0.0.0.0",
                    "HostPort": "8080"
                }
            ]
        },
        "SandboxKey": "/var/run/docker/netns/abc123",
        "SecondaryIPAddresses": None,
        "SecondaryIPv6Addresses": None,
        "EndpointID": "abc123",
        "Gateway": "172.17.0.1",
        "GlobalIPv6Address": "",
        "GlobalIPv6PrefixLen": 0,
        "IPAddress": "172.17.0.2",
        "IPPrefixLen": 16,
        "IPv6Gateway": "",
        "MacAddress": "02:42:ac:11:00:02",
        "Networks": {
            "bridge": {
                "IPAMConfig": None,
                "Links": None,
                "Aliases": None,
                "NetworkID": "abc123",
                "EndpointID": "abc123",
                "Gateway": "172.17.0.1",
                "IPAddress": "172.17.0.2",
                "IPPrefixLen": 16,
                "IPv6Gateway": "",
                "GlobalIPv6Address": "",
                "GlobalIPv6PrefixLen": 0,
                "MacAddress": "02:42:ac:11:00:02",
                "DriverOpts": None
            }
        }
    }
}

# Sample image list response
sample_image_list = [
    {
        "Id": "sha256:abc123",
        "RepoTags": ["mcp/filesystem:latest"],
        "RepoDigests": ["mcp/filesystem@sha256:abc123"],
        "Created": 1625097600,
        "Size": 100000000,
        "VirtualSize": 100000000,
        "SharedSize": 0,
        "Labels": {
            "com.example.mcp": "filesystem",
            "com.example.version": "1.0"
        },
        "Containers": 1
    },
    {
        "Id": "sha256:def456",
        "RepoTags": ["mcp/github:latest"],
        "RepoDigests": ["mcp/github@sha256:def456"],
        "Created": 1625097500,
        "Size": 120000000,
        "VirtualSize": 120000000,
        "SharedSize": 0,
        "Labels": {
            "com.example.mcp": "github",
            "com.example.version": "1.0"
        },
        "Containers": 1
    },
    {
        "Id": "sha256:ghi789",
        "RepoTags": ["mcp/postgres:latest"],
        "RepoDigests": ["mcp/postgres@sha256:ghi789"],
        "Created": 1625097400,
        "Size": 150000000,
        "VirtualSize": 150000000,
        "SharedSize": 0,
        "Labels": {
            "com.example.mcp": "postgres",
            "com.example.version": "1.0"
        },
        "Containers": 1
    }
]

# Sample image inspect response
sample_image_inspect = {
    "Id": "sha256:abc123",
    "RepoTags": ["mcp/filesystem:latest"],
    "RepoDigests": ["mcp/filesystem@sha256:abc123"],
    "Parent": "",
    "Comment": "",
    "Created": "2023-01-01T00:00:00Z",
    "Container": "container1",
    "ContainerConfig": {
        "Hostname": "container1",
        "Domainname": "",
        "User": "",
        "AttachStdin": False,
        "AttachStdout": False,
        "AttachStderr": False,
        "ExposedPorts": {
            "8080/tcp": {}
        },
        "Tty": False,
        "OpenStdin": False,
        "StdinOnce": False,
        "Env": [
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "NODE_VERSION=14.17.3",
            "YARN_VERSION=1.22.5"
        ],
        "Cmd": ["/bin/sh", "-c", "npm start"],
        "Image": "mcp/filesystem:latest",
        "Volumes": None,
        "WorkingDir": "/app",
        "Entrypoint": None,
        "OnBuild": None,
        "Labels": {
            "com.example.mcp": "filesystem",
            "com.example.version": "1.0"
        }
    },
    "DockerVersion": "20.10.0",
    "Author": "",
    "Config": {
        "Hostname": "container1",
        "Domainname": "",
        "User": "",
        "AttachStdin": False,
        "AttachStdout": False,
        "AttachStderr": False,
        "ExposedPorts": {
            "8080/tcp": {}
        },
        "Tty": False,
        "OpenStdin": False,
        "StdinOnce": False,
        "Env": [
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "NODE_VERSION=14.17.3",
            "YARN_VERSION=1.22.5"
        ],
        "Cmd": ["/bin/sh", "-c", "npm start"],
        "Image": "mcp/filesystem:latest",
        "Volumes": None,
        "WorkingDir": "/app",
        "Entrypoint": None,
        "OnBuild": None,
        "Labels": {
            "com.example.mcp": "filesystem",
            "com.example.version": "1.0"
        }
    },
    "Architecture": "amd64",
    "Os": "linux",
    "Size": 100000000,
    "VirtualSize": 100000000,
    "GraphDriver": {
        "Data": {
            "LowerDir": "/var/lib/docker/overlay2/abc123/diff",
            "MergedDir": "/var/lib/docker/overlay2/abc123/merged",
            "UpperDir": "/var/lib/docker/overlay2/abc123/diff",
            "WorkDir": "/var/lib/docker/overlay2/abc123/work"
        },
        "Name": "overlay2"
    },
    "RootFS": {
        "Type": "layers",
        "Layers": [
            "sha256:abc123",
            "sha256:def456",
            "sha256:ghi789"
        ]
    },
    "Metadata": {
        "LastTagTime": "2023-01-01T00:00:00Z"
    }
}

# Sample volume list response
sample_volume_list = [
    {
        "CreatedAt": "2023-01-01T00:00:00Z",
        "Driver": "local",
        "Labels": {
            "com.example.mcp": "filesystem",
            "com.example.version": "1.0"
        },
        "Mountpoint": "/var/lib/docker/volumes/mcp-filesystem-data/_data",
        "Name": "mcp-filesystem-data",
        "Options": {},
        "Scope": "local"
    },
    {
        "CreatedAt": "2023-01-01T00:00:00Z",
        "Driver": "local",
        "Labels": {
            "com.example.mcp": "github",
            "com.example.version": "1.0"
        },
        "Mountpoint": "/var/lib/docker/volumes/mcp-github-data/_data",
        "Name": "mcp-github-data",
        "Options": {},
        "Scope": "local"
    },
    {
        "CreatedAt": "2023-01-01T00:00:00Z",
        "Driver": "local",
        "Labels": {
            "com.example.mcp": "postgres",
            "com.example.version": "1.0"
        },
        "Mountpoint": "/var/lib/docker/volumes/mcp-postgres-data/_data",
        "Name": "mcp-postgres-data",
        "Options": {},
        "Scope": "local"
    }
]

# Sample volume inspect response
sample_volume_inspect = {
    "CreatedAt": "2023-01-01T00:00:00Z",
    "Driver": "local",
    "Labels": {
        "com.example.mcp": "filesystem",
        "com.example.version": "1.0"
    },
    "Mountpoint": "/var/lib/docker/volumes/mcp-filesystem-data/_data",
    "Name": "mcp-filesystem-data",
    "Options": {},
    "Scope": "local"
}

# Sample network list response
sample_network_list = [
    {
        "Name": "bridge",
        "Id": "net1",
        "Created": "2023-01-01T00:00:00Z",
        "Scope": "local",
        "Driver": "bridge",
        "EnableIPv6": False,
        "IPAM": {
            "Driver": "default",
            "Options": {},
            "Config": [
                {
                    "Subnet": "172.17.0.0/16",
                    "Gateway": "172.17.0.1"
                }
            ]
        },
        "Internal": False,
        "Attachable": False,
        "Ingress": False,
        "ConfigFrom": {
            "Network": ""
        },
        "ConfigOnly": False,
        "Containers": {
            "container1": {
                "Name": "mcp-filesystem",
                "EndpointID": "abc123",
                "MacAddress": "02:42:ac:11:00:02",
                "IPv4Address": "172.17.0.2/16",
                "IPv6Address": ""
            },
            "container2": {
                "Name": "mcp-github",
                "EndpointID": "def456",
                "MacAddress": "02:42:ac:11:00:03",
                "IPv4Address": "172.17.0.3/16",
                "IPv6Address": ""
            }
        },
        "Options": {
            "com.docker.network.bridge.default_bridge": "true",
            "com.docker.network.bridge.enable_icc": "true",
            "com.docker.network.bridge.enable_ip_masquerade": "true",
            "com.docker.network.bridge.host_binding_ipv4": "0.0.0.0",
            "com.docker.network.bridge.name": "docker0",
            "com.docker.network.driver.mtu": "1500"
        },
        "Labels": {}
    },
    {
        "Name": "host",
        "Id": "net2",
        "Created": "2023-01-01T00:00:00Z",
        "Scope": "local",
        "Driver": "host",
        "EnableIPv6": False,
        "IPAM": {
            "Driver": "default",
            "Options": {},
            "Config": []
        },
        "Internal": False,
        "Attachable": False,
        "Ingress": False,
        "ConfigFrom": {
            "Network": ""
        },
        "ConfigOnly": False,
        "Containers": {},
        "Options": {},
        "Labels": {}
    },
    {
        "Name": "none",
        "Id": "net3",
        "Created": "2023-01-01T00:00:00Z",
        "Scope": "local",
        "Driver": "null",
        "EnableIPv6": False,
        "IPAM": {
            "Driver": "default",
            "Options": {},
            "Config": []
        },
        "Internal": False,
        "Attachable": False,
        "Ingress": False,
        "ConfigFrom": {
            "Network": ""
        },
        "ConfigOnly": False,
        "Containers": {},
        "Options": {},
        "Labels": {}
    }
]

# Sample network inspect response
sample_network_inspect = {
    "Name": "bridge",
    "Id": "net1",
    "Created": "2023-01-01T00:00:00Z",
    "Scope": "local",
    "Driver": "bridge",
    "EnableIPv6": False,
    "IPAM": {
        "Driver": "default",
        "Options": {},
        "Config": [
            {
                "Subnet": "172.17.0.0/16",
                "Gateway": "172.17.0.1"
            }
        ]
    },
    "Internal": False,
    "Attachable": False,
    "Ingress": False,
    "ConfigFrom": {
        "Network": ""
    },
    "ConfigOnly": False,
    "Containers": {
        "container1": {
            "Name": "mcp-filesystem",
            "EndpointID": "abc123",
            "MacAddress": "02:42:ac:11:00:02",
            "IPv4Address": "172.17.0.2/16",
            "IPv6Address": ""
        },
        "container2": {
            "Name": "mcp-github",
            "EndpointID": "def456",
            "MacAddress": "02:42:ac:11:00:03",
            "IPv4Address": "172.17.0.3/16",
            "IPv6Address": ""
        }
    },
    "Options": {
        "com.docker.network.bridge.default_bridge": "true",
        "com.docker.network.bridge.enable_icc": "true",
        "com.docker.network.bridge.enable_ip_masquerade": "true",
        "com.docker.network.bridge.host_binding_ipv4": "0.0.0.0",
        "com.docker.network.bridge.name": "docker0",
        "com.docker.network.driver.mtu": "1500"
    },
    "Labels": {}
}
