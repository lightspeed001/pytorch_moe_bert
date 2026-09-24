# pytorch_moe_bert
## PyTorch Mixture of Experts (MOE) Model with BERT Scale

```mermaid
graph TD
    A[Input] --> B[BERT Encoder]
    B --> C[Sequence Output]
    C --> D[Gating Network]
    C --> E[Expert 1]
    C --> F[Expert 2]
    C --> G[Expert 3]
    D -->|Weights| H[Weighted Sum]
    E --> H
    F --> H
    G --> H
    H --> I[Output]

```

---

### Key Features :spiral_notepad:

**1. BERT Scale Architecture**

- Uses BERT's transformer architecture as the base
- Configurable hidden size (768 by default)
- Multiple attention heads (12 by default)

**2. Mixture of Experts**

- 8 experts by default (configurable)
- Each expert is a 2-layer feed-forward network
- Top-2 routing by default (configurable)
- Noisy gating mechanism for better training stability

**3. ONNX Export:**

- Includes example code to export the model to ONNX format
- Handles dynamic axes for varible sequence lengths

**4. Training Considerations**

- Proper weight initialization
- Dropout for regularization
- GELU activations

**Notes** :notebook:
    
> For production serving, you may want to add:
- Batch processing support
- Proper tokenization handling
- Pre/post processing steps

> The model assumes a classification task
> Model can be scaled up by increasing the number of layers, experts, or hidden dimensions

### Model Serving: Kubernetes (k3s) :cloud:

__K3s Production Architecture__

```mermaid
graph TD
    subgraph "K3s Cluster"
        A[Master Node] --> B[Worker Node 1]
        A --> C[Worker Node 2]
        A --> D[Worker Node 3]
        B --> E[Model Pod 1]
        B --> F[Model Pod 2]
        C --> G[Model Pod 3]
        D --> H[Model Pod 4]
    end

    subgraph "External Services"
        I[Load Balancer] -->|Traffic| J[Ingress Controller]
        J --> E
        J --> F
        J --> G
        J --> H
        K[Persistent Storage] --> L[Model PVC]
        L --> E
        L --> F
        L --> G
        L --> H
    end

    subgraph "Monitoring Stack"
        M[Prometheus] --> N[Grafana]
        M --> O[AlertManager]
        E -->|Metrics| M
        F -->|Metrics| M
        G -->|Metrics| M
        H -->|Metrics| M
    end

    style A fill:#f9f,stroke:#333
    style I fill:#bbf,stroke:#333
    style M fill:#9f9,stroke:#333

```

__Components__:

- Master Node: Runs control plane components (API server, scheduler, controller manager)
- Worker Nodes: Run the model pods and other workloads.
- Load Balancer: Distributes traffic to ingress controller.
- Ingress Controller: Routes external traffic to model pods.
- Persistent Storage: Provides shared storage for model files.
- Monitoring Stack: Collects metrics from modal pods.

__Detailed K3s Networking__

```mermaid
graph TD
    subgraph "Networking Stack"
        A[External Traffic] --> B[MetalLB]
        B --> C[Traefik Ingress]
        C --> D[Model Service]
        D --> E[Model Pods]
        F[Internal Traffic] --> G[ClusterIP Services]
        G --> E
    end

    subgraph "Storage"
        H[Longhorn] --> I[Model PVC]
        I --> E
    end

    subgraph "Monitoring"
        J[Prometheus] --> K[Grafana]
        E -->|Metrics| J
    end

    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style J fill:#9f9,stroke:#333


```
__Components__:

- MetalLB: Provides LoadBalancer services in bare-metal k3s
- Traefik Ingress: Handles external HTTP/HTTPS traffic
- ClusterIP Services: Internal service discovery
- Longhorn: Distributed block storage for persistent volumes
- Prometheus/Grafana: Monitoring stack

### Model Serving: Kubernetes (GKE) :cloud:

__GKE Production Architecture__

```mermaid
graph TD
    subgraph "GKE Cluster"
        A[Control Plane] --> B[Node Pool 1: CPU]
        A --> C[Node Pool 2: GPU]
        B --> D[Model Pod 1]
        B --> E[Model Pod 2]
        C --> F[Model Pod 3]
        C --> G[Model Pod 4]
    end

    subgraph "GCP Services"
        H[Cloud Load Balancer] --> I[Ingress Controller]
        I --> D
        I --> E
        I --> F
        I --> G
        J[Persistent Disk] --> K[Model PVC]
        K --> D
        K --> E
        K --> F
        K --> G
        L[Cloud Monitoring] --> M[Metrics]
        D -->|Metrics| L
        E -->|Metrics| L
        F -->|Metrics| L
        G -->|Metrics| L
    end

    subgraph "Security"
        N[Workload Identity] --> O[Service Account]
        O --> D
        O --> E
        O --> F
        O --> G
        P[Binary Authorization] --> Q[Image Verification]
    end

    style A fill:#f9f,stroke:#333
    style H fill:#bbf,stroke:#333
    style L fill:#9f9,stroke:#333
    style N fill:#ff9,stroke:#333

```
__Components__:

- Control Plane: Managed by GCP (highly available)
> Node Pools:
- CPU pool for general workloads
- GPU pool for model serving

- Cloud Load Balancer: GCP-managed ingress
- Persistent Disk: GCP-native monitoring solution
- Workload Identity: Secure access to GCP services
- Binary Authorization: Image verification for security

__Detailed GKE Networking__

```mermaid
graph TD
    subgraph "GKE Networking"
        A[External Traffic] --> B[GCP Load Balancer]
        B --> C[Ingress Controller]
        C --> D[Model Service]
        D --> E[Model Pods]
        F[Internal Traffic] --> G[ClusterIP Services]
        G --> E
    end

    subgraph "GCP Networking"
        H[VPC Network] --> I[Subnet]
        I --> J[Node Pools]
        J --> E
    end

    subgraph "Security"
        K[Network Policies] --> L[Pod-to-Pod Traffic]
        M[Firewall Rules] --> N[External Access]
    end

    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style K fill:#ff9,stroke:#333

```
__Components__:

- GCP Load Balancer: Global HTTP/S load balancer
- Ingress Controller: GKE-managed ingress
- VPC Network: GCP Virtual Private Cloud
- Network Policies: Kubernetes-native network segmentation
- Firewall Rules: GCP firewall for additional security




