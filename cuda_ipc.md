```
The official NVIDIA documentation for CUDA Interprocess Communication (IPC), which details the APIs and best practices, is found within the CUDA Programming Guide and the CUDA Runtime API Reference Manual. 
Key Documentation Links
Interprocess Communication section of the CUDA Programming Guide: This is the primary resource for understanding the concepts and general usage of CUDA IPC, including the traditional (legacy) and Virtual Memory Management (VMM) APIs.
CUDA Runtime API Documentation: This reference manual provides detailed descriptions of the specific functions, such as cudaIpcGetMemHandle, cudaIpcOpenMemHandle, cudaIpcGetEventHandle, and cudaIpcOpenEventHandle.
CUDA Samples: The simpleIPC sample code provides a practical example of how to implement IPC. 
Addressing Specific Questions
Based on the documentation and community discussions, here are insights into your specific concerns:
Alternative approaches to sharing GPU memory:
NVSHMEM is a library that offers a more comprehensive approach for complex inter-process interactions and synchronization, potentially simpler than rolling your own with low-level IPC APIs.
The CUDA Virtual Memory Management (VMM) APIs (cuMemCreate, cuMemMap, etc.) offer a more modern and flexible way to create IPC-shareable allocations compared to the legacy cudaMalloc approach.
Async allocator workarounds:
Allocations made by cudaMallocAsync are managed by a driver-level memory pool, and it is not directly compatible with the legacy CUDA IPC APIs (cudaIpcGetMemHandle will not work).
To use IPC, you must use memory allocated via the traditional cudaMalloc() or the VMM APIs which support IPC.
In PyTorch, you can selectively use different allocators for specific regions of code using torch.cuda.MemPool, or disable the async allocator entirely via an environment variable (PYTORCH_CUDA_ALLOC_CONF=backend:cudaMalloc) to ensure compatibility with IPC.
IPC handle lifecycle:
The original memory allocation must remain alive for as long as any process is using a view of it.
The receiving process must call cudaIpcOpenMemHandle() to get a process-local pointer and cudaIpcCloseMemHandle() when it is done.
It is unsafe to mix different API versions or versions of the CUDA toolkit during the lifetime of resources like IPC handles.
Memory pinning requirements:
For memory allocated with the legacy cudaMalloc (or PyTorch's default synchronous allocator), no special allocation flags are explicitly needed for the allocation itself, but the memory must be on a CUDA-capable device.
For IPC events, you must create them with the cudaEventInterprocess flag.
Multi-GPU considerations:
Communication between different GPU contexts on the same or different physical devices is possible via IPC handles. You will need to manage the device context in each process correctly.
Ensure that peer-to-peer access is enabled between the relevant GPUs for optimal performance, using cudaDeviceEnablePeerAccess. 
The documentation should help you migrate to a compatible memory allocation method to resolve the async allocator conflict.
```
```
4.15. Interprocess Communication
Communication between multiple GPUs managed by different host processes is supported through the use of interprocess communication (IPC) APIs and IPC-shareable memory buffers, by creating process-portable handles that are subsequently used to obtain process-local device pointers to the device memory on peer GPUs.

Any device memory pointer or event handle created by a host thread can be directly referenced by any other thread within the same process. However, device pointers or event handles are not valid outside the process that created them, and therefore cannot be directly referenced by threads belonging to a different process. To access device memory and CUDA events across processes, an application must use CUDA Interprocess Communication (IPC) or Virtual Memory Management APIs to create process-portable handles that can be shared with other processes using standard host operating system IPC mechanisms, e.g., interprocess shared memory or files. Once the process-portable handles have been exchanged between processes, process-local device pointers must be obtained from the handles using CUDA IPC or VMM APIs. Process-local device pointers can then be used just as they would within a single process.

The same kind of portable-handle approach used for IPC within a single-node and single operating system instance is also used for peer-to-peer communication among the GPUs in multi-node NVLink-connected clusters. In the multi-node case, communicating GPUs are managed by processes running within independent operating system instances on each cluster node, requiring additional abstraction above the level of operating system instances. Multi-node peer communication is achieved by creating and exchanging so-called “fabric” handles between multi-node GPU peers, and by then obtaining process-local device pointers within the participating processes and operating system instances corresponding to the multi-node ranks.

See below (single-node CUDA IPC) and ref::virtual-memory-management for the specific APIs used to establish and exchange process-portable and node and operating system instance-portable handles that are used to obtain process-local device pointers for GPU communication.

Note

There are individual advantages and limitations associated with the use of the CUDA IPC APIs and Virtual Memory Management (VMM) APIs when used for IPC.

The CUDA IPC API is only currently supported on Linux platforms.

The CUDA Virtual Memory Management APIs permit per-allocation control over peer accessibility and sharing at memory allocation time, but require the use of the CUDA Driver API.

4.15.1. IPC using the Legacy Interprocess Communication API
To share device memory pointers and events across processes, an application must use the CUDA Interprocess Communication API, which is described in detail in the reference manual. The IPC API permits an application to get the IPC handle for a given device memory pointer using cudaIpcGetMemHandle(). A CUDA IPC handle can be passed to another process using standard host operating system IPC mechanisms, e.g., interprocess shared memory or files. cudaIpcOpenMemHandle() uses the IPC handle to retrieve a valid device pointer that can be used within the other process. Event handles can be shared using similar entry points.

An example of using the IPC API is where a single primary process generates a batch of input data, making the data available to multiple secondary processes without requiring regeneration or copying.

Note

The IPC API is only supported on Linux.

Note that the IPC API is not supported for cudaMallocManaged allocations.

Applications using CUDA IPC to communicate with each other should be compiled, linked, and run with the same CUDA driver and runtime.

Allocations made by cudaMalloc() may be sub-allocated from a larger block of memory for performance reasons. In such case, CUDA IPC APIs will share the entire underlying memory block which may cause other sub-allocations to be shared, which can potentially lead to information disclosure between processes. To prevent this behavior, it is recommended to only share allocations with a 2MiB aligned size.

Only the IPC events-sharing APIs are supported on L4T and embedded Linux Tegra devices with compute capability 7.x and higher. The IPC memory-sharing APIs are not supported on Tegra platforms.

4.15.2. IPC using the Virtual Memory Management API
The CUDA Virtual Memory Management API allows the creation of IPC-shareable memory allocations, and it supports multiple operating systems by virtue of operating-system specific IPC handle data structures.
```
```
In the CUDA programming model, memory allocation calls (such as cudaMalloc()) return a memory address that in GPU memory. The address can be used with any CUDA API or inside a device kernel. Developers can enable peer device access to that memory allocations by using cudaEnablePeerAccess. By doing so, kernels on different devices can access the same data. However, all past and future user allocations are also mapped to the target peer device. This can lead to users unintentionally paying a runtime cost for mapping all cudaMalloc allocations to peer devices. In most situations, applications communicate by sharing only a few allocations with another device. It is usually not necessary to map all allocations to all devices. In addition, extending this approach to multi-node settings becomes inherently difficult.

CUDA provides a virtual memory management (VMM) API to give developers explicit, low-level control over this process.

Virtual memory allocation, a complex process managed by the operating system and the Memory Management Unit (MMU), works in two key stages. First, the OS reserves a contiguous range of virtual addresses for a program without assigning any physical memory. Then, when the program attempts to use that memory for the first time, the OS commits the virtual addresses, assigning physical storage to the virtual pages as needed.

CUDA’s VMM API brings a similar concept to GPU memory management by allowing developers to explicitly reserve a virtual address range and then later map it to physical GPU memory. With VMM, applications can specifically choose certain allocations to be accessible by other devices.

The VMM API lets complex applications to manage memory more efficiently across multiple GPUs (and CPU cores). By enabling manual control over memory reservation, mapping, and access permissions, the VMM API enables advanced techniques like fine-grained data sharing, zero-copy transfers, and custom memory allocators. The CUDA VMM API expose fine grained control to the user for managing the GPU memory in applications.

Developers can benefit from the VMM API in several key ways:

Fine-grained control over virtual and physical memory management, allowing allocation and mapping of non-contiguous physical memory chunks to contiguous virtual address spaces. This helps reduce GPU memory fragmentation and improve memory utilization, especially for large workloads like deep neural network training.

Efficient memory allocation and deallocation by separating the reservation of virtual address space from the physical memory allocation. Developers can reserve large virtual memory regions and map physical memory on demand without costly memory copies or reallocations, leading to performance improvements in dynamic data structures and variable-sized memory allocations.

The ability to grow GPU memory allocations dynamically without needing to copy and reallocate all data, similar to how realloc or std::vector works in CPU memory management. This supports more flexible and efficient GPU memory use patterns.

Enhancements to developer productivity and application performance by providing low-level APIs that allow building sophisticated memory allocators and cache management systems, such as dynamically managing key-value caches in large language models, improving throughput and latency.

The CUDA VMM API is highly valuable in distributed multi-GPU settings as it enables efficient memory sharing and access across multiple GPUs. By decoupling virtual addresses from physical memory, the API allows developers to create a unified virtual address space where data can be dynamically mapped to different GPUs. This optimizes memory usage and reduces data transfer overhead. For instance, NVIDIA’s libraries like NCCL, and NVShmem actively uses VMM.

In summary, the CUDA VMM API gives developers advanced tools for fine-tuned, efficient, flexible, and scalable GPU memory management beyond traditional malloc-like abstractions, which is important for high-performance and large-memory applications

Note

The suite of APIs described in this section require a system that supports UVA. See The Virtual Memory Management APIs.

4.16.1. Preliminaries
4.16.1.1. Definitions
Fabric Memory: Fabric memory refers to memory that is accessible over a high-speed interconnect fabric such as NVIDIA’s NVLink and NVSwitch. This fabric provides a memory coherence and high-bandwidth communication layer between multiple GPUs or nodes, enabling them to share memory efficiently as if the memory is attached to a unified fabric rather than isolated on individual devices.

CUDA 12.4 and later have a VMM allocation handle type CU_MEM_HANDLE_TYPE_FABRIC. On supported platforms and provided the NVIDIA IMEX daemon is running, this allocation handle type enables sharing allocations not only intra-node with any communication mechanism, e.g. MPI, but also inter-node. This allows GPUs in a multi-node NVLink system to map the memory of all other GPUs part of the same NVLink fabric even if they are in different nodes.

Memory Handles: In VMM, handles are opaque identifiers that represent physical memory allocations. These handles are central to managing memory in the low-level CUDA VMM API. They enable flexible control over physical memory objects that can be mapped into virtual address spaces. A handle uniquely identifies a physical memory allocation. Handles serve as an abstract reference to memory resources without exposing direct pointers. Handles allow operations like exporting and importing memory across processes or devices, facilitating memory sharing and virtualization.

IMEX Channels: The name IMEX stands for internode memory exchange and is part of NVIDIA’s solution for GPU-to-GPU communication across different nodes. IMEX channels are a GPU driver feature that provides user-based memory isolation in multi-user or multi-node environments within an IMEX domain. IMEX channels serve as a security and isolation mechanism.

IMEX channels are directly related to the fabric handle and has to be enabled in multi-node GPU communication. When a GPU allocates memory and wants to make it accessible to a GPU on a different node, it first needs to export a handle to that memory. The IMEX channel is used during this export process to generate a secure fabric handle that can only be imported by a remote process with the correct channel access.

Unicast Memory Access: Unicast memory access in the context of VMM API refers to the controlled, direct mapping and access of physical memory to a unique virtual address range by a specific device or process. Instead of broadcasting access to multiple devices, unicast memory access means that a particular GPU device is granted explicit read/write permissions to a reserved virtual address range that maps to a physical memory allocation.

Multicast Memory Access: Multicast memory access in the context of the VMM API refers to the capability for a single physical memory allocation or region to be mapped simultaneously to multiple devices’ virtual address spaces using a multicast mechanism. This allows data to be efficiently shared in a one-to-many fashion across multiple GPUs, reducing redundant data transfers and improving communication efficiency. NVIDIA’s CUDA VMM API supports creating a multicast object that binds together physical memory allocations from multiple devices.

4.16.1.2. Query for Support
Applications should query for feature support before attempting to use them, as their availability can vary depending on the GPU architecture, driver version, and specific software libraries being used. The following sections detail how to programmatically check for the necessary support.

VMM Support Before attempting to use VMM APIs, applications must ensure that the devices they want to use support CUDA virtual memory management. The following code sample shows querying for VMM support:

int deviceSupportsVmm;
CUresult result = cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
if (deviceSupportsVmm != 0) {
    // `device` supports Virtual Memory Management
}
Fabric Memory Support: Before attempting to use fabric memory, applications must ensure that the devices they want to use support fabric memory. The following code sample shows querying for fabric memory support:

int deviceSupportsFabricMem;
CUresult result = cuDeviceGetAttribute(&deviceSupportsFabricMem, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
if (deviceSupportsFabricMem != 0) {
    // `device` supports Fabric Memory
}
Aside from using CU_MEM_HANDLE_TYPE_FABRIC as handle type and not requiring OS native mechanisms for inter-process communication to exchange sharable handles, there is no difference in using fabric memory compared to other allocation handle types.

IMEX Channels Support Within an IMEX domain, IMEX channels enable secure memory sharing in multi-user environments. The NVIDIA driver implements this by creating a character device, nvidia-caps-imex-channels. To use fabric handle-based sharing, users should verify two things:

First, applications must verify that this device exists under /proc/devices:

# cat /proc/devices | grep nvidia
195 nvidia
195 nvidiactl
234 nvidia-caps-imex-channels
509 nvidia-nvswitch

The nvidia-caps-imex-channels device should have a major number (e.g., 234).
Second, for two CUDA processes (an exporter and an importer) to share memory, they must both have access to the same IMEX channel file. These files, such as /dev/nvidia-caps-imex-channels/channel0, are nodes that represent individual IMEX channels. System administrators must create these files, for example, using the mknod() command.

# mknod /dev/nvidia-caps-imex-channels/channelN c <major_number> 0

This command creates channelN using the major number obtained from
/proc/devices.
Note

By default, the driver can create channel0 if the NVreg_CreateImexChannel0 module parameter is specified.

Multicast Object Support: Before attempting to use multicast objects, applications must ensure that the devices they want to use support them. The following code sample shows querying for multicast object support:

int deviceSupportsMultiCast;
CUresult result = cuDeviceGetAttribute(&deviceSupportsMultiCast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
if (deviceSupportsMultiCast != 0) {
    // `device` supports Multicast Objects
}
4.16.2. API Overview
The VMM API provides developers with granular control over virtual memory management. VMM, being a very low-level API, requires use of the CUDA Driver API directly. This versatile API can be used in both single-node and multi-node environments.

To use VMM effectively, developers must have a solid grasp of a few key concepts in memory management: - Knowledge of the operating system’s virtual memory fundamentals, including how it handles pages and address spaces - An understanding of memory hierarchy and hardware characteristics is necessary - Familiarity with inter-process communication (IPC) methods, such as sockets or message passing, - A basic knowledge of security for memory access rights

VMM Usage Overview Diagram
Figure 52 VMM Usage Overview. This diagram outlines the series of steps required for VMM utilization. The process begins by evaluating the environmental setup. Based on this assessment, the user must make a critical initial decision: whether to utilize fabric memory handles or OS-specific handles. A distinct series of subsequent steps must be taken based on the initial handle choice. However, the final memory management operations—specifically mapping, reserving, and setting access rights of the allocated memory—are identical to the type of handle that was selected.
The VMM API workflow involves a sequence of steps for memory management, with a key focus on sharing memory between different devices or processes. Initially, a developer must allocate physical memory on the source device. To facilitate sharing, the VMM API utilizes handles to convey necessary information to the target device or process. The user must export a handle for sharing, which can be either an OS-specific handle or a fabric-specific handle. OS-specific handles are limited to inter-process communication on a single node, while fabric-specific handles offer greater versatility and can be used in both single-node and multi-node environments. It’s important to note that using fabric-specific handles requires the enablement of IMEX channels.

Once the handle is exported, it must be shared with the receiving process or processes using an inter-process communication protocol, with the choice of method left to the developer. The receiving process then uses the VMM API to import the handle. After the handle has been successfully exported, shared, and imported, both the source and target processes must reserve virtual address space where the allocated physical memory will be mapped. The final step is to set the memory access rights for each device, ensuring proper permissions are established. This entire process, including both handle approaches, is further detailed in the accompanying figure.

4.16.3. Unicast Memory Sharing
Sharing GPU memory can happen on one machine with multiple GPUs or across a network of machines. The process follows these steps:

Allocate and Export: A CUDA program on one GPU allocates memory and gets a sharable handle for it.

Share and Import: The handle is then sent to other programs on the node using IPC, MPI, or NCCL etc. In the receiving GPUs, the CUDA driver imports the handle, creates the necessary memory objects

Reserve and Map: The driver creates a mapping from the program’s Virtual Address (VA) to the GPU’s Physical Address (PA) to its network Fabric Address (FA).

Access Rights: Setting access rights for the allocation.

Releasing the Memory: Freeing all allocations when program ends its execution.

Unicast Memory Sharing Example
Figure 53 Unicast Memory Sharing Example
4.16.3.1. Allocate and Export
Allocating Physical Memory The first step in memory allocation using virtual memory management APIs is to create a physical memory chunk that will provide a backing for the allocation. In order to allocate physical memory, applications must use the cuMemCreate API. The allocation created by this function does not have any device or host mappings. The function argument CUmemGenericAllocationHandle describes the properties of the memory to allocate such as the location of the allocation, if the allocation is going to be shared to another process (or graphics APIs), or the physical attributes of the memory to be allocated. Users must ensure the requested allocation’s size is aligned to appropriate granularity. Information regarding an allocation’s granularity requirements can be queried using cuMemGetAllocationGranularity.


OS-Specific Handle (Linux)
CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleType = handleType;

    size_t granularity = 0;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    // Allocate physical memory
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);

    return allocHandle;
}

Fabric Handle
Note

The memory allocated by cuMemCreate is referenced by the CUmemGenericAllocationHandle it returns. Note that this reference is not a pointer and its memory is not accessible yet.

Note

Properties of the allocation handle can be queried using cuMemGetAllocationPropertiesFromHandle.

Exporting Memory Handle The CUDA virtual memory management API expose a new mechanism for interprocess communication using handles to exchange necessary information about the allocation and physical address space. One can export handles for OS-specific IPC or fabric-specific IPC. OS-specific IPC handles can only be used on a single-node setup. Fabric-specific handles can be used on a single or multi node setups.


OS-Specific Handle (Linux)
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
CUmemGenericAllocationHandle handle = allocatePhysicalMemory(0, 1<<21);
int fd;
cuMemExportToShareableHandle(&fd, handle, handleType, 0);

Fabric Handle
Note

OS-specific handles require all processes to be part of the same OS.

Note

Fabric-specific handles require IMEX channels to be enabled by sysadmin.

The memMapIpcDrv sample can be used as an example for using IPC with VMM allocations.

4.16.3.2. Share and Import
Sharing Memory Handle Once the handle is exported, it must be shared with the receiving process or processes using an inter-process communication protocol. The developer is free to use any method for sharing the handle. The specific IPC method used depends on the application’s design and environment. Common methods include OS-specific inter-process sockets and distributed message passing. Using OS-specific IPC offers high-performance transfer, but is limited to processes on the same machine and not portable. Fabric-specific IPC is simpler and more portable. However, fabric-specific IPC requires system-level support. The chosen method must securely and reliably transfer the handle data to the target process so it can be used to import the memory and establish a valid mapping. The flexibility in choosing the IPC method allows the VMM API to be integrated into a wide range of system architectures, from single-node applications to distributed, multi-node setups. In the following code snippets, we’ll provide examples for sharing and receiving handles using both socket programming and MPI.


Send: OS-Specific IPC (Linux)
int ipcSendShareableHandle(int socket, int fd, pid_t process) {
    struct msghdr msg;
    struct iovec iov[1];

    union {
        struct cmsghdr cm;
        char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int)) * sizeof(char);
    control_un.control = (char*) malloc(sizeof_control);

    struct cmsghdr *cmptr;
    ssize_t readResult;
    struct sockaddr_un cliaddr;
    socklen_t len = sizeof(cliaddr);

    // Construct client address to send this SHareable handle to
    memset(&cliaddr, 0, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char temp[20];
    sprintf(temp, "%s%u", "/tmp/", process);
    strcpy(cliaddr.sun_path, temp);
    len = sizeof(cliaddr);

    // Send corresponding shareable handle to the client
    int sendfd = fd;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

    msg.msg_name = (void *)&cliaddr;
    msg.msg_namelen = sizeof(struct sockaddr_un);

    iov[0].iov_base = (void *)"";
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    ssize_t sendResult = sendmsg(socket, &msg, 0);
    if (sendResult <= 0) {
        perror("IPC failure: Sending data over socket failed");
        free(control_un.control);
        return -1;
    }

    free(control_un.control);
    return 0;
}

Send: OS-Specific IPC (WIN)

Send: Fabric IPC

Receive: OS-Specific IPC (Linux)

Receive: OS-Specific IPC (WIN)

Receive: Fabric IPC
Importing Memory Handle Again, the user can import handles for OS-specific IPC or fabric-specific IPC. OS-specific IPC handles can only be used on a single-node. Fabric-specific handles can be used for single or multi node.


OS-Specific Handle (Linux)
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
cuMemImportFromShareableHandle(handle, (void*) &fd, handleType);

Fabric Handle
4.16.3.3. Reserve and Map
Reserving a Virtual Address Range

Since notions of address and memory are distinct in VMM, applications must carve out an address range that can hold the memory allocations made by cuMemCreate. The address range reserved must be at least as large as the sum of the sizes of all the physical memory allocations the user plans to place in them.

Applications can reserve a virtual address range by passing appropriate parameters to cuMemAddressReserve. The address range obtained will not have any device or host physical memory associated with it. The reserved virtual address range can be mapped to memory chunks belonging to any device in the system, thus providing the application a continuous VA range backed and mapped by memory belonging to different devices. Applications are expected to return the virtual address range back to CUDA using cuMemAddressFree. Users must ensure that the entire VA range is unmapped before calling cuMemAddressFree. These functions are conceptually similar to mmap and munmap on Linux or VirtualAlloc AND VirtualFree on Windows. The following code snippet illustrates the usage for the function:

CUdeviceptr ptr;
// `ptr` holds the returned start of virtual address range reserved.
CUresult result = cuMemAddressReserve(&ptr, size, 0, 0, 0); // alignment = 0 for default alignment
Mapping Memory

The allocated physical memory and the carved out virtual address space from the previous two sections represent the memory and address distinction introduced by the VMM APIs. For the allocated memory to be useable, the user must map the memory to the address space. The address range obtained from cuMemAddressReserve and the physical allocation obtained from cuMemCreate or cuMemImportFromShareableHandle must be associated with each other by using cuMemMap.

Users can associate allocations from multiple devices to reside in contiguous virtual address ranges as long as they have carved out enough address space. To decouple the physical allocation and the address range, users must unmap the address of the mapping with cuMemUnmap. Users can map and unmap memory to the same address range as many times as they want, so long as they ensure that they don’t attempt to create mappings on VA range reservations that are already mapped. The following code snippet illustrates the usage for the function:

CUdeviceptr ptr;
// `ptr`: address in the address range previously reserved by cuMemAddressReserve.
// `allocHandle`: CUmemGenericAllocationHandle obtained by a previous call to cuMemCreate.
CUresult result = cuMemMap(ptr, size, 0, allocHandle, 0);
4.16.3.4. Access Rights
CUDA’s virtual memory management APIs enable applications to explicitly protect their VA ranges with access control mechanisms. Mapping the allocation to a region of the address range using cuMemMap does not make the address accessible, and would result in a program crash if accessed by a CUDA kernel. Users must specifically select access control using the cuMemSetAccess function on source and accessing devices. This allows or restricts access for specific devices to a mapped address range. The following code snippet illustrates the usage for the function:

void setAccessOnDevice(int device, CUdeviceptr ptr, size_t size) {
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Make the address accessible
    cuMemSetAccess(ptr, size, &accessDesc, 1);
}
The access control mechanism exposed with VMM allows users to be explicit about which allocations they want to share with other peer devices on the system. As specified earlier, cudaEnablePeerAccess forces all prior and future allocations made with cudaMalloc to be mapped to the target peer device. This can be convenient in many cases as user doesn’t have to worry about tracking the mapping state of every allocation to every device in the system. But this approach has performance implications. With access control at allocation granularity, VMM allows peer mappings with minimal overhead.

The vectorAddMMAP sample can be used as an example for using the Virtual Memory Management APIs.

4.16.3.5. Releasing the Memory
To release the allocated memory and address space, both the source and target processes should use cuMemUnmap, cuMemRelease, and cuMemAddressFree functions in that order. The cuMemUnmap function un-maps a previously mapped memory region from an address range, effectively detaching the physical memory from the reserved virtual address space. Next, cuMemRelease deallocates the physical memory that was previously created, returning it to the system. Finally, cuMemAddressFree frees a virtual address range that was previously reserved, making it available for future use. This specific order ensures a clean and complete deallocation of both the physical memory and the virtual address space.

cuMemUnmap(ptr, size);
cuMemRelease(handle);
cuMemAddressFree(ptr, size);
Note

In the OS-specific case, the exported handle must be closed using fclose. This step is not applicable to the fabric-based case.

4.16.4. Multicast Memory Sharing
The Multicast Object Management APIs provide a way for the application to create multicast objects and, in combination with the Virtual Memory Management APIs described above, allow applications to leverage NVLink SHARP on supported NVLink connected GPUs connected with NVSwitch. NVLink SHARP allows CUDA applications to leverage in-fabric computing to accelerate operations like broadcast and reductions between GPUs connected with NVSwitch. For this to work, multiple NVLink connected GPUs form a multicast team and each GPU from the team backs up a multicast object with physical memory. So a multicast team of N GPUs has N physical replicas of a multicast object, each local to one participating GPU. The multimem PTX instructions using mappings of multicast objects work with all replicas of the multicast object.

To work with multicast objects, an application needs to

Query multicast support

Create a multicast handle with cuMulticastCreate.

Share the multicast handle with all processes that control a GPU which should participate in a multicast team. This works with cuMemExportToShareableHandle as described above.

Add all GPUs that should participate in the multicast team with cuMulticastAddDevice.

For each participating GPU, bind physical memory allocated with cuMemCreate as described above to the multicast handle. All devices need to be added to the multicast team before binding memory on any device.

Reserve an address range, map the multicast handle and set access rights as described above for regular unicast mappings. Unicast and multicast mappings to the same physical memory are possible. See the Virtual Aliasing Support section above on how to ensure consistency between multiple mappings to the same physical memory.

Use the multimem PTX instructions with the multicast mappings.

The multi_node_p2p example in the Multi GPU Programming Models GitHub repository contains a complete example using fabric memory including multicast objects to leverage NVLink SHARP. Please note that this example is for developers of libraries like NCCL or NVSHMEM. It shows how higher-level programming models like NVSHMEM work internally within a (multi-node) NVLink domain. Application developers generally should use the higher-level MPI, NCCL, or NVSHMEM interfaces instead of this API.

4.16.4.1. Allocating Multicast Objects
Multicast objects can be created with cuMulticastCreate:

CUmemGenericAllocationHandle createMCHandle(int numDevices, size_t size) {
    CUmemAllocationProp mcProp = {};
    mcProp.numDevices = numDevices;
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC; // or on single node CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    size_t granularity = 0;
    cuMulticastGetGranularity(&granularity, &mcProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    mcProp.size = padded_size;

    // Create Multicast Object this has no devices and no physical memory associated yet
    CUmemGenericAllocationHandle mcHandle;
    cuMulticastCreate(&mcHandle, &mcProp);

    return mcHandle;
}
4.16.4.2. Add Devices to Multicast Objects
Devices can be added to a multicast team with cuMulticastAddDevice:

cuMulticastAddDevice(&mcHandle, device);
This step needs to be completed on all processes controlling devices that participate in a multicast team before memory on any device is bound to the multicast object.

4.16.4.3. Bind Memory to Multicast Objects
After a multicast object has been created and all participating devices have been added to the multicast object it needs to be backed with physical memory allocated with cuMemCreate for each device:

cuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, 0 /*flags*/);
4.16.4.4. Use Multicast Mappings
To use multicast mappings in CUDA C++, it is necessary to use the multimem PTX instructions with inline PTX:

__global__ void all_reduce_norm_barrier_kernel(float* l2_norm,
                                               float* partial_l2_norm_mc,
                                               unsigned int* arrival_counter_uc, unsigned int* arrival_counter_mc,
                                               const unsigned int expected_count) {
    assert( 1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z );
    float l2_norm_sum = 0.0;
#if __CUDA_ARCH__ >= 900

    // atomic reduction to all replicas
    // this can be conceptually thought of as __threadfence_system(); atomicAdd_system(arrival_counter_mc, 1);
    cuda::ptx::multimem_red(cuda::ptx::release_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, arrival_counter_mc, n);

    // Need a fence between Multicast (mc) and Unicast (uc) access to the same memory `arrival_counter_uc` and `arrival_counter_mc`:
    // - fence.proxy instructions establish an ordering between memory accesses that may happen through different proxies
    // - Value .alias of the .proxykind qualifier refers to memory accesses performed using virtually aliased addresses to the same memory location.
    // from https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar
    cuda::ptx::fence_proxy_alias();

    // spin wait with acquire ordering on UC mapping till all peers have arrived in this iteration
    // Note: all ranks need to reach another barrier after this kernel, such that it is not possible for the barrier to be unblocked by an
    // arrival of a rank for the next iteration if some other rank is slow.
    cuda::atomic_ref<unsigned int,cuda::thread_scope_system> ac(arrival_counter_uc);
    while (expected_count > ac.load(cuda::memory_order_acquire));

    // Atomic load reduction from all replicas. It does not provide ordering so it can be relaxed.
    asm volatile ("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(l2_norm_sum) : "l"(partial_l2_norm_mc) : "memory");

#else
    #error "ERROR: multimem instructions require compute capability 9.0 or larger."
#endif

    *l2_norm = std::sqrt(l2_norm_sum);
}
4.16.5. Advanced Configuration
4.16.5.1. Memory Type
VMM also provides a mechanism for applications to allocate special types of memory that certain devices may support. With cuMemCreate, applications can specify memory type requirements using the CUmemAllocationProp::allocFlags to opt-in to specific memory features. Applications must ensure that the requested memory type is supported by the device.

4.16.5.2. Compressible Memory
Compressible memory can be used to accelerate accesses to data with unstructured sparsity and other compressible data patterns. Compression can save DRAM bandwidth, L2 read bandwidth, and L2 capacity depending on the data. Applications that want to allocate compressible memory on devices that support compute data compression can do so by setting CUmemAllocationProp::allocFlags::compressionType to CU_MEM_ALLOCATION_COMP_GENERIC. Users must query if device supports Compute Data Compression by using CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED. The following code snippet illustrates querying compressible memory support cuDeviceGetAttribute.

int compressionSupported = 0;
cuDeviceGetAttribute(&compressionSupported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device);
On devices that support compute data compression, users must opt in at allocation time as shown below:

prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
For a variety of reasons such as limited hardware resources, the allocation may not have compression attributes. To verify that the flags worked, the user query the properties of the allocated memory using cuMemGetAllocationPropertiesFromHandle.

CUmemAllocationProp allocationProp = {};
cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);

if (allocationProp.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC)
{
    // Obtained compressible memory allocation
}
4.16.5.3. Virtual Aliasing Support
The virtual memory management APIs provide a way to create multiple virtual memory mappings or “proxies” to the same allocation using multiple calls to cuMemMap with different virtual addresses. This is called virtual aliasing. Unless otherwise noted in the PTX ISA, writes to one proxy of the allocation are considered inconsistent and incoherent with any other proxy of the same memory until the writing device operation (grid launch, memcpy, memset, and so on) completes. Grids present on the GPU prior to a writing device operation but reading after the writing device operation completes are also considered to have inconsistent and incoherent proxies.

For example, the following snippet is considered undefined, assuming device pointers A and B are virtual aliases of the same memory allocation:

__global__ void foo(char *A, char *B) {
  *A = 0x1;
  printf("%d\n", *B);    // Undefined behavior!  *B can take on either
// the previous value or some value in-between.
}
The following is defined behavior, assuming these two kernels are ordered monotonically (by streams or events).

__global__ void foo1(char *A) {
  *A = 0x1;
}

__global__ void foo2(char *B) {
  printf("%d\n", *B);    // *B == *A == 0x1 assuming foo2 waits for foo1
// to complete before launching
}

cudaMemcpyAsync(B, input, size, stream1);    // Aliases are allowed at
// operation boundaries
foo1<<<1,1,0,stream1>>>(A);                  // allowing foo1 to access A.
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);
foo2<<<1,1,0,stream2>>>(B);
cudaStreamWaitEvent(stream3, event);
cudaMemcpyAsync(output, B, size, stream3);  // Both launches of foo2 and
                                            // cudaMemcpy (which both
                                            // read) wait for foo1 (which writes)
                                            // to complete before proceeding
If accessing same allocation through different “proxies” is required in the same kernel, a fence.proxy.alias can be used between the two accesses. The above example can thus be made legal with inline PTX assembly:

__global__ void foo(char *A, char *B) {
  *A = 0x1;
  cuda::ptx::fence_proxy_alias();
  printf("%d\n", *B);    // *B == *A == 0x1
}
4.16.5.4. OS-Specific Handle Details for IPC
With cuMemCreate, users have can indicate at allocation time that they have earmarked a particular allocation for inter-process communication or graphics interop purposes. Applications can do this by setting CUmemAllocationProp::requestedHandleTypes to a platform-specific field. On Windows, when CUmemAllocationProp::requestedHandleTypes is set to CU_MEM_HANDLE_TYPE_WIN32 applications must also specify an LPSECURITYATTRIBUTES attribute in CUmemAllocationProp::win32HandleMetaData. This security attribute defines the scope of which exported allocations may be transferred to other processes.

Users must ensure they query for support of the requested handle type before attempting to export memory allocated with cuMemCreate. The following code snippet illustrates query for handle type support in a platform-specific way.

int deviceSupportsIpcHandle;
#if defined(__linux__)
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
#else
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, device));
#endif
Users should set the CUmemAllocationProp::requestedHandleTypes appropriately as shown below:

#if defined(__linux__)
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
    prop.win32HandleMetaData = // Windows specific LPSECURITYATTRIBUTES attribute.
#endif
```
```
6.14. Virtual Memory Management
This section describes the virtual memory management functions of the low-level CUDA driver application programming interface.

Functions
CUresult cuMemAddressFree ( CUdeviceptr ptr, size_t size )
Free an address range reservation.
CUresult cuMemAddressReserve ( CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags )
Allocate an address range reservation.
CUresult cuMemCreate ( CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags )
Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.
CUresult cuMemExportToShareableHandle ( void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags )
Exports an allocation to a requested shareable handle type.
CUresult cuMemGetAccess ( unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr )
Get the access flags set for the given location and ptr.
CUresult cuMemGetAllocationGranularity ( size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option )
Calculates either the minimal or recommended granularity.
CUresult cuMemGetAllocationPropertiesFromHandle ( CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle )
Retrieve the contents of the property structure defining properties for this handle.
CUresult cuMemImportFromShareableHandle ( CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType )
Imports an allocation from a requested shareable handle type.
CUresult cuMemMap ( CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags )
Maps an allocation handle to a reserved virtual address range.
CUresult cuMemMapArrayAsync ( CUarrayMapInfo* mapInfoList, unsigned int  count, CUstream hStream )
Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.
CUresult cuMemRelease ( CUmemGenericAllocationHandle handle )
Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate.
CUresult cuMemRetainAllocationHandle ( CUmemGenericAllocationHandle* handle, void* addr )
Given an address addr, returns the allocation handle of the backing memory allocation.
CUresult cuMemSetAccess ( CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count )
Set the access flags for each location specified in desc for the given virtual address range.
CUresult cuMemUnmap ( CUdeviceptr ptr, size_t size )
Unmap the backing memory of a given address range.
Functions
CUresult cuMemAddressFree ( CUdeviceptr ptr, size_t size )
Free an address range reservation.
Parameters
ptr
- Starting address of the virtual address range to free
size
- Size of the virtual address region to free
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

Description
Frees a virtual address range reserved by cuMemAddressReserve. The size must match what was given to memAddressReserve and the ptr given must match what was returned from memAddressReserve.

See also:

cuMemAddressReserve

CUresult cuMemAddressReserve ( CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags )
Allocate an address range reservation.
Parameters
ptr
- Resulting pointer to start of virtual address range allocated
size
- Size of the reserved virtual address range requested
alignment
- Alignment of the reserved virtual address range requested
addr
- Hint address for the start of the address range
flags
- Currently unused, must be zero
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

Description
Reserves a virtual address range based on the given parameters, giving the starting address of the range in ptr. This API requires a system that supports UVA. The size and address parameters must be a multiple of the host page size and the alignment must be a power of two or zero for default alignment. If addr is 0, then the driver chooses the address at which to place the start of the reservation whereas when it is non-zero then the driver treats it as a hint about where to place the reservation.

See also:

cuMemAddressFree

CUresult cuMemCreate ( CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags )
Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.
Parameters
handle
- Value of handle returned. All operations on this allocation are to be performed using this handle.
size
- Size of the allocation requested
prop
- Properties of the allocation to create.
flags
- flags for future use, must be zero now.
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

Description
This creates a memory allocation on the target device specified through the prop structure. The created allocation will not have any device or host mappings. The generic memory handle for the allocation can be mapped to the address space of calling process via cuMemMap. This handle cannot be transmitted directly to other processes (see cuMemExportToShareableHandle). On Windows, the caller must also pass an LPSECURITYATTRIBUTE in prop to be associated with this handle which limits or allows access to this handle for a recipient process (see CUmemAllocationProp::win32HandleMetaData for more). The size of this allocation must be a multiple of the the value given via cuMemGetAllocationGranularity with the CU_MEM_ALLOC_GRANULARITY_MINIMUM flag. To create a CPU allocation that doesn't target any specific NUMA nodes, applications must set CUmemAllocationProp::CUmemLocation::type to CU_MEM_LOCATION_TYPE_HOST. CUmemAllocationProp::CUmemLocation::id is ignored for HOST allocations. HOST allocations are not IPC capable and CUmemAllocationProp::requestedHandleTypes must be 0, any other value will result in CUDA_ERROR_INVALID_VALUE. To create a CPU allocation targeting a specific host NUMA node, applications must set CUmemAllocationProp::CUmemLocation::type to CU_MEM_LOCATION_TYPE_HOST_NUMA and CUmemAllocationProp::CUmemLocation::id must specify the NUMA ID of the CPU. On systems where NUMA is not available CUmemAllocationProp::CUmemLocation::id must be set to 0. Specifying CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT as the CUmemLocation::type will result in CUDA_ERROR_INVALID_VALUE.

Applications that intend to use CU_MEM_HANDLE_TYPE_FABRIC based memory sharing must ensure: (1) `nvidia-caps-imex-channels` character device is created by the driver and is listed under /proc/devices (2) have at least one IMEX channel file accessible by the user launching the application.

When exporter and importer CUDA processes have been granted access to the same IMEX channel, they can securely share memory.

The IMEX channel security model works on a per user basis. Which means all processes under a user can share memory if the user has access to a valid IMEX channel. When multi-user isolation is desired, a separate IMEX channel is required for each user.

These channel files exist in /dev/nvidia-caps-imex-channels/channel* and can be created using standard OS native calls like mknod on Linux. For example: To create channel0 with the major number from /proc/devices users can execute the following command: `mknod /dev/nvidia-caps-imex-channels/channel0 c <major number>=""> 0`

If CUmemAllocationProp::allocFlags::usage contains CU_MEM_CREATE_USAGE_TILE_POOL flag then the memory allocation is intended only to be used as backing tile pool for sparse CUDA arrays and sparse CUDA mipmapped arrays. (see cuMemMapArrayAsync).


Note:
Note that this function may also return error codes from previous, asynchronous launches.

See also:

cuMemRelease, cuMemExportToShareableHandle, cuMemImportFromShareableHandle

CUresult cuMemExportToShareableHandle ( void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags )
Exports an allocation to a requested shareable handle type.
Parameters
shareableHandle
- Pointer to the location in which to store the requested handle type
handle
- CUDA handle for the memory allocation
handleType
- Type of shareable handle requested (defines type and size of the shareableHandle output parameter)
flags
- Reserved, must be zero
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

Description
Given a CUDA memory handle, create a shareable memory allocation handle that can be used to share the memory with other processes. The recipient process can convert the shareable handle back into a CUDA memory handle using cuMemImportFromShareableHandle and map it with cuMemMap. The implementation of what this handle is and how it can be transferred is defined by the requested handle type in handleType

Once all shareable handles are closed and the allocation is released, the allocated memory referenced will be released back to the OS and uses of the CUDA handle afterward will lead to undefined behavior.

This API can also be used in conjunction with other APIs (e.g. Vulkan, OpenGL) that support importing memory from the shareable type

See also:

cuMemImportFromShareableHandle

CUresult cuMemGetAccess ( unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr )
Get the access flags set for the given location and ptr.
Parameters
flags
- Flags set for this location
location
- Location in which to check the flags for
ptr
- Address in which to check the access flags for
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

Description
See also:

cuMemSetAccess

CUresult cuMemGetAllocationGranularity ( size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option )
Calculates either the minimal or recommended granularity.
Parameters
granularity
Returned granularity.
prop
Property for which to determine the granularity for
option
Determines which granularity to return
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

Description
Calculates either the minimal or recommended granularity for a given allocation specification and returns it in granularity. This granularity can be used as a multiple for alignment, size, or address mapping.

See also:

cuMemCreate, cuMemMap

CUresult cuMemGetAllocationPropertiesFromHandle ( CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle )
Retrieve the contents of the property structure defining properties for this handle.
Parameters
prop
- Pointer to a properties structure which will hold the information about this handle
handle
- Handle which to perform the query on
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

Description
See also:

cuMemCreate, cuMemImportFromShareableHandle

CUresult cuMemImportFromShareableHandle ( CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType )
Imports an allocation from a requested shareable handle type.
Parameters
handle
- CUDA Memory handle for the memory allocation.
osHandle
- Shareable Handle representing the memory allocation that is to be imported.
shHandleType
- handle type of the exported handle CUmemAllocationHandleType.
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

Description
If the current process cannot support the memory described by this shareable handle, this API will error as CUDA_ERROR_NOT_SUPPORTED.

If shHandleType is CU_MEM_HANDLE_TYPE_FABRIC and the importer process has not been granted access to the same IMEX channel as the exporter process, this API will error as CUDA_ERROR_NOT_PERMITTED.


Note:
Importing shareable handles exported from some graphics APIs(VUlkan, OpenGL, etc) created on devices under an SLI group may not be supported, and thus this API will return CUDA_ERROR_NOT_SUPPORTED. There is no guarantee that the contents of handle will be the same CUDA memory handle for the same given OS shareable handle, or the same underlying allocation.

See also:

cuMemExportToShareableHandle, cuMemMap, cuMemRelease

CUresult cuMemMap ( CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags )
Maps an allocation handle to a reserved virtual address range.
Parameters
ptr
- Address where memory will be mapped.
size
- Size of the memory mapping.
offset
- Offset into the memory represented by
handle from which to start mapping
Note: currently must be zero.
handle
- Handle to a shareable memory
flags
- flags for future use, must be zero now.
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_ILLEGAL_STATE

Description
Maps bytes of memory represented by handle starting from byte offset to size to address range [addr, addr + size]. This range must be an address reservation previously reserved with cuMemAddressReserve, and offset + size must be less than the size of the memory allocation. Both ptr, size, and offset must be a multiple of the value given via cuMemGetAllocationGranularity with the CU_MEM_ALLOC_GRANULARITY_MINIMUM flag. If handle represents a multicast object, ptr, size and offset must be aligned to the value returned by cuMulticastGetGranularity with the flag CU_MULTICAST_MINIMUM_GRANULARITY. For best performance however, it is recommended that ptr, size and offset be aligned to the value returned by cuMulticastGetGranularity with the flag CU_MULTICAST_RECOMMENDED_GRANULARITY.

When handle represents a multicast object, this call may return CUDA_ERROR_ILLEGAL_STATE if the system configuration is in an illegal state. In such cases, to continue using multicast, verify that the system configuration is in a valid state and all required driver daemons are running properly.

Please note calling cuMemMap does not make the address accessible, the caller needs to update accessibility of a contiguous mapped VA range by calling cuMemSetAccess.

Once a recipient process obtains a shareable memory handle from cuMemImportFromShareableHandle, the process must use cuMemMap to map the memory into its address ranges before setting accessibility with cuMemSetAccess.

cuMemMap can only create mappings on VA range reservations that are not currently mapped.


Note:
Note that this function may also return error codes from previous, asynchronous launches.

See also:

cuMemUnmap, cuMemSetAccess, cuMemCreate, cuMemAddressReserve, cuMemImportFromShareableHandle

CUresult cuMemMapArrayAsync ( CUarrayMapInfo* mapInfoList, unsigned int  count, CUstream hStream )
Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.
Parameters
mapInfoList
- List of CUarrayMapInfo
count
- Count of CUarrayMapInfo in mapInfoList
hStream
- Stream identifier for the stream to use for map or unmap operations
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE

Description
Performs map or unmap operations on subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays. Each operation is specified by a CUarrayMapInfo entry in the mapInfoList array of size count. The structure CUarrayMapInfo is defined as follow:

‎     typedef struct CUarrayMapInfo_st {
              CUresourcetype resourceType;                   
              union {
                  CUmipmappedArray mipmap;
                  CUarray array;
              } resource;
      
              CUarraySparseSubresourceType subresourceType;   
              union {
                  struct {
                      unsigned int level;                     
                      unsigned int layer;                     
                      unsigned int offsetX;                   
                      unsigned int offsetY;                   
                      unsigned int offsetZ;                   
                      unsigned int extentWidth;               
                      unsigned int extentHeight;              
                      unsigned int extentDepth;               
                  } sparseLevel;
                  struct {
                      unsigned int layer;
                      unsigned long long offset;              
                      unsigned long long size;                
                  } miptail;
              } subresource;
      
              CUmemOperationType memOperationType;
              
              CUmemHandleType memHandleType;                  
              union {
                  CUmemGenericAllocationHandle memHandle;
              } memHandle;
      
              unsigned long long offset;                      
              unsigned int deviceBitMask;                     
              unsigned int flags;                             
              unsigned int reserved[2];                       
          } CUarrayMapInfo;
where CUarrayMapInfo::resourceType specifies the type of resource to be operated on. If CUarrayMapInfo::resourceType is set to CUresourcetype::CU_RESOURCE_TYPE_ARRAY then CUarrayMapInfo::resource::array must be set to a valid sparse CUDA array handle. The CUDA array must be either a 2D, 2D layered or 3D CUDA array and must have been allocated using cuArrayCreate or cuArray3DCreate with the flag CUDA_ARRAY3D_SPARSE or CUDA_ARRAY3D_DEFERRED_MAPPING. For CUDA arrays obtained using cuMipmappedArrayGetLevel, CUDA_ERROR_INVALID_VALUE will be returned. If CUarrayMapInfo::resourceType is set to CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY then CUarrayMapInfo::resource::mipmap must be set to a valid sparse CUDA mipmapped array handle. The CUDA mipmapped array must be either a 2D, 2D layered or 3D CUDA mipmapped array and must have been allocated using cuMipmappedArrayCreate with the flag CUDA_ARRAY3D_SPARSE or CUDA_ARRAY3D_DEFERRED_MAPPING.

CUarrayMapInfo::subresourceType specifies the type of subresource within the resource. CUarraySparseSubresourceType_enum is defined as:

‎    typedef enum CUarraySparseSubresourceType_enum {
              CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = 0,
              CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = 1
          } CUarraySparseSubresourceType;
where CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL indicates a sparse-miplevel which spans at least one tile in every dimension. The remaining miplevels which are too small to span at least one tile in any dimension constitute the mip tail region as indicated by CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL subresource type.

If CUarrayMapInfo::subresourceType is set to CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL then CUarrayMapInfo::subresource::sparseLevel struct must contain valid array subregion offsets and extents. The CUarrayMapInfo::subresource::sparseLevel::offsetX, CUarrayMapInfo::subresource::sparseLevel::offsetY and CUarrayMapInfo::subresource::sparseLevel::offsetZ must specify valid X, Y and Z offsets respectively. The CUarrayMapInfo::subresource::sparseLevel::extentWidth, CUarrayMapInfo::subresource::sparseLevel::extentHeight and CUarrayMapInfo::subresource::sparseLevel::extentDepth must specify valid width, height and depth extents respectively. These offsets and extents must be aligned to the corresponding tile dimension. For CUDA mipmapped arrays CUarrayMapInfo::subresource::sparseLevel::level must specify a valid mip level index. Otherwise, must be zero. For layered CUDA arrays and layered CUDA mipmapped arrays CUarrayMapInfo::subresource::sparseLevel::layer must specify a valid layer index. Otherwise, must be zero. CUarrayMapInfo::subresource::sparseLevel::offsetZ must be zero and CUarrayMapInfo::subresource::sparseLevel::extentDepth must be set to 1 for 2D and 2D layered CUDA arrays and CUDA mipmapped arrays. Tile extents can be obtained by calling cuArrayGetSparseProperties and cuMipmappedArrayGetSparseProperties

If CUarrayMapInfo::subresourceType is set to CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL then CUarrayMapInfo::subresource::miptail struct must contain valid mip tail offset in CUarrayMapInfo::subresource::miptail::offset and size in CUarrayMapInfo::subresource::miptail::size. Both, mip tail offset and mip tail size must be aligned to the tile size. For layered CUDA mipmapped arrays which don't have the flag CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL set in CUDA_ARRAY_SPARSE_PROPERTIES::flags as returned by cuMipmappedArrayGetSparseProperties, CUarrayMapInfo::subresource::miptail::layer must specify a valid layer index. Otherwise, must be zero.

If CUarrayMapInfo::resource::array or CUarrayMapInfo::resource::mipmap was created with CUDA_ARRAY3D_DEFERRED_MAPPING flag set the CUarrayMapInfo::subresourceType and the contents of CUarrayMapInfo::subresource will be ignored.

CUarrayMapInfo::memOperationType specifies the type of operation. CUmemOperationType is defined as:

‎    typedef enum CUmemOperationType_enum {
              CU_MEM_OPERATION_TYPE_MAP = 1,
              CU_MEM_OPERATION_TYPE_UNMAP = 2
          } CUmemOperationType;
If CUarrayMapInfo::memOperationType is set to CUmemOperationType::CU_MEM_OPERATION_TYPE_MAP then the subresource will be mapped onto the tile pool memory specified by CUarrayMapInfo::memHandle at offset CUarrayMapInfo::offset. The tile pool allocation has to be created by specifying the CU_MEM_CREATE_USAGE_TILE_POOL flag when calling cuMemCreate. Also, CUarrayMapInfo::memHandleType must be set to CUmemHandleType::CU_MEM_HANDLE_TYPE_GENERIC.
If CUarrayMapInfo::memOperationType is set to CUmemOperationType::CU_MEM_OPERATION_TYPE_UNMAP then an unmapping operation is performed. CUarrayMapInfo::memHandle must be NULL.

CUarrayMapInfo::deviceBitMask specifies the list of devices that must map or unmap physical memory. Currently, this mask must have exactly one bit set, and the corresponding device must match the device associated with the stream. If CUarrayMapInfo::memOperationType is set to CUmemOperationType::CU_MEM_OPERATION_TYPE_MAP, the device must also match the device associated with the tile pool memory allocation as specified by CUarrayMapInfo::memHandle.

CUarrayMapInfo::flags and CUarrayMapInfo::reserved[] are unused and must be set to zero.

See also:

cuMipmappedArrayCreate, cuArrayCreate, cuArray3DCreate, cuMemCreate, cuArrayGetSparseProperties, cuMipmappedArrayGetSparseProperties

CUresult cuMemRelease ( CUmemGenericAllocationHandle handle )
Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate.
Parameters
handle
Value of handle which was returned previously by cuMemCreate.
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

Description
Frees the memory that was allocated on a device through cuMemCreate.

The memory allocation will be freed when all outstanding mappings to the memory are unmapped and when all outstanding references to the handle (including it's shareable counterparts) are also released. The generic memory handle can be freed when there are still outstanding mappings made with this handle. Each time a recipient process imports a shareable handle, it needs to pair it with cuMemRelease for the handle to be freed. If handle is not a valid handle the behavior is undefined.


Note:
Note that this function may also return error codes from previous, asynchronous launches.

See also:

cuMemCreate

CUresult cuMemRetainAllocationHandle ( CUmemGenericAllocationHandle* handle, void* addr )
Given an address addr, returns the allocation handle of the backing memory allocation.
Parameters
handle
CUDA Memory handle for the backing memory allocation.
addr
Memory address to query, that has been mapped previously.
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

Description
The handle is guaranteed to be the same handle value used to map the memory. If the address requested is not mapped, the function will fail. The returned handle must be released with corresponding number of calls to cuMemRelease.


Note:
The address addr, can be any address in a range previously mapped by cuMemMap, and not necessarily the start address.

See also:

cuMemCreate, cuMemRelease, cuMemMap

CUresult cuMemSetAccess ( CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count )
Set the access flags for each location specified in desc for the given virtual address range.
Parameters
ptr
- Starting address for the virtual address range
size
- Length of the virtual address range
desc
- Array of CUmemAccessDesc that describe how to change the
mapping for each location specified
count
- Number of CUmemAccessDesc in desc
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_NOT_SUPPORTED

Description
Given the virtual address range via ptr and size, and the locations in the array given by desc and count, set the access flags for the target locations. The range must be a fully mapped address range containing all allocations created by cuMemMap / cuMemCreate. Users cannot specify CU_MEM_LOCATION_TYPE_HOST_NUMA accessibility for allocations created on with other location types. Note: When CUmemAccessDesc::CUmemLocation::type is CU_MEM_LOCATION_TYPE_HOST_NUMA, CUmemAccessDesc::CUmemLocation::id is ignored. When setting the access flags for a virtual address range mapping a multicast object, ptr and size must be aligned to the value returned by cuMulticastGetGranularity with the flag CU_MULTICAST_MINIMUM_GRANULARITY. For best performance however, it is recommended that ptr and size be aligned to the value returned by cuMulticastGetGranularity with the flag CU_MULTICAST_RECOMMENDED_GRANULARITY.


Note:
Note that this function may also return error codes from previous, asynchronous launches.

This function exhibits synchronous behavior for most use cases.

See also:

cuMemSetAccess, cuMemCreate, :cuMemMap

CUresult cuMemUnmap ( CUdeviceptr ptr, size_t size )
Unmap the backing memory of a given address range.
Parameters
ptr
- Starting address for the virtual address range to unmap
size
- Size of the virtual address range to unmap
Returns
CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

Description
The range must be the entire contiguous address range that was mapped to. In other words, cuMemUnmap cannot unmap a sub-range of an address range mapped by cuMemCreate / cuMemMap. Any backing memory allocations will be freed if there are no existing mappings and there are no unreleased memory handles.

When cuMemUnmap returns successfully the address range is converted to an address reservation and can be used for a future calls to cuMemMap. Any new mapping to this virtual address will need to have access granted through cuMemSetAccess, as all mappings start with no accessibility setup.


Note:
Note that this function may also return error codes from previous, asynchronous launches.

This function exhibits synchronous behavior for most use cases.
```