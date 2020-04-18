// glfw
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// cpp
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <set>
#include <array>

// windows
#include <aclapi.h>
#include <dxgi1_2.h>
#include <vulkan/vulkan_win32.h>
#include <windows.h>
#include <VersionHelpers.h>
#define _USE_MATH_DEFINES

// cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_image.h>
#include <helper_math.h>
#include "linmath.h"

#include "kernel.cuh"

// ----------------------------------------------------------------------- global constants

// resolution
const int WIDTH = 2560;
const int HEIGHT = 1440;

// Max frames in flight - controls CPU send maxium number of commands to GPU before GPU finish work
// number too low - may not hide enough CPU-GPU bandwidth latency
// number too high - commands are queuing up - consume too much cpu side memory
const int MAX_FRAMES_IN_FLIGHT = 1;

// validation layer
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// extensions
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,

    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,

    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
};

// If debug mode, enable validation layer
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// ----------------------------------------------------------------------- global functions

// Create VK debug utils messenger
VkResult CreateDebugUtilsMessengerEXT(
          VkInstance                          instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks*              pAllocator,
          VkDebugUtilsMessengerEXT*           pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// Destroy VK debug utils messenger
void DestroyDebugUtilsMessengerEXT(
          VkInstance               instance,
          VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks*   pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// ----------------------------------------------------------------------- global structures

// queue family
struct QueueFamilyIndices {
	int graphicsFamily = -1;
	int presentFamily = -1;

	bool isComplete() { return graphicsFamily >= 0 && presentFamily >= 0; }
};

// swap chain helper structure
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities; // capabilities
    std::vector<VkSurfaceFormatKHR> formats;      // formats
    std::vector<VkPresentModeKHR>   presentModes; // present modes
};

// windows security stuffs
class WindowsSecurityAttributes {
protected:
    SECURITY_ATTRIBUTES  m_winSecurityAttributes;
    PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
    WindowsSecurityAttributes();
    SECURITY_ATTRIBUTES* operator&();
    ~WindowsSecurityAttributes();
};

WindowsSecurityAttributes::WindowsSecurityAttributes()
{
    m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));
    // CHECK_NEQ(m_winPSecurityDescriptor, (PSECURITY_DESCRIPTOR)NULL);

    PSID* ppSID = (PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

    InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);

    SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority =
        SECURITY_WORLD_SID_AUTHORITY;
    AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

    EXPLICIT_ACCESS explicitAccess;
    ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
    explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
    explicitAccess.grfAccessMode        = SET_ACCESS;
    explicitAccess.grfInheritance       = INHERIT_ONLY;
    explicitAccess.Trustee.TrusteeForm  = TRUSTEE_IS_SID;
    explicitAccess.Trustee.TrusteeType  = TRUSTEE_IS_WELL_KNOWN_GROUP;
    explicitAccess.Trustee.ptstrName    = (LPTSTR)*ppSID;

    SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

    SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

    m_winSecurityAttributes.nLength              = sizeof(m_winSecurityAttributes);
    m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
    m_winSecurityAttributes.bInheritHandle       = TRUE;
}

SECURITY_ATTRIBUTES* WindowsSecurityAttributes::operator&() {
    return &m_winSecurityAttributes;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
    PSID* ppSID = (PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

    if (*ppSID) {
        FreeSid(*ppSID);
    }
    if (*ppACL) {
        LocalFree(*ppACL);
    }
    free(m_winPSecurityDescriptor);
}

struct UniformBufferObject {
    alignas(16) mat4x4 model;
    alignas(16) mat4x4 view;
    alignas(16) mat4x4 proj;
};

// main structure
class HelloTriangleApplication {
public:
    void run()
    {
        initWindow();
        initVulkan();
        initCuda();

        m_rayTracer = new RayTracer(WIDTH, HEIGHT, WIDTH / 2, HEIGHT / 2);
        m_rayTracer->init();

        mainLoop();

        delete m_rayTracer;

        cleanup();
    }

private:
    // window
    GLFWwindow*                        m_window;

    // vk instance
    VkInstance                         m_instance;

    // vk debug utils messenger
    VkDebugUtilsMessengerEXT           m_debugMessenger;

    // vk surface
    VkSurfaceKHR                       m_surface;

    // phisical device
    VkPhysicalDevice                   m_physicalDevice = VK_NULL_HANDLE;

    // device
    VkDevice                           m_device;

    // vk device UUID
    uint8_t vkDeviceUUID[VK_UUID_SIZE];

    // graphics queue, present queue
    VkQueue                            m_graphicsQueue;
    VkQueue                            m_presentQueue;

    // swapchain
    VkSwapchainKHR                     m_swapChain;

    // swapchain images
    std::vector<VkImage>               m_swapChainImages;

    // swapchain image format, extent
    VkFormat                           m_swapChainImageFormat;
    VkExtent2D                         m_swapChainExtent;

    // swapchain image views
    std::vector<VkImageView>           m_swapChainImageViews;

    // swapchain frame buffers
    std::vector<VkFramebuffer>         m_swapChainFramebuffers;

    // texture image, image GPU memory, image view
    VkImage                            m_textureImage;
    VkDeviceMemory                     m_textureImageMemory;
    VkImageView                        m_textureImageView;

    // texture sampler
    VkSampler                          m_textureSampler;

    // descriptor pool, sets
    VkDescriptorPool                   m_descriptorPool;
    std::vector<VkDescriptorSet>       m_descriptorSets;
    VkDescriptorSetLayout              m_descriptorSetLayout;

    // uniform buffer
    std::vector<VkBuffer>              m_uniformBuffers;
    std::vector<VkDeviceMemory>        m_uniformBuffersMemory;

    // renderpass
    VkRenderPass                       m_renderPass;

    // pipeline layout
    VkPipelineLayout                   m_pipelineLayout;

    // graphics pipeline
    VkPipeline                         m_graphicsPipeline;

    // command pool
    VkCommandPool                      m_commandPool;

    // command buffers
    std::vector<VkCommandBuffer>       m_commandBuffers;

    // semaphores, fences
    std::vector<VkSemaphore>           m_imageAvailableSemaphores;
    std::vector<VkSemaphore>           m_renderFinishedSemaphores;
    std::vector<VkFence>               m_inFlightFences;
    std::vector<VkFence>               m_imagesInFlight;

    // current frame - for multiple commands synchronization
    size_t                             m_currentFrame = 0;

    // Get memory and semaphore win32 function pointer
    PFN_vkGetMemoryWin32HandleKHR      m_fpGetMemoryWin32HandleKHR;
    PFN_vkGetSemaphoreWin32HandleKHR   m_fpGetSemaphoreWin32HandleKHR;

    // Gey physical device properties function pointer
    PFN_vkGetPhysicalDeviceProperties2 m_fpGetPhysicalDeviceProperties2;

    // Texture image pointer, extent, mip levels and memory size
    // unsigned int*                      m_imageData = NULL;
    unsigned int                       m_imageWidth;
    unsigned int                       m_imageHeight;
    unsigned int                       m_imageMipLevels;
    size_t                             m_totalImageMemSize;

    // CUDA objects
    cudaExternalMemory_t               m_cudaExtMemImageBuffer;
    cudaMipmappedArray_t               m_cudaMipmappedImageArray;
    std::vector<SurfObj>   m_surfaceObjectList;
    SurfObj*               d_surfaceObjectList;
    cudaStream_t                       m_streamToRun;

    // vk semaphore
    VkSemaphore                        m_cudaUpdateVkSemaphore;
    VkSemaphore                        m_vkUpdateCudaSemaphore;
    cudaExternalSemaphore_t            m_cudaExtCudaUpdateVkSemaphore;
    cudaExternalSemaphore_t            m_cudaExtVkUpdateCudaSemaphore;

    //
    RayTracer* m_rayTracer;

    // ----------------------------------------------------------------------------------- functions

    // init window
    void initWindow() {
        glfwInit();

        // basic window attributes
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        // window width, height, title
        m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    // init vulkan
    void initVulkan() {
        // instance, debug, surface, physical device, device
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        getKhrExtensionsFn();

        // swap chain, image views
        createSwapChain();
        createSwapChainImageViews();

        // render pass
        createRenderPass();

        // graphics pipeline
		createDescriptorSetLayout();
        createGraphicsPipeline();

        // frame buffers
        createFramebuffers();

        // command pool
        createCommandPool();

        // texture image, image view, sampler
        createTextureImage();
        createTextureImageView();
        createTextureSampler();

        // uniform buffer
        createUniformBuffers();

        // descriptor pool, set
        createDescriptorPool();
        createDescriptorSets();

        // commands buffers
        createCommandBuffers();

        // sync objects
        createSyncObjects();
        createSyncObjectsExt();
     }

    // init cuda
    void initCuda() {
        // return value not used, function as device checking
        int deviceId = setCudaVkDevice();

        // Create stream
        checkCudaErrors(cudaStreamCreate(&m_streamToRun));

        // Create Cuda memory objects, imported from VK
        cudaVkImportImageMem();

        // CUDA import semaphores
        cudaVkImportSemaphore();
    }

    // main loop
    void mainLoop() {

        updateUniformBuffer();

        // glfw handles the loop
        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();

            // call drawFrame()
            drawFrame();
        }

        // on loop finish: vkDeviceWaitIdle()
        vkDeviceWaitIdle(m_device);
    }

    // Clean up swap chain
    void cleanupSwapChain()
    {
        // 3 frame buffers
        for (auto framebuffer : m_swapChainFramebuffers) {
            vkDestroyFramebuffer(m_device, framebuffer, nullptr);
        }

        // 3 command buffers
        vkFreeCommandBuffers(m_device, m_commandPool, static_cast<uint32_t>(m_commandBuffers.size()), m_commandBuffers.data());

        // pipeline, pipeline layout, render pass
        vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        vkDestroyRenderPass(m_device, m_renderPass, nullptr);

        // 3 image views
        for (auto imageView : m_swapChainImageViews) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }

        for (size_t i = 0; i < m_swapChainImages.size(); i++) {
            vkDestroyBuffer(m_device, m_uniformBuffers[i], nullptr);
            vkFreeMemory(m_device, m_uniformBuffersMemory[i], nullptr);
        }

        // swapchain itself
        vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);

        // descriptor pool
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    }

    // clean up
    void cleanup()
    {
        // clean up swap chain
        cleanupSwapChain();

        // texture sampler, image view
        vkDestroySampler(m_device, m_textureSampler, nullptr);
        vkDestroyImageView(m_device, m_textureImageView, nullptr);

        // CUDA surface objects
        for (unsigned int i = 0; i < m_imageMipLevels; i++) {
            checkCudaErrors(cudaDestroySurfaceObject(m_surfaceObjectList[i]));
        }

        // CUDA surface object
        checkCudaErrors(cudaFree(d_surfaceObjectList));

        // CUDA mipmapped array
        checkCudaErrors(cudaFreeMipmappedArray(m_cudaMipmappedImageArray));

        // CUDA external memory
        checkCudaErrors(cudaDestroyExternalMemory(m_cudaExtMemImageBuffer));

        // CUDA external semaphores
        checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaExtCudaUpdateVkSemaphore));
        checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaExtVkUpdateCudaSemaphore));

        // texture image, image memory
        vkDestroyImage(m_device, m_textureImage, nullptr);
        vkFreeMemory(m_device, m_textureImageMemory, nullptr);

        // descriptor set layout
        vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);

        // For each frame-in-flight, destroy semaphore and fence
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(m_device, m_renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(m_device, m_imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(m_device, m_inFlightFences[i], nullptr);
        }

        vkDestroySemaphore(m_device, m_cudaUpdateVkSemaphore, nullptr);
        vkDestroySemaphore(m_device, m_vkUpdateCudaSemaphore, nullptr);

        // command pool
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);

        // device
        vkDestroyDevice(m_device, nullptr);

        // validation layer
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
        }

        // surface
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);

        // instance
        vkDestroyInstance(m_instance, nullptr);

        // window
        glfwDestroyWindow(m_window);

        // glfw terminate
        glfwTerminate();

        cudaDeviceReset();
    }

    // create instance
    void createInstance() {
        // validation layer
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        // basic vulkan app info
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        // instance create info
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // ext
        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // validation layer
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        // instance
        if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

        // fp get physical device properties 2
        m_fpGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(m_instance, "vkGetPhysicalDeviceProperties2");
        if (m_fpGetPhysicalDeviceProperties2 == NULL) {
            throw std::runtime_error(
                "Vulkan: Proc address for \"vkGetPhysicalDeviceProperties2KHR\" not "
                "found.\n");
        }

        // fp get memory handle
        m_fpGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)vkGetInstanceProcAddr(m_instance, "vkGetMemoryWin32HandleKHR");
        if (m_fpGetMemoryWin32HandleKHR == NULL) {
            throw std::runtime_error(
                "Vulkan: Proc address for \"vkGetMemoryWin32HandleKHR\" not "
                "found.\n");
        }
    }

    // create surface - glfw window
    void createSurface() {
        if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    // pick physical device - enumerate, suitable, select
    void pickPhysicalDevice() {
        // enumerate
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

        // suitable
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                m_physicalDevice = device;
                break;
            }
        }

        if (m_physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        // log
        std::cout << "Selected physical device = " << m_physicalDevice << std::endl;

        // get physical device ID properties
        VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
        vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        vkPhysicalDeviceIDProperties.pNext = NULL;

        VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
        vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

        m_fpGetPhysicalDeviceProperties2(m_physicalDevice, &vkPhysicalDeviceProperties2);

        // get device UUID
        memcpy(vkDeviceUUID, vkPhysicalDeviceIDProperties.deviceUUID, sizeof(vkDeviceUUID));
    }

    // create logical device
    void createLogicalDevice() {
        // queue create info
        QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<int> uniqueQueueFamilies = {indices.graphicsFamily, indices.presentFamily};

        float queuePriority = 1.0f;
        for (int queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex        = queueFamily;
            queueCreateInfo.queueCount              = 1;
            queueCreateInfo.pQueuePriorities        = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // physical device features
        VkPhysicalDeviceFeatures deviceFeatures = {};

        // device create info
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        // queu create info
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        // device features
        createInfo.pEnabledFeatures = &deviceFeatures;

        // ext
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // validation layer
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        // create device
        if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        // get queue
        vkGetDeviceQueue(m_device, indices.graphicsFamily, 0, &m_graphicsQueue);
        vkGetDeviceQueue(m_device, indices.presentFamily, 0, &m_presentQueue);
    }

    // create swap chain
    void createSwapChain() {
        // swap chain support
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(m_physicalDevice);

        // Choose swap chain details
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR   presentMode   = chooseSwapPresentMode  (swapChainSupport.presentModes);
        VkExtent2D         extent        = chooseSwapExtent       (swapChainSupport.capabilities);

        // choose image count
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        // swap chain create info
        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType                    = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface                  = m_surface;
        createInfo.minImageCount            = imageCount;
        createInfo.imageFormat              = surfaceFormat.format;
        createInfo.imageColorSpace          = surfaceFormat.colorSpace;
        createInfo.imageExtent              = extent;
        createInfo.imageArrayLayers         = 1;
        createInfo.imageUsage               = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        // get queue index
        QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);
        uint32_t queueFamilyIndices[] = { (uint32_t)indices.graphicsFamily, (uint32_t)indices.presentFamily};

        // image sharing mode, concurrent or exclusive
        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices   = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform   = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode    = presentMode;
        createInfo.clipped        = VK_TRUE;
        createInfo.oldSwapchain   = VK_NULL_HANDLE;

        // create
        if (vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        // get images
        vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, nullptr);
        m_swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, m_swapChainImages.data());

        // save things
        m_swapChainImageFormat = surfaceFormat.format;
        m_swapChainExtent      = extent;
    }

    // create image view
    void createSwapChainImageViews() {
        m_swapChainImageViews.resize(m_swapChainImages.size());
        for (size_t i = 0; i < m_swapChainImages.size(); i++) {
            m_swapChainImageViews[i] = createImageView(m_swapChainImages[i], m_swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }

    // create render pass
    void createRenderPass()
    {
        // color attachment - color render target
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format         = m_swapChainImageFormat;
        colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createGraphicsPipeline()
    {
        auto vertShaderCode = readFile("resources/shaders/tri.vert.spv");
        auto fragShaderCode = readFile("resources/shaders/tri.frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
        vertShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName  = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
        fragShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName  = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount   = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport = {};
        viewport.x        = 0.0f;
        viewport.y        = 0.0f;
        viewport.width    = (float) m_swapChainExtent.width;
        viewport.height   = (float) m_swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = {0, 0};
        scissor.extent = m_swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports    = &viewport;
        viewportState.scissorCount  = 1;
        viewportState.pScissors     = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable        = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth               = 1.0f;
        rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace               = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable         = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable  = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                              VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT |
                                              VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable    = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable     = VK_FALSE;
        colorBlending.logicOp           = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount   = 1;
        colorBlending.pAttachments      = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount         = 1;
        pipelineLayoutInfo.pSetLayouts            = &m_descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount          = 2;
        pipelineInfo.pStages             = shaderStages;
        pipelineInfo.pVertexInputState   = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState      = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState   = &multisampling;
        pipelineInfo.pColorBlendState    = &colorBlending;
        pipelineInfo.layout              = m_pipelineLayout;
        pipelineInfo.renderPass          = m_renderPass;
        pipelineInfo.subpass             = 0;
        pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, vertShaderModule, nullptr);
    }

    void createFramebuffers() {
        m_swapChainFramebuffers.resize(m_swapChainImageViews.size());

        for (size_t i = 0; i < m_swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                m_swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = m_renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = m_swapChainExtent.width;
            framebufferInfo.height = m_swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(m_physicalDevice);

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createCommandBuffers()
	{
        // resize cmd buffer
        m_commandBuffers.resize(m_swapChainFramebuffers.size());

        // alloc info
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = m_commandPool;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) m_commandBuffers.size();

        // alloc cmd buffer
        if (vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        // for each cmd buffer
        for (size_t i = 0; i < m_commandBuffers.size(); i++)
        {
            // cmd buffer begin info
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            vkResetCommandBuffer(m_commandBuffers[i], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

            // begin cmd buffer
            if (vkBeginCommandBuffer(m_commandBuffers[i], &beginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            // render pass info
            VkRenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass        = m_renderPass;
            renderPassInfo.framebuffer       = m_swapChainFramebuffers[i];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = m_swapChainExtent;

            // clear color
            VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues    = &clearColor;

            // cmd: begin render pass
            vkCmdBeginRenderPass(m_commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            // cmd: bind pipeline
            vkCmdBindPipeline(m_commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

            // cmd: bind descriptor sets
            vkCmdBindDescriptorSets(
				m_commandBuffers[i]            , // VkCommandBuffer        commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS, // VkPipelineBindPoint    pipelineBindPoint,
                m_pipelineLayout               , // VkPipelineLayout       layout,
                0                              , // uint32_t               firstSet,
                1                              , // uint32_t               descriptorSetCount,
                &m_descriptorSets[i]           , // const VkDescriptorSet* pDescriptorSets,
                0                              , // uint32_t               dynamicOffsetCount,
                nullptr                       ); // const uint32_t*        pDynamicOffsets);

            // cmd: draw
            vkCmdDraw(m_commandBuffers[i], // VkCommandBuffer commandBuffer,
                      3                  , // uint32_t        vertexCount,
                      1                  , // uint32_t        instanceCount,
                      0                  , // uint32_t        firstVertex,
                      0                 ); // uint32_t        firstInstance);

            // cmd: end render pass
            vkCmdEndRenderPass(m_commandBuffers[i]);

            // end cmd buffer
            if (vkEndCommandBuffer(m_commandBuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }

    void createSyncObjects()
    {
        m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        m_imagesInFlight.resize(m_swapChainImages.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(m_device, &fenceInfo, nullptr, &m_inFlightFences[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void createSyncObjectsExt()
    {
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        memset(&semaphoreInfo, 0, sizeof(semaphoreInfo));
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        WindowsSecurityAttributes winSecurityAttributes;

        VkExportSemaphoreWin32HandleInfoKHR vulkanExportSemaphoreWin32HandleInfoKHR = {};
        vulkanExportSemaphoreWin32HandleInfoKHR.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
        vulkanExportSemaphoreWin32HandleInfoKHR.pNext       = NULL;
        vulkanExportSemaphoreWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
        vulkanExportSemaphoreWin32HandleInfoKHR.dwAccess    = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
        vulkanExportSemaphoreWin32HandleInfoKHR.name        = (LPCWSTR)NULL;

        VkExportSemaphoreCreateInfoKHR vulkanExportSemaphoreCreateInfo = {};
        vulkanExportSemaphoreCreateInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
        vulkanExportSemaphoreCreateInfo.pNext       = &vulkanExportSemaphoreWin32HandleInfoKHR;
        vulkanExportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        semaphoreInfo.pNext = &vulkanExportSemaphoreCreateInfo;

        if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_cudaUpdateVkSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_vkUpdateCudaSemaphore) != VK_SUCCESS)
        {
            throw std::runtime_error( "failed to create synchronization objects for a CUDA-Vulkan!");
        }
    }

    void drawFrame()
    {
        // cpu-gpu current frame fence
        vkWaitForFences(
            m_device                         , // VkDevice       device,
            1                                , // uint32_t       fenceCount,
            &m_inFlightFences[m_currentFrame], // const VkFence* pFences,
            VK_TRUE                          , // VkBool32       waitAll,
            UINT64_MAX                      ); // uint64_t       timeout);

        // acquire next image in swap chain (present -> graphics)
        uint32_t imageIndex;
        vkAcquireNextImageKHR(
            m_device                                  , // VkDevice       device,
            m_swapChain                               , // VkSwapchainKHR swapchain,
            UINT64_MAX                                , // uint64_t       timeout,
            m_imageAvailableSemaphores[m_currentFrame], // VkSemaphore    semaphore,
            VK_NULL_HANDLE                            , // VkFence        fence,
            &imageIndex                              ); // uint32_t*      pImageIndex);

        // cpu-gpu current image fence
        if (m_imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        {
            vkWaitForFences(
                m_device                     , // VkDevice       device,
                1                            , // uint32_t       fenceCount,
                &m_imagesInFlight[imageIndex], // const VkFence* pFences,
                VK_TRUE                      , // VkBool32       waitAll,
                UINT64_MAX                  ); // uint64_t       timeout);
        }

        // set current image fence to current frame fence
        m_imagesInFlight[imageIndex] = m_inFlightFences[m_currentFrame];

        // reset fence of current frame
        vkResetFences(m_device, 1, &m_inFlightFences[m_currentFrame]);

        static int startSubmit = 0;
        if (!startSubmit)
        {
            // semaphores
            VkSemaphore waitSemaphores[]   = { m_imageAvailableSemaphores[m_currentFrame] };
            VkSemaphore signalSemaphores[] = { m_renderFinishedSemaphores[m_currentFrame], m_vkUpdateCudaSemaphore };

            // wait stage
            VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

            // submit info
            VkSubmitInfo submitInfo = {};
            submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount   = 1;
            submitInfo.pCommandBuffers      = &m_commandBuffers[imageIndex];
            submitInfo.waitSemaphoreCount   = 1;
            submitInfo.pWaitSemaphores      = waitSemaphores;
            submitInfo.pWaitDstStageMask    = waitStages;
            submitInfo.signalSemaphoreCount = 2;
            submitInfo.pSignalSemaphores    = signalSemaphores;

            // submit command of graphics to queue
            if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrame]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to submit draw command buffer!");
            }

            startSubmit = 1;
        }
        else
        {
            // semaphores
            VkSemaphore waitSemaphores[]      = { m_imageAvailableSemaphores[m_currentFrame], m_cudaUpdateVkSemaphore };
            VkSemaphore signalSemaphores[]    = { m_renderFinishedSemaphores[m_currentFrame], m_vkUpdateCudaSemaphore };

            // wait stage
            VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };

            // submit info
            VkSubmitInfo submitInfo           = {};
            submitInfo.sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount     = 2;
            submitInfo.pWaitSemaphores        = waitSemaphores;
            submitInfo.pWaitDstStageMask      = waitStages;
            submitInfo.commandBufferCount     = 1;
            submitInfo.pCommandBuffers        = &m_commandBuffers[imageIndex];
            submitInfo.signalSemaphoreCount   = 2;
            submitInfo.pSignalSemaphores      = signalSemaphores;

            // submit command of graphics to queue
            if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrame]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to submit draw command buffer!");
            }
        }

        // swap chain
        VkSwapchainKHR swapChains[] = { m_swapChain };

        VkSemaphore presentSignalSemaphores[] = { m_renderFinishedSemaphores[m_currentFrame] };

        // present info
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores    = presentSignalSemaphores;
        presentInfo.swapchainCount     = 1;
        presentInfo.pSwapchains        = swapChains;
        presentInfo.pImageIndices      = &imageIndex;

        // submit command of present to queue
        vkQueuePresentKHR(m_presentQueue, &presentInfo);

        // cuda update
        cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;
        memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));
        extSemaphoreWaitParams.params.fence.value = 0;
        extSemaphoreWaitParams.flags = 0;
        checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaExtVkUpdateCudaSemaphore, &extSemaphoreWaitParams, 1, m_streamToRun));

        m_rayTracer->draw(d_surfaceObjectList);

        cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
        memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));
        extSemaphoreSignalParams.params.fence.value = 0;
        extSemaphoreSignalParams.flags = 0;
        checkCudaErrors(cudaSignalExternalSemaphoresAsync(&m_cudaExtCudaUpdateVkSemaphore, &extSemaphoreSignalParams, 1, m_streamToRun));

        // current in-flight frame number
        m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    // ------------------------------------------------------------ swap chain -----------------------------------------------------------------
    // choose swap chain format - RGBA8_UNORM and SRGB
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    // choose swap chain present mode - mailbox mode (triple buffering)
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    // choose swap chain extent - clamped size of user setting
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        } else {
            VkExtent2D actualExtent = {WIDTH, HEIGHT};
            actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
            return actualExtent;
        }
    }

    // swap chain support - capabilities, formats, present modes
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice physicalDevice) {
        SwapChainSupportDetails details;
        // capabilities
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, m_surface, &details.capabilities);
        // formats
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, m_surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, m_surface, &formatCount, details.formats.data());
        }
        // present modes
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, m_surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, m_surface, &presentModeCount, details.presentModes.data());
        }
        return details;
    }

    // ------------------------------------------------------------ device creation utils -----------------------------------------------------------------
    // is device suitable
    bool isDeviceSuitable(VkPhysicalDevice physicalDevice) {
        // get queue index
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        // if ext supported
        bool extensionsSupported = checkDeviceExtensionSupport(physicalDevice);
        // swap chain ok
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }
        // queue index ok, glfw ext ok, swap chain ok
        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    // check device ext support
    bool checkDeviceExtensionSupport(VkPhysicalDevice physicalDevice)
	{
        // enumerate ext
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

        // global set required ext
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        // if required ext is supported
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    // find queue family index, graphics queue index and present queue index
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, m_surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    // get required extensions
    std::vector<const char*> getRequiredExtensions() {
        // glfw ext
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        // validation layer ext
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }

    // check if validation layer is supported
    bool checkValidationLayerSupport() {
        // enumerate all avaliable layers
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        // compare layer names
        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound) {
                return false;
            }
        }
        return true;
    }

    // ----------------------------------------------------------------- file utils --------------------------------------------------------------------------------
    // simply read file into a string
    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    // ----------------------------------------------------------------- validation layer debug ------------------------------------------------------------------------------------------
    // populate debug messenger create info - set debug call back function
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = {};
        createInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    // setup debug messenger
    void setupDebugMessenger()
    {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(m_instance, &createInfo, nullptr, &m_debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    // the debug call back
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    // create texture image
    void createTextureImage()
    {
        m_imageWidth = WIDTH;
        m_imageHeight = HEIGHT;

        // size
        VkDeviceSize imageSize = m_imageWidth * m_imageHeight * 4;

        // mip level
        m_imageMipLevels = 1;

        // create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory);

        void* data;
        vkMapMemory(m_device, stagingBufferMemory, 0, imageSize, 0, &data);
        memset(data, 0, static_cast<size_t>(imageSize));
        vkUnmapMemory(m_device, stagingBufferMemory);

        // VK_FORMAT_R8G8B8A8_UNORM changed to VK_FORMAT_R8G8B8A8_UINT
        createImage(
            m_imageWidth,
            m_imageHeight,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_textureImage,
            m_textureImageMemory);

        transitionImageLayout(m_textureImage,
                              VK_FORMAT_R8G8B8A8_UNORM,
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        copyBufferToImage(stagingBuffer,
                          m_textureImage,
                          static_cast<uint32_t>(m_imageWidth),
                          static_cast<uint32_t>(m_imageHeight));

        transitionImageLayout(m_textureImage,
                              VK_FORMAT_R8G8B8A8_UNORM,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        vkFreeMemory(m_device, stagingBufferMemory, nullptr);

        // generateMipmaps(m_textureImage, VK_FORMAT_R8G8B8A8_UNORM);
    }

    void createTextureImageView()
    {
        m_textureImageView = createImageView(m_textureImage,
                                             VK_FORMAT_R8G8B8A8_UNORM,
                                             VK_IMAGE_ASPECT_COLOR_BIT,
                                             m_imageMipLevels);
    }

    void createTextureSampler()
    {
        VkSamplerCreateInfo samplerInfo = {};
        samplerInfo.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter               = VK_FILTER_LINEAR;
        samplerInfo.minFilter               = VK_FILTER_LINEAR;
        samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable        = VK_FALSE;
        samplerInfo.maxAnisotropy           = 16;
        samplerInfo.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable           = VK_FALSE;
        samplerInfo.compareOp               = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.minLod                  = 0;  // Optional
        samplerInfo.maxLod                  = static_cast<float>(m_imageMipLevels);
        samplerInfo.mipLodBias              = 0;  // Optional

        if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_textureSampler) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    // create image view
    // default settings are
    // - 2D image
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels)
    {
        VkImageViewCreateInfo viewInfo = {};

        viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image                           = image;
        viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format                          = format;
        viewInfo.subresourceRange.aspectMask     = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;

        VkImageView imageView;
        if (vkCreateImageView(m_device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }
        return imageView;
    }

    void generateMipmaps(VkImage image, VkFormat imageFormat)
    {
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(m_physicalDevice, imageFormat, &formatProperties);

        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
		{
            throw std::runtime_error("texture image format does not support linear blitting!");
        }

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier = {};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image                           = image;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.subresourceRange.levelCount     = 1;

        int32_t mipWidth  = m_imageWidth;
        int32_t mipHeight = m_imageHeight;

        for (uint32_t i = 1; i < m_imageMipLevels; i++)
        {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout                     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout                     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask                 = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask                 = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(
                commandBuffer                 , // VkCommandBuffer               commandBuffer
                VK_PIPELINE_STAGE_TRANSFER_BIT, // VkPipelineStageFlags          srcStageMask
                VK_PIPELINE_STAGE_TRANSFER_BIT, // VkPipelineStageFlags          dstStageMask
                0                             , // VkDependencyFlags             dependencyFlags
                0                             , // uint32_t                      memoryBarrierCount
                nullptr                       , // const VkMemoryBarrier*        pMemoryBarriers
                0                             , // uint32_t                      bufferMemoryBarrierCount
                nullptr                       , // const VkBufferMemoryBarrier*  pBufferMemoryBarriers
                1                             , // uint32_t                      imageMemoryBarrierCount
                &barrier                     ); // const VkImageMemoryBarrier*   pImageMemoryBarriers

            VkImageBlit blit = {};
            blit.srcOffsets[0]                 = { 0, 0, 0 };
            blit.srcOffsets[1]                 = { mipWidth, mipHeight, 1 };
            blit.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel       = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount     = 1;
            blit.dstOffsets[0]                 = { 0, 0, 0 };
            blit.dstOffsets[1]                 = {
                mipWidth > 1  ? mipWidth  / 2 : 1,
                mipHeight > 1 ? mipHeight / 2 : 1,
                1
            };
            blit.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel       = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount     = 1;

            vkCmdBlitImage(
                commandBuffer                       , // VkCommandBuffer     commandBuffer,
                image                               , // VkImage             srcImage,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, // VkImageLayout       srcImageLayout,
                image                               , // VkImage             dstImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, // VkImageLayout       dstImageLayout,
                1                                   , // uint32_t            regionCount,
                &blit                               , // const VkImageBlit*  pRegions,
                VK_FILTER_LINEAR                   ); // VkFilter            filter);

            barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
                commandBuffer                        , // VkCommandBuffer               commandBuffer
                VK_PIPELINE_STAGE_TRANSFER_BIT       , // VkPipelineStageFlags          srcStageMask
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, // VkPipelineStageFlags          dstStageMask
                0                                    , // VkDependencyFlags             dependencyFlags
                0                                    , // uint32_t                      memoryBarrierCount
                nullptr                              , // const VkMemoryBarrier*        pMemoryBarriers
                0                                    , // uint32_t                      bufferMemoryBarrierCount
                nullptr                              , // const VkBufferMemoryBarrier*  pBufferMemoryBarriers
                1                                    , // uint32_t                      imageMemoryBarrierCount
                &barrier                            ); // const VkImageMemoryBarrier*   pImageMemoryBarriers

            if (mipWidth  > 1) mipWidth  /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        barrier.subresourceRange.baseMipLevel = m_imageMipLevels - 1;
        barrier.oldLayout                     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout                     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask                 = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask                 = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            commandBuffer                        , // VkCommandBuffer               commandBuffer
            VK_PIPELINE_STAGE_TRANSFER_BIT       , // VkPipelineStageFlags          srcStageMask
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, // VkPipelineStageFlags          dstStageMask
            0                                    , // VkDependencyFlags             dependencyFlags
            0                                    , // uint32_t                      memoryBarrierCount
            nullptr                              , // const VkMemoryBarrier*        pMemoryBarriers
            0                                    , // uint32_t                      bufferMemoryBarrierCount
            nullptr                              , // const VkBufferMemoryBarrier*  pBufferMemoryBarriers
            1                                    , // uint32_t                      imageMemoryBarrierCount
            &barrier                            ); // const VkImageMemoryBarrier*   pImageMemoryBarriers

        endSingleTimeCommands(commandBuffer);
    }

    // get vk image handle
    HANDLE getVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType)
    {
        HANDLE handle;
        VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};

        vkMemoryGetWin32HandleInfoKHR.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        vkMemoryGetWin32HandleInfoKHR.pNext      = NULL;
        vkMemoryGetWin32HandleInfoKHR.memory     = m_textureImageMemory;
        vkMemoryGetWin32HandleInfoKHR.handleType = (VkExternalMemoryHandleTypeFlagBitsKHR)externalMemoryHandleType;

        m_fpGetMemoryWin32HandleKHR(m_device, &vkMemoryGetWin32HandleInfoKHR, &handle);
        return handle;
    }

    HANDLE getVkSemaphoreHandle(
        VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType,
        VkSemaphore&                             semVkCuda)
	{
        HANDLE handle;

        VkSemaphoreGetWin32HandleInfoKHR vulkanSemaphoreGetWin32HandleInfoKHR = {};
        vulkanSemaphoreGetWin32HandleInfoKHR.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
        vulkanSemaphoreGetWin32HandleInfoKHR.pNext      = NULL;
        vulkanSemaphoreGetWin32HandleInfoKHR.semaphore  = semVkCuda;
        vulkanSemaphoreGetWin32HandleInfoKHR.handleType = externalSemaphoreHandleType;

        m_fpGetSemaphoreWin32HandleKHR(m_device,
                                       &vulkanSemaphoreGetWin32HandleInfoKHR,
                                       &handle);

        return handle;
    }

    // void loadImageData(const std::string& filename)
    // {
    //     sdkLoadPPM4(filename.c_str(), (unsigned char**)&m_imageData, &m_imageWidth, &m_imageHeight);

    //     if (!m_imageData) {
    //         printf("Error opening file '%s'\n", filename.c_str());
    //         exit(EXIT_FAILURE);
    //     }

    //     printf("Loaded '%s', %d x %d pixels\n", filename.c_str(), m_imageWidth, m_imageHeight);
    // }

    // get ext fn
    void getKhrExtensionsFn()
    {
        m_fpGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(m_device, "vkGetSemaphoreWin32HandleKHR");
        if (m_fpGetSemaphoreWin32HandleKHR == NULL) {
            throw std::runtime_error("Vulkan: Proc address for \"vkGetSemaphoreWin32HandleKHR\" not found.\n");
        }
    }

    // set cuda vk device
    // find suitable physical device, return deviceId
    int setCudaVkDevice()
    {
        int current_device = 0;
        int device_count = 0;
        int devices_prohibited = 0;

        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceCount(&device_count));

        if (device_count == 0) {
            fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
            exit(EXIT_FAILURE);
        }

        // Find the GPU which is selected by Vulkan
        while (current_device < device_count) {
            cudaGetDeviceProperties(&deviceProp, current_device);

            if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
            // Compare the cuda device UUID with vulkan UUID
            int ret = memcmp(&deviceProp.uuid, &vkDeviceUUID, VK_UUID_SIZE);
            if (ret == 0) {
                checkCudaErrors(cudaSetDevice(current_device));
                checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
                printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
                        current_device, deviceProp.name, deviceProp.major,
                        deviceProp.minor);

                return current_device;
            }

            } else {
                devices_prohibited++;
            }

            current_device++;
        }

        if (devices_prohibited == device_count) {
            fprintf(stderr,
                    "CUDA error:"
                    " No Vulkan-CUDA Interop capable GPU found.\n");
            exit(EXIT_FAILURE);
        }

        return -1;
    }

    void createBuffer(
        VkDeviceSize          size,
        VkBufferUsageFlags    usage,
        VkMemoryPropertyFlags properties,
        VkBuffer&             buffer,
        VkDeviceMemory&       bufferMemory)
    {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(m_device, buffer, bufferMemory, 0);
    }

    VkCommandBuffer beginSingleTimeCommands()
    {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool        = m_commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkResetCommandBuffer(commandBuffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

        vkBeginCommandBuffer(commandBuffer, &beginInfo);


        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &commandBuffer;

        vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_graphicsQueue);

        vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion = {};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    // create descriptor pool, set size of pool here
    void createDescriptorPool()
    {
        // size of pool = size of UBO + size of texture sampler
        std::array<VkDescriptorPoolSize, 2> poolSizes = {};

        poolSizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(m_swapChainImages.size());

        poolSizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(m_swapChainImages.size());

        // pool info
        VkDescriptorPoolCreateInfo poolInfo = {};

        poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes    = poolSizes.data();
        poolInfo.maxSets       = static_cast<uint32_t>(m_swapChainImages.size());

        // create descriptor pool
        if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    // create descriptor set layout (binding info)
    void createDescriptorSetLayout()
    {
        // layout binding
        VkDescriptorSetLayoutBinding uboLayoutBinding = {};
        uboLayoutBinding.binding            = 0;
        uboLayoutBinding.descriptorCount    = 1;
        uboLayoutBinding.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        uboLayoutBinding.stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;

        // sampler layout binding
        VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
        samplerLayoutBinding.binding            = 1;
        samplerLayoutBinding.descriptorCount    = 1;
        samplerLayoutBinding.descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;

        // bindings
        std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };

        // create info
        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings    = bindings.data();

        // create descriptor set layout
        if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
   }

    // create descriptor sets
    void createDescriptorSets()
    {
        // layouts vector for each swap chain image
        std::vector<VkDescriptorSetLayout> layouts(m_swapChainImages.size(), m_descriptorSetLayout);

        // allocation info
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool              = m_descriptorPool;
        allocInfo.descriptorSetCount          = static_cast<uint32_t>(m_swapChainImages.size());
        allocInfo.pSetLayouts                 = layouts.data();

        // resize descriptor sets
        m_descriptorSets.resize(m_swapChainImages.size());

        // allocate descriptor sets
        if (vkAllocateDescriptorSets(m_device, &allocInfo, m_descriptorSets.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        // for each
        for (size_t i = 0; i < m_swapChainImages.size(); i++)
        {
            // descriptor buffer info
            VkDescriptorBufferInfo bufferInfo = {};
            bufferInfo.buffer = m_uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range  = sizeof(UniformBufferObject);

            // descriptor image info
            VkDescriptorImageInfo imageInfo = {};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView   = m_textureImageView;
            imageInfo.sampler     = m_textureSampler;

            // descriptor write
            std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

            // write descriptor set 0
            descriptorWrites[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet          = m_descriptorSets[i];
            descriptorWrites[0].dstBinding      = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo     = &bufferInfo;

            // write descriptor set 1
            descriptorWrites[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet          = m_descriptorSets[i];
            descriptorWrites[1].dstBinding      = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo      = &imageInfo;

            // update descriptor sets
            vkUpdateDescriptorSets(
                m_device                                      , // VkDevice                    device,
                static_cast<uint32_t>(descriptorWrites.size()), // uint32_t                    descriptorWriteCount,
                descriptorWrites.data()                       , // const VkWriteDescriptorSet* pDescriptorWrites,
                0                                             , // uint32_t                    descriptorCopyCount,
                nullptr                                      ); // const VkCopyDescriptorSet*  pDescriptorCopies);
        }
    }

    void createUniformBuffers()
    {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        m_uniformBuffers.resize(m_swapChainImages.size());
        m_uniformBuffersMemory.resize(m_swapChainImages.size());

        for (size_t i = 0; i < m_swapChainImages.size(); i++) {
            createBuffer(
                bufferSize,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_uniformBuffers[i],
                m_uniformBuffersMemory[i]);
        }
    }

    void updateUniformBuffer()
    {
        UniformBufferObject ubo = {};

        mat4x4_identity(ubo.model);
        mat4x4 Model;
        mat4x4_dup(Model, ubo.model);
        mat4x4_rotate(ubo.model, Model, 0.0f, 0.0f, 1.0f, degreesToRadians(135.0f));

        vec3 eye = {2.0f, 2.0f, 2.0f};
        vec3 center = {0.0f, 0.0f, 0.0f};
        vec3 up = {0.0f, 0.0f, 1.0f};
        mat4x4_look_at(ubo.view, eye, center, up);

        mat4x4_perspective(ubo.proj, degreesToRadians(45.0f), m_swapChainExtent.width / (float)m_swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        for (size_t i = 0; i < m_swapChainImages.size(); i++) {
            void* data;
            vkMapMemory(m_device, m_uniformBuffersMemory[i], 0, sizeof(ubo), 0, &data);
            memcpy(data, &ubo, sizeof(ubo));
            vkUnmapMemory(m_device, m_uniformBuffersMemory[i]);
        }
    }

    void createImage(
        uint32_t              width,
        uint32_t              height,
        VkFormat              format,
        VkImageTiling         tiling,
        VkImageUsageFlags     usage,
        VkMemoryPropertyFlags properties,
        VkImage&              image,
        VkDeviceMemory&       imageMemory)
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType     = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width  = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth  = 1;
        imageInfo.mipLevels     = m_imageMipLevels;
        imageInfo.arrayLayers   = 1;
        imageInfo.format        = format;
        imageInfo.tiling        = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage         = usage;
        imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

        VkExternalMemoryImageCreateInfo vkExternalMemImageCreateInfo = {};
        vkExternalMemImageCreateInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        vkExternalMemImageCreateInfo.pNext       = NULL;
        vkExternalMemImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

        imageInfo.pNext = &vkExternalMemImageCreateInfo;

        if (vkCreateImage(m_device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(m_device, image, &memRequirements);

        WindowsSecurityAttributes winSecurityAttributes;

        VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
        vulkanExportMemoryWin32HandleInfoKHR.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        vulkanExportMemoryWin32HandleInfoKHR.pNext       = NULL;
        vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
        vulkanExportMemoryWin32HandleInfoKHR.dwAccess    = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
        vulkanExportMemoryWin32HandleInfoKHR.name        = (LPCWSTR)NULL;

        VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
        vulkanExportMemoryAllocateInfoKHR.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
        vulkanExportMemoryAllocateInfoKHR.pNext       = &vulkanExportMemoryWin32HandleInfoKHR;
        vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memRequirements.size;
        allocInfo.pNext           = &vulkanExportMemoryAllocateInfoKHR;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        VkMemoryRequirements vkMemoryRequirements = {};
        vkGetImageMemoryRequirements(m_device, image, &vkMemoryRequirements);
        m_totalImageMemSize = vkMemoryRequirements.size;

        if (vkAllocateMemory(m_device, &allocInfo, nullptr, &m_textureImageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(m_device, image, m_textureImageMemory, 0);
    }

    // cuda VK import semaphore
    // - Import CUDA-update-VK-semaphore
    // - Import VK-update-CUDA-semaphore
    void cudaVkImportSemaphore()
    {
        // ------------------------------ Import CUDA-update-VK-semaphore --------------------------------
        // cuda external semaphore handle desc
        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
        memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));

        // Get VK semaphore handle for "m_cudaUpdateVkSemaphore"
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT, m_cudaUpdateVkSemaphore);

        externalSemaphoreHandleDesc.flags = 0;

        // cuda import external semaphore
        checkCudaErrors(cudaImportExternalSemaphore(&m_cudaExtCudaUpdateVkSemaphore,  // out
                                                    &externalSemaphoreHandleDesc)); // in

        // ------------------------------ Import VK-update-CUDA-semaphore --------------------------------
        // Reset desc
        memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));

        // Get VK semaphore handle for "vkUpdateCudaSemaphore"
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT, m_vkUpdateCudaSemaphore);

        externalSemaphoreHandleDesc.flags = 0;

        // cuda import external semaphore
        checkCudaErrors(cudaImportExternalSemaphore(&m_cudaExtVkUpdateCudaSemaphore,  // out
                                                    &externalSemaphoreHandleDesc)); // in

        printf("CUDA Imported Vulkan semaphore\n");
    }

    // cuda VK import image memory
    // - Get external memory buffer
    // - Get external mipmapped array image
    // - Create cuda surface object for each mip level image
    // - Create cuda texture image
    // - Create cuda surface object
    void cudaVkImportImageMem()
    {
        // ------------- Get ext memory buffer ---------------
        // Cuda external memory handle desc
        cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
        memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));

        // Set type and get handle
        cudaExtMemHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        cudaExtMemHandleDesc.handle.win32.handle = getVkImageMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT);

        // Set size to "vkGetImageMemoryRequirements" of the image
        cudaExtMemHandleDesc.size                = m_totalImageMemSize;

        // Import the external memory to cuda, get a cuda buffer
        checkCudaErrors(cudaImportExternalMemory(
            &m_cudaExtMemImageBuffer, // cudaExternalMemory_t*               extMem_out,
            &cudaExtMemHandleDesc));  // const cudaExternalMemoryHandleDesc* memHandleDesc

        // --------------- Get ext mipmapped array image -----------------
        // Cuda external memory mipmapped array desc
        cudaExternalMemoryMipmappedArrayDesc externalMemoryMipmappedArrayDesc;
        memset(&externalMemoryMipmappedArrayDesc, 0, sizeof(externalMemoryMipmappedArrayDesc));

        // Specify image extent and format
        cudaExtent extent = make_cudaExtent(m_imageWidth, m_imageHeight, 0);
        cudaChannelFormatDesc formatDesc;
        formatDesc.x = 8;
        formatDesc.y = 8;
        formatDesc.z = 8;
        formatDesc.w = 8;
        formatDesc.f = cudaChannelFormatKindUnsigned;

        // Fill the external memory mipmapped array desc
        externalMemoryMipmappedArrayDesc.offset     = 0;
        externalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
        externalMemoryMipmappedArrayDesc.extent     = extent;
        externalMemoryMipmappedArrayDesc.flags      = 0;
        externalMemoryMipmappedArrayDesc.numLevels  = m_imageMipLevels;

        // Get "m_cudaMipmappedImageArray"
        checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(
            &m_cudaMipmappedImageArray,          // cudaMipmappedArray_t*                       mipmap
            m_cudaExtMemImageBuffer,             // cudaExternalMemory_t                        extMem
            &externalMemoryMipmappedArrayDesc)); // const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc


        // ----------------- Create surface object for each mip level ------------------------
        for (unsigned int mipLevelIdx = 0; mipLevelIdx < m_imageMipLevels; mipLevelIdx++)
        {
            // Cuda array
            cudaArray_t cudaMipLevelArray;
            checkCudaErrors(cudaGetMipmappedArrayLevel(&cudaMipLevelArray, m_cudaMipmappedImageArray, mipLevelIdx));

            // Get proper width, height
            //uint32_t width  = (m_imageWidth  >> mipLevelIdx) ? (m_imageWidth  >> mipLevelIdx) : 1;
            //uint32_t height = (m_imageHeight >> mipLevelIdx) ? (m_imageHeight >> mipLevelIdx) : 1;

            // Resource desc
            cudaResourceDesc resourceDesc;
            memset(&resourceDesc, 0, sizeof(resourceDesc));
            resourceDesc.resType         = cudaResourceTypeArray;
            resourceDesc.res.array.array = cudaMipLevelArray;

            // Surface object
            SurfObj surfaceObject;
            checkCudaErrors(cudaCreateSurfaceObject(&surfaceObject, &resourceDesc));
            m_surfaceObjectList.push_back(surfaceObject);
        }

        // ------------------------------ Alloc and copy two buffers for surface ------------------------------
        // Alloc surface memory
        checkCudaErrors(cudaMalloc((void**)&d_surfaceObjectList, sizeof(SurfObj) * m_imageMipLevels));

        // Copy surface buffer to GPU
        checkCudaErrors(cudaMemcpy(
            d_surfaceObjectList,
            m_surfaceObjectList.data(),
            sizeof(SurfObj) * m_imageMipLevels,
            cudaMemcpyHostToDevice));

        printf("CUDA Kernel Vulkan image buffer\n");
    }

    void transitionImageLayout(
        VkImage       image,
        VkFormat      format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier = {};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                       = oldLayout;
        barrier.newLayout                       = newLayout;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = image;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = m_imageMipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage           = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                 newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage           = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage      = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                 newLayout == VK_IMAGE_LAYOUT_GENERAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;

            sourceStage           = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage      = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
        else
        {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage,
            destinationStage,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier);

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region = {};
        region.bufferOffset                    = 0;
        region.bufferRowLength                 = 0;
        region.bufferImageHeight               = 0;
        region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel       = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount     = 1;
        region.imageOffset                     = {0, 0, 0};
        region.imageExtent                     = {width, height, 1};

        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region);

        endSingleTimeCommands(commandBuffer);
    }
};

int main()
{
    HelloTriangleApplication app;

    try
    {
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
