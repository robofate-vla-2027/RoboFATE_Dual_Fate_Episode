#include "copy_plan.hpp"
#include "deployment_binding.hpp"
#include <vulkan/vk_layer.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unistd.h>

namespace {
std::mutex mutex_, log_mutex;
void* key(const void* h) {return *static_cast<void* const*>(h);}
void logline(const char* fmt,...) {
    std::lock_guard<std::mutex> lock(log_mutex);
    static FILE* file=[](){const char* p=std::getenv("REFATE_VK_COPY_LOG");return p ? std::fopen(p,"a") : nullptr;}();
    if(!file) return;
    std::fprintf(file,"pid=%d ",getpid());va_list args;va_start(args,fmt);std::vfprintf(file,fmt,args);va_end(args);std::fputc('\n',file);std::fflush(file);
}
struct Instance {VkInstance handle;PFN_vkGetInstanceProcAddr gipa;PFN_GetPhysicalDeviceProcAddr phys;PFN_vkDestroyInstance destroy;};
struct Buffer {VkBuffer buffer;VkDeviceMemory memory;VkDeviceSize bytes;};
struct Device {
    VkDevice handle;PFN_vkGetDeviceProcAddr gdpa;PFN_vkDestroyDevice destroy;VkPhysicalDeviceMemoryProperties memory;
    bool stage=true;VkDeviceSize live=0;uint64_t count=0;
    std::unordered_map<VkImage,VkImageCreateInfo> images;
    std::unordered_map<VkCommandBuffer,VkCommandPool> pools;
    std::unordered_map<VkCommandBuffer,std::vector<Buffer>> owned;
    template<class T>T get(const char* name) const {return reinterpret_cast<T>(gdpa(handle,name));}
};
std::unordered_map<void*,std::shared_ptr<Instance>> instances;
std::unordered_map<void*,std::shared_ptr<Device>> devices;
std::shared_ptr<Instance> inst(const void* h){std::lock_guard<std::mutex> l(mutex_);auto i=instances.find(key(h));return i==instances.end()?nullptr:i->second;}
std::shared_ptr<Device> dev(const void* h){std::lock_guard<std::mutex> l(mutex_);auto i=devices.find(key(h));return i==devices.end()?nullptr:i->second;}
void release(const std::shared_ptr<Device>& d, const std::vector<VkCommandBuffer>& cbs) {
    std::vector<Buffer> old;
    {std::lock_guard<std::mutex> l(mutex_);for(auto cb:cbs){auto i=d->owned.find(cb);if(i==d->owned.end())continue;for(auto b:i->second){old.push_back(b);d->live-=b.bytes;}d->owned.erase(i);}}
    for(auto b:old){d->get<PFN_vkDestroyBuffer>("vkDestroyBuffer")(d->handle,b.buffer,nullptr);d->get<PFN_vkFreeMemory>("vkFreeMemory")(d->handle,b.memory,nullptr);}
    if(!old.empty())logline("RECLAIM buffers=%zu",old.size());
}
std::vector<VkCommandBuffer> pool_buffers(const std::shared_ptr<Device>& d,VkCommandPool pool,bool erase) {
    std::vector<VkCommandBuffer> out;std::lock_guard<std::mutex> l(mutex_);
    for(auto i=d->pools.begin();i!=d->pools.end();){if(i->second==pool){out.push_back(i->first);if(erase){i=d->pools.erase(i);continue;}}++i;}return out;
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL L_GetInstanceProcAddr(VkInstance,const char*);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL L_GetDeviceProcAddr(VkDevice,const char*);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL L_GetPhysicalDeviceProcAddr(VkInstance h,const char* name){auto s=inst(h);return s?(s->phys?s->phys(h,name):s->gipa(h,name)):nullptr;}
VKAPI_ATTR VkResult VKAPI_CALL L_CreateInstance(const VkInstanceCreateInfo* info,const VkAllocationCallbacks* alloc,VkInstance* out) {
    char host[128]{};gethostname(host,sizeof(host)-1);const char* alias=std::getenv("ROBOFATE_SSH_ALIAS");
    if(std::strcmp(host,ROBOFATE_EXPECTED_HOST) || !alias || std::strcmp(alias,ROBOFATE_EXPECTED_ALIAS) || !std::getenv("REFATE_VK_COPY_LOG") || !std::getenv("ROBOFATE_EXECUTION_OWNER") || std::strcmp(std::getenv("ROBOFATE_EXECUTION_OWNER"),ROBOFATE_EXPECTED_OWNER))return VK_ERROR_INITIALIZATION_FAILED;
    auto chain=reinterpret_cast<const VkLayerInstanceCreateInfo*>(info->pNext);
    while(chain && (chain->sType!=VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO || chain->function!=VK_LAYER_LINK_INFO))chain=reinterpret_cast<const VkLayerInstanceCreateInfo*>(chain->pNext);
    if(!chain || !chain->u.pLayerInfo)return VK_ERROR_INITIALIZATION_FAILED;
    auto next=chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;auto phys=chain->u.pLayerInfo->pfnNextGetPhysicalDeviceProcAddr;
    const_cast<VkLayerInstanceCreateInfo*>(chain)->u.pLayerInfo=chain->u.pLayerInfo->pNext;
    auto fn=reinterpret_cast<PFN_vkCreateInstance>(next(VK_NULL_HANDLE,"vkCreateInstance"));
    VkResult result=fn(info,alloc,out);if(result==VK_SUCCESS){std::lock_guard<std::mutex> l(mutex_);instances[key(*out)]=std::make_shared<Instance>(Instance{*out,next,phys,reinterpret_cast<PFN_vkDestroyInstance>(next(*out,"vkDestroyInstance"))});}logline("CREATE_INSTANCE result=%d",result);return result;
}
VKAPI_ATTR void VKAPI_CALL L_DestroyInstance(VkInstance h,const VkAllocationCallbacks* a){auto s=inst(h);if(!s)return;void* saved=key(h);logline("DESTROY_INSTANCE_BEGIN next=%p",reinterpret_cast<void*>(s->destroy));s->destroy(h,a);{std::lock_guard<std::mutex> l(mutex_);instances.erase(saved);}logline("DESTROY_INSTANCE_RETURNED");}
VKAPI_ATTR VkResult VKAPI_CALL L_CreateDevice(VkPhysicalDevice physical,const VkDeviceCreateInfo* info,const VkAllocationCallbacks* alloc,VkDevice* out){
    auto s=inst(physical);if(!s)return VK_ERROR_INITIALIZATION_FAILED;
    auto chain=reinterpret_cast<const VkLayerDeviceCreateInfo*>(info->pNext);
    while(chain && (chain->sType!=VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO || chain->function!=VK_LAYER_LINK_INFO))chain=reinterpret_cast<const VkLayerDeviceCreateInfo*>(chain->pNext);
    if(!chain || !chain->u.pLayerInfo)return VK_ERROR_INITIALIZATION_FAILED;
    auto next=chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;auto gdpa=chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    const_cast<VkLayerDeviceCreateInfo*>(chain)->u.pLayerInfo=chain->u.pLayerInfo->pNext;
    auto create=reinterpret_cast<PFN_vkCreateDevice>(next(s->handle,"vkCreateDevice"));VkResult result=create(physical,info,alloc,out);
    if(result==VK_SUCCESS){auto d=std::make_shared<Device>();d->handle=*out;d->gdpa=gdpa;d->destroy=reinterpret_cast<PFN_vkDestroyDevice>(gdpa(*out,"vkDestroyDevice"));const char* mode=std::getenv("REFATE_VK_COPY_MODE");d->stage=mode && !std::strcmp(mode,"stage");
        reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(s->gipa(s->handle,"vkGetPhysicalDeviceMemoryProperties"))(physical,&d->memory);
        for(auto p=reinterpret_cast<const VkBaseInStructure*>(info->pNext);p;p=p->pNext)if(p->sType==VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO && reinterpret_cast<const VkDeviceGroupDeviceCreateInfo*>(p)->physicalDeviceCount>1)d->stage=false;
        {std::lock_guard<std::mutex> l(mutex_);devices[key(*out)]=d;}logline("CREATE_DEVICE stage=%d",d->stage);
    }return result;
}
VKAPI_ATTR void VKAPI_CALL L_DestroyDevice(VkDevice h,const VkAllocationCallbacks* a){auto d=dev(h);if(!d)return;std::vector<VkCommandBuffer> all;{std::lock_guard<std::mutex> l(mutex_);for(auto& x:d->owned)all.push_back(x.first);}release(d,all);void* saved=key(h);logline("DESTROY_DEVICE_BEGIN total_copies=%llu live_bytes=%llu next=%p",(unsigned long long)d->count,(unsigned long long)d->live,reinterpret_cast<void*>(d->destroy));d->destroy(h,a);{std::lock_guard<std::mutex> l(mutex_);devices.erase(saved);}logline("DESTROY_DEVICE_RETURNED");}
VKAPI_ATTR VkResult VKAPI_CALL L_CreateImage(VkDevice h,const VkImageCreateInfo* info,const VkAllocationCallbacks* a,VkImage* out){auto d=dev(h);auto result=d->get<PFN_vkCreateImage>("vkCreateImage")(h,info,a,out);if(result==VK_SUCCESS){auto copy=*info;copy.pNext=nullptr;copy.pQueueFamilyIndices=nullptr;std::lock_guard<std::mutex> l(mutex_);d->images[*out]=copy;}return result;}
VKAPI_ATTR void VKAPI_CALL L_DestroyImage(VkDevice h,VkImage image,const VkAllocationCallbacks* a){auto d=dev(h);{std::lock_guard<std::mutex> l(mutex_);d->images.erase(image);}d->get<PFN_vkDestroyImage>("vkDestroyImage")(h,image,a);}
VKAPI_ATTR VkResult VKAPI_CALL L_AllocateCommandBuffers(VkDevice h,const VkCommandBufferAllocateInfo* info,VkCommandBuffer* out){auto d=dev(h);auto result=d->get<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(h,info,out);if(result==VK_SUCCESS){std::lock_guard<std::mutex> l(mutex_);for(uint32_t i=0;i<info->commandBufferCount;++i)d->pools[out[i]]=info->commandPool;}return result;}
VKAPI_ATTR VkResult VKAPI_CALL L_BeginCommandBuffer(VkCommandBuffer cb,const VkCommandBufferBeginInfo* info){auto d=dev(cb);auto result=d->get<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(cb,info);if(result==VK_SUCCESS)release(d,{cb});return result;}
VKAPI_ATTR VkResult VKAPI_CALL L_ResetCommandBuffer(VkCommandBuffer cb,VkCommandBufferResetFlags flags){auto d=dev(cb);auto result=d->get<PFN_vkResetCommandBuffer>("vkResetCommandBuffer")(cb,flags);if(result==VK_SUCCESS)release(d,{cb});return result;}
VKAPI_ATTR VkResult VKAPI_CALL L_ResetCommandPool(VkDevice h,VkCommandPool pool,VkCommandPoolResetFlags flags){auto d=dev(h);auto result=d->get<PFN_vkResetCommandPool>("vkResetCommandPool")(h,pool,flags);if(result==VK_SUCCESS)release(d,pool_buffers(d,pool,false));return result;}
VKAPI_ATTR void VKAPI_CALL L_FreeCommandBuffers(VkDevice h,VkCommandPool pool,uint32_t n,const VkCommandBuffer* cbs){auto d=dev(h);d->get<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(h,pool,n,cbs);release(d,std::vector<VkCommandBuffer>(cbs,cbs+n));std::lock_guard<std::mutex> l(mutex_);for(uint32_t i=0;i<n;++i)d->pools.erase(cbs[i]);}
VKAPI_ATTR void VKAPI_CALL L_DestroyCommandPool(VkDevice h,VkCommandPool pool,const VkAllocationCallbacks* a){auto d=dev(h);d->get<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(h,pool,a);release(d,pool_buffers(d,pool,true));}
VKAPI_ATTR void VKAPI_CALL L_CmdCopyImage(VkCommandBuffer cb,VkImage src,VkImageLayout sl,VkImage dst,VkImageLayout dl,uint32_t n,const VkImageCopy* regions){
    auto d=dev(cb);VkImageCreateInfo si{},di{};uint64_t id=0;bool known=false;
    {std::lock_guard<std::mutex> l(mutex_);id=++d->count;auto a=d->images.find(src),b=d->images.find(dst);known=a!=d->images.end()&&b!=d->images.end();if(known){si=a->second;di=b->second;}}
    CopyPlan p;bool supported=known && make_plan(si,di,n,regions,p);
    logline("COPY id=%llu cb=%p src=%p dst=%p formats=%d,%d samples=%d,%d regions=%u bytes=%llu supported=%d stage=%d",(unsigned long long)id,(void*)cb,(void*)src,(void*)dst,si.format,di.format,si.samples,di.samples,n,(unsigned long long)p.bytes,supported,d->stage);
    for(uint32_t i=0;i<n;++i)logline("REGION id=%llu index=%u extent=%u,%u,%u src_mip=%u dst_mip=%u layers=%u src_offset=%d,%d,%d dst_offset=%d,%d,%d",(unsigned long long)id,i,regions[i].extent.width,regions[i].extent.height,regions[i].extent.depth,regions[i].srcSubresource.mipLevel,regions[i].dstSubresource.mipLevel,regions[i].srcSubresource.layerCount,regions[i].srcOffset.x,regions[i].srcOffset.y,regions[i].srcOffset.z,regions[i].dstOffset.x,regions[i].dstOffset.y,regions[i].dstOffset.z);
    auto original=[&](){d->get<PFN_vkCmdCopyImage>("vkCmdCopyImage")(cb,src,sl,dst,dl,n,regions);};
    if(!supported || !d->stage){logline("ORIGINAL id=%llu",(unsigned long long)id);original();return;}
    Buffer b{};VkResult result=VK_SUCCESS;bool admitted=false;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};info.size=p.bytes;info.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    result=d->get<PFN_vkCreateBuffer>("vkCreateBuffer")(d->handle,&info,nullptr,&b.buffer);
    if(result==VK_SUCCESS){VkMemoryRequirements req{};d->get<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(d->handle,b.buffer,&req);b.bytes=req.size;
        {std::lock_guard<std::mutex> l(mutex_);admitted=b.bytes<=512ull*1024*1024 && d->live<=2ull*1024*1024*1024-b.bytes;if(admitted)d->live+=b.bytes;}
        uint32_t type=UINT32_MAX;for(uint32_t i=0;i<d->memory.memoryTypeCount;++i)if((req.memoryTypeBits&(1u<<i)) && (d->memory.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)){type=i;break;}
        if(!admitted || type==UINT32_MAX)result=VK_ERROR_OUT_OF_DEVICE_MEMORY;
        else {VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};alloc.allocationSize=req.size;alloc.memoryTypeIndex=type;result=d->get<PFN_vkAllocateMemory>("vkAllocateMemory")(d->handle,&alloc,nullptr,&b.memory);if(result==VK_SUCCESS)result=d->get<PFN_vkBindBufferMemory>("vkBindBufferMemory")(d->handle,b.buffer,b.memory,0);}
    }
    if(result!=VK_SUCCESS){if(b.buffer)d->get<PFN_vkDestroyBuffer>("vkDestroyBuffer")(d->handle,b.buffer,nullptr);if(b.memory)d->get<PFN_vkFreeMemory>("vkFreeMemory")(d->handle,b.memory,nullptr);if(admitted){std::lock_guard<std::mutex> l(mutex_);d->live-=b.bytes;}logline("ALLOCATION_FAILURE id=%llu result=%d original_operation_preserved=1",(unsigned long long)id,result);original();return;}
    {std::lock_guard<std::mutex> l(mutex_);d->owned[cb].push_back(b);}
    d->get<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(cb,src,sl,b.buffer,n,p.src.data());
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;barrier.buffer=b.buffer;barrier.offset=0;barrier.size=VK_WHOLE_SIZE;
    d->get<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    d->get<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(cb,b.buffer,dst,dl,n,p.dst.data());
    logline("STAGED id=%llu allocated=%llu",(unsigned long long)id,(unsigned long long)b.bytes);
}
PFN_vkVoidFunction hook(const char* name) {
#define HOOK(n) if(!std::strcmp(name,"vk" #n))return reinterpret_cast<PFN_vkVoidFunction>(L_##n);
    HOOK(GetInstanceProcAddr) HOOK(GetDeviceProcAddr) HOOK(CreateInstance) HOOK(DestroyInstance) HOOK(CreateDevice) HOOK(DestroyDevice)
    HOOK(CreateImage) HOOK(DestroyImage) HOOK(AllocateCommandBuffers) HOOK(BeginCommandBuffer) HOOK(ResetCommandBuffer)
    HOOK(ResetCommandPool) HOOK(FreeCommandBuffers) HOOK(DestroyCommandPool) HOOK(CmdCopyImage)
#undef HOOK
    return nullptr;
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL L_GetDeviceProcAddr(VkDevice h,const char* name){if(auto f=hook(name))return f;auto d=dev(h);return d?d->gdpa(h,name):nullptr;}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL L_GetInstanceProcAddr(VkInstance h,const char* name){if(auto f=hook(name))return f;if(!h)return nullptr;auto s=inst(h);return s?s->gipa(h,name):nullptr;}
}
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v){if(!v || v->loaderLayerInterfaceVersion<2)return VK_ERROR_INITIALIZATION_FAILED;v->loaderLayerInterfaceVersion=2;v->pfnGetInstanceProcAddr=L_GetInstanceProcAddr;v->pfnGetDeviceProcAddr=L_GetDeviceProcAddr;v->pfnGetPhysicalDeviceProcAddr=L_GetPhysicalDeviceProcAddr;return VK_SUCCESS;}
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance h,const char* n){return L_GetInstanceProcAddr(h,n);}
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice h,const char* n){return L_GetDeviceProcAddr(h,n);}
