#pragma once
#include <vulkan/vulkan.h>
#include <algorithm>
#include <numeric>
#include <vector>
struct Block { uint32_t bytes, w, h; };
inline Block block(VkFormat f) {
    switch(f) {
#define FMT(x,n) case VK_FORMAT_##x: return {n,1,1};
    FMT(R8_UNORM,1) FMT(R8_SNORM,1) FMT(R8_UINT,1) FMT(R8_SINT,1) FMT(R8_SRGB,1)
    FMT(R8G8_UNORM,2) FMT(R8G8_SNORM,2) FMT(R8G8_UINT,2) FMT(R8G8_SINT,2) FMT(R8G8_SRGB,2)
    FMT(R8G8B8_UNORM,3) FMT(R8G8B8_SRGB,3) FMT(B8G8R8_UNORM,3) FMT(B8G8R8_SRGB,3)
    FMT(R8G8B8A8_UNORM,4) FMT(R8G8B8A8_SNORM,4) FMT(R8G8B8A8_UINT,4) FMT(R8G8B8A8_SINT,4) FMT(R8G8B8A8_SRGB,4)
    FMT(B8G8R8A8_UNORM,4) FMT(B8G8R8A8_SRGB,4) FMT(A8B8G8R8_UNORM_PACK32,4) FMT(A8B8G8R8_SRGB_PACK32,4)
    FMT(A2R10G10B10_UNORM_PACK32,4) FMT(A2B10G10R10_UNORM_PACK32,4)
    FMT(B10G11R11_UFLOAT_PACK32,4) FMT(E5B9G9R9_UFLOAT_PACK32,4)
    FMT(R16_UNORM,2) FMT(R16_SNORM,2) FMT(R16_UINT,2) FMT(R16_SINT,2) FMT(R16_SFLOAT,2)
    FMT(R16G16_UNORM,4) FMT(R16G16_SNORM,4) FMT(R16G16_UINT,4) FMT(R16G16_SINT,4) FMT(R16G16_SFLOAT,4)
    FMT(R16G16B16_SFLOAT,6) FMT(R16G16B16A16_UNORM,8) FMT(R16G16B16A16_SNORM,8)
    FMT(R16G16B16A16_UINT,8) FMT(R16G16B16A16_SINT,8) FMT(R16G16B16A16_SFLOAT,8)
    FMT(R32_UINT,4) FMT(R32_SINT,4) FMT(R32_SFLOAT,4)
    FMT(R32G32_UINT,8) FMT(R32G32_SINT,8) FMT(R32G32_SFLOAT,8)
    FMT(R32G32B32_UINT,12) FMT(R32G32B32_SINT,12) FMT(R32G32B32_SFLOAT,12)
    FMT(R32G32B32A32_UINT,16) FMT(R32G32B32A32_SINT,16) FMT(R32G32B32A32_SFLOAT,16)
#undef FMT
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK: case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK: case VK_FORMAT_BC4_SNORM_BLOCK: return {8,4,4};
    case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK: case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_BC6H_UFLOAT_BLOCK: case VK_FORMAT_BC6H_SFLOAT_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK: case VK_FORMAT_BC7_SRGB_BLOCK: return {16,4,4};
    default: return {0,0,0};
    }
}
struct CopyPlan {
    VkDeviceSize bytes=0;
    std::vector<VkBufferImageCopy> src, dst;
};
inline bool inside(const VkImageCreateInfo& image, const VkImageSubresourceLayers& sub,
                   const VkOffset3D& offset, const VkExtent3D& extent, Block b) {
    if(sub.aspectMask!=VK_IMAGE_ASPECT_COLOR_BIT || !sub.layerCount || sub.mipLevel>=image.mipLevels ||
       sub.baseArrayLayer>=image.arrayLayers || sub.layerCount>image.arrayLayers-sub.baseArrayLayer ||
       offset.x<0 || offset.y<0 || offset.z<0 || !extent.width || !extent.height || !extent.depth) return false;
    uint32_t w=std::max(1u,image.extent.width>>sub.mipLevel),h=std::max(1u,image.extent.height>>sub.mipLevel),d=std::max(1u,image.extent.depth>>sub.mipLevel);
    if(uint32_t(offset.x)>w || extent.width>w-uint32_t(offset.x) || uint32_t(offset.y)>h || extent.height>h-uint32_t(offset.y) || uint32_t(offset.z)>d || extent.depth>d-uint32_t(offset.z)) return false;
    return uint32_t(offset.x)%b.w==0 && uint32_t(offset.y)%b.h==0 &&
           (extent.width%b.w==0 || uint32_t(offset.x)+extent.width==w) &&
           (extent.height%b.h==0 || uint32_t(offset.y)+extent.height==h);
}
inline bool make_plan(const VkImageCreateInfo& a, const VkImageCreateInfo& b, uint32_t n,
                      const VkImageCopy* regions, CopyPlan& p) {
    p={}; Block f=block(a.format);
    constexpr VkDeviceSize cap=512ull*1024*1024;
    constexpr VkImageCreateFlags excluded=VK_IMAGE_CREATE_SPARSE_BINDING_BIT|VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT|VK_IMAGE_CREATE_PROTECTED_BIT|VK_IMAGE_CREATE_DISJOINT_BIT;
    if(!f.bytes || a.format!=b.format || a.imageType!=b.imageType || a.samples!=VK_SAMPLE_COUNT_1_BIT || b.samples!=VK_SAMPLE_COUNT_1_BIT ||
       ((a.flags|b.flags)&excluded) || !(a.usage&VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || !(b.usage&VK_IMAGE_USAGE_TRANSFER_DST_BIT) || !n || n>4096 || !regions) return false;
    for(uint32_t i=0;i<n;++i) {
        const auto& r=regions[i];
        if(r.srcSubresource.layerCount!=r.dstSubresource.layerCount || !inside(a,r.srcSubresource,r.srcOffset,r.extent,f) || !inside(b,r.dstSubresource,r.dstOffset,r.extent,f)) return false;
        VkDeviceSize bytes=(VkDeviceSize(r.extent.width)+f.w-1)/f.w;
        for(VkDeviceSize mul : {(VkDeviceSize(r.extent.height)+f.h-1)/f.h,VkDeviceSize(r.extent.depth),VkDeviceSize(r.srcSubresource.layerCount),VkDeviceSize(f.bytes)}) {
            if(mul>cap || bytes>cap/mul) return false;
            bytes*=mul;
        }
        VkDeviceSize align=std::lcm(4u,f.bytes),offset=(p.bytes+align-1)/align*align;
        if(offset>cap || bytes>cap-offset) return false;
        VkBufferImageCopy x{};x.bufferOffset=offset;x.imageSubresource=r.srcSubresource;x.imageOffset=r.srcOffset;x.imageExtent=r.extent;
        p.src.push_back(x);x.imageSubresource=r.dstSubresource;x.imageOffset=r.dstOffset;p.dst.push_back(x);p.bytes=offset+bytes;
    }
    return true;
}
