// Synthetic CPU-only planning checks. No Vulkan loader, device or rendering calls.
#include "../robofate_pipeline/native/vulkan/copy_plan.hpp"
#include <cassert>

int main() {
    VkImageCreateInfo a{}, b{};
    a.imageType = b.imageType = VK_IMAGE_TYPE_2D;
    a.format = b.format = VK_FORMAT_R8G8B8A8_UNORM;
    a.extent = b.extent = {16, 16, 1};
    a.mipLevels = b.mipLevels = 5;
    a.arrayLayers = b.arrayLayers = 2;
    a.samples = b.samples = VK_SAMPLE_COUNT_1_BIT;
    a.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    b.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageCopy r{};
    r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.extent = {16, 16, 1};
    CopyPlan plan;
    assert(make_plan(a, b, 1, &r, plan) && plan.bytes == 1024);
    assert(plan.src.size() == 1 && plan.dst.size() == 1);
    r.srcOffset.x = -1;
    assert(!make_plan(a, b, 1, &r, plan));
    r.srcOffset.x = 1;
    assert(!make_plan(a, b, 1, &r, plan));
    r.srcOffset.x = 0;
    b.samples = VK_SAMPLE_COUNT_4_BIT;
    assert(!make_plan(a, b, 1, &r, plan));
    b.samples = VK_SAMPLE_COUNT_1_BIT;
    b.flags = VK_IMAGE_CREATE_PROTECTED_BIT;
    assert(!make_plan(a, b, 1, &r, plan));
    b.flags = 0;
    r.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    assert(!make_plan(a, b, 1, &r, plan));
    r.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r.srcSubresource.layerCount = 2;
    assert(!make_plan(a, b, 1, &r, plan));
    r.dstSubresource.layerCount = 2;
    assert(make_plan(a, b, 1, &r, plan) && plan.bytes == 2048);
    assert(!make_plan(a, b, 0, &r, plan));
    assert(!make_plan(a, b, 4097, &r, plan));
    assert(!make_plan(a, b, 1, nullptr, plan));
    a.format = b.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    assert(make_plan(a, b, 1, &r, plan) && plan.bytes == 256);
    r.srcOffset.x = 1;
    r.extent.width = 12;
    assert(!make_plan(a, b, 1, &r, plan));
    r.srcOffset.x = 0;
    a.extent = b.extent = {15, 16, 1};
    r.extent.width = 15;
    assert(make_plan(a, b, 1, &r, plan) && plan.bytes == 256);
    a.format = b.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    a.extent = b.extent = {65536, 65536, 1};
    r.extent = a.extent;
    assert(!make_plan(a, b, 1, &r, plan));
    return 0;
}
