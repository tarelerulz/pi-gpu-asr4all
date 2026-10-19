/* ygpu.c — a small "run a list of GPU jobs" library for the Raspberry Pi 4 / 400 GPU (V3D 4.2, Vulkan).
 *
 * It knows nothing about the model: Python (a4a_gpu.py) plans the jobs, this file runs them.
 * Two buffers: the ARENA (all activations, read+write, binding 0) and the WEIGHTS (read only, binding 1).
 * Every job = one shader + 32 uint push constants (offsets into the buffers, sizes) + a dispatch size.
 *
 *   yg_init(shader_dir, names, n_names, arena_floats, weights, n_weights)  -> 0 ok
 *   yg_arena()                         -> pointer to the mapped arena (Python reads/writes it with numpy)
 *   yg_plan(n, pipe, push[n*32], groups[n*3])   record the job list once (with timestamps between jobs)
 *   yg_run(&gpu_ms)                    submit the recorded list, wait
 *   yg_times(ms[n])                    per-job GPU time of the last run, from timestamps
 *   yg_add_set(weights, n, arena_off)  a second graph in the same context: own weights buffer, arena from
 *                                      arena_off floats on; yg_set(k) picks the set the next yg_plan binds
 * Build: make lib   (links the system Vulkan loader, or VK=/dir/with/libvulkan.so)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "ygpu: %s failed: %d (line %d)\n", #x, r_, __LINE__); return -1; } } while (0)
#define MAXP 160
#define MAXS 4                                   /* plan slots: 2 graphs x 2 ring parities */

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; size_t size; } Buf;

static VkInstance inst; static VkPhysicalDevice phys; static VkDevice dev; static VkQueue queue;
static VkCommandPool pool; static VkCommandBuffer cb, cbs[MAXS]; static VkPhysicalDeviceMemoryProperties memprops;
static VkDescriptorSetLayout dsl; static VkPipelineLayout layout; static VkDescriptorPool dpool;
static VkDescriptorSet sets[MAXS]; static int nsets, cur_set;
static VkPipeline pipes[MAXP]; static int npipes;
/* one timestamp pool per slot: growing a shared pool would invalidate the other slots' recorded buffers */
static VkQueryPool qpools[MAXS]; static int qcounts[MAXS], njobs_s[MAXS], cur; static float ts_period = 1.0f;
static Buf arena, wts[MAXS]; static VkDeviceSize ssbo_align = 1;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* prefer cached host memory for the arena: the CPU reads the outputs back, and uncached reads are ~10x slower */
static int buf_new(Buf *b, size_t size, int want_cached) {
    b->size = size;
    VkBufferCreateInfo bc = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
    CK(vkCreateBuffer(dev, &bc, NULL, &b->buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b->buf, &mr);
    uint32_t base = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int idx = -1;
    for (int pass = want_cached ? 0 : 1; pass < 2 && idx < 0; pass++) {
        uint32_t want = base | (pass == 0 ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0);
        for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
            if ((mr.memoryTypeBits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want) { idx = i; break; }
    }
    if (idx < 0) { fprintf(stderr, "ygpu: no host-visible memory type\n"); return -1; }
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = idx };
    CK(vkAllocateMemory(dev, &ai, NULL, &b->mem));
    CK(vkBindBufferMemory(dev, b->buf, b->mem, 0));
    CK(vkMapMemory(dev, b->mem, 0, size, 0, &b->map));
    return 0;
}

static int pipe_new(const char *dir, const char *name, const uint32_t *shape, VkPipeline *out) {
    char path[512]; snprintf(path, sizeof path, "%s/%s.spv", dir, name);
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    uint32_t *code = malloc(len); if (fread(code, 1, len, f) != (size_t)len) { fclose(f); free(code); return -1; } fclose(f);
    VkShaderModuleCreateInfo sm = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = len, .pCode = code };
    VkShaderModule mod; CK(vkCreateShaderModule(dev, &sm, NULL, &mod));
    VkSpecializationMapEntry entries[32];
    for (uint32_t i = 0; i < 32; i++)
        entries[i] = (VkSpecializationMapEntry){ .constantID = i, .offset = i * sizeof(uint32_t), .size = sizeof(uint32_t) };
    VkSpecializationInfo si = { .mapEntryCount = 32, .pMapEntries = entries,
        .dataSize = 32 * sizeof(uint32_t), .pData = shape };
    VkComputePipelineCreateInfo cp = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = layout,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main", .pSpecializationInfo = shape ? &si : NULL } };
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cp, NULL, out));
    vkDestroyShaderModule(dev, mod, NULL); free(code);
    return 0;
}

int yg_add_set(const float *weights, long n_weights, long arena_off);

static int init(const char *shader_dir, const char **names, int n_names, long arena_floats,
                const float *weights, long n_weights, const uint32_t *shapes) {
    if (dev) return -100;                        /* one context per process: all state below is global */
    if (n_names <= 0 || n_names > MAXP) return -1;
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "ygpu", .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ic = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    CK(vkCreateInstance(&ic, NULL, &inst));
    uint32_t n = 1; vkEnumeratePhysicalDevices(inst, &n, &phys);
    if (n == 0) { fprintf(stderr, "ygpu: no Vulkan GPU\n"); return -1; }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props); ts_period = props.limits.timestampPeriod;
    ssbo_align = props.limits.minStorageBufferOffsetAlignment;
    uint32_t nq = 8, qfam = 0; VkQueueFamilyProperties qp[8]; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
    while (qfam < nq && !(qp[qfam].queueFlags & VK_QUEUE_COMPUTE_BIT)) qfam++;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qc = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfam, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dc = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qc };
    CK(vkCreateDevice(phys, &dc, NULL, &dev));
    vkGetDeviceQueue(dev, qfam, 0, &queue);
    VkCommandPoolCreateInfo pc = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qfam, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
    CK(vkCreateCommandPool(dev, &pc, NULL, &pool));
    VkCommandBufferAllocateInfo ca = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = MAXS };
    CK(vkAllocateCommandBuffers(dev, &ca, cbs));
    cb = cbs[0];
    vkGetPhysicalDeviceMemoryProperties(phys, &memprops);

    VkDescriptorSetLayoutBinding lb[2];
    for (uint32_t i = 0; i < 2; i++) lb[i] = (VkDescriptorSetLayoutBinding){ .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dl = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = lb };
    CK(vkCreateDescriptorSetLayout(dev, &dl, NULL, &dsl));
    VkPushConstantRange pr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 32 * 4 };
    VkPipelineLayoutCreateInfo pl = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
    CK(vkCreatePipelineLayout(dev, &pl, NULL, &layout));
    for (int i = 0; i < n_names; i++) if (pipe_new(shader_dir, names[i], shapes ? shapes + (size_t)i * 32 : NULL, &pipes[i])) return -1;
    npipes = n_names;

    if (buf_new(&arena, (size_t)arena_floats * 4, 1)) return -1;
    memset(arena.map, 0, arena.size);

    VkDescriptorPoolSize ps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 * MAXS };
    VkDescriptorPoolCreateInfo dp = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = MAXS, .poolSizeCount = 1, .pPoolSizes = &ps };
    CK(vkCreateDescriptorPool(dev, &dp, NULL, &dpool));
    return yg_add_set(weights, n_weights, 0) == 0 ? 0 : -1;
}

/* Weights buffer + descriptor set for one graph; its arena binding starts arena_off floats in, so the graph's own
   offsets need no relocation. Returns the set index (0 = the one init made). */
int yg_add_set(const float *weights, long n_weights, long arena_off) {
    if (!dev || nsets >= MAXS) return -1;
    VkDeviceSize off = (VkDeviceSize)arena_off * 4;
    if (off % ssbo_align || off >= arena.size) { fprintf(stderr, "ygpu: arena offset %ld not aligned to %lu bytes\n", arena_off, (unsigned long)ssbo_align); return -1; }
    int k = nsets;
    if (buf_new(&wts[k], (size_t)(n_weights > 0 ? n_weights : 1) * 4, 0)) return -1;
    if (n_weights > 0) memcpy(wts[k].map, weights, (size_t)n_weights * 4);
    VkDescriptorSetAllocateInfo da = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    CK(vkAllocateDescriptorSets(dev, &da, &sets[k]));
    VkDescriptorBufferInfo bi[2] = { { .buffer = arena.buf, .offset = off, .range = VK_WHOLE_SIZE },
                                     { .buffer = wts[k].buf, .range = VK_WHOLE_SIZE } };
    for (uint32_t i = 0; i < 2; i++) {
        VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[k], .dstBinding = i, .descriptorCount = 1,
                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i] };
        vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
    }
    nsets++;
    return k;
}

int yg_set(int k) {
    if (k < 0 || k >= nsets) return -1;
    cur_set = k;
    return 0;
}

/* Keep the original ABI for callers that do not specialize pipelines. */
int yg_init(const char *dir, const char **names, int count, long arena_floats, const float *weights, long n_weights) {
    return init(dir, names, count, arena_floats, weights, n_weights, NULL);
}

int yg_init_shapes(const char *dir, const char **names, int count, long arena_floats,
                   const float *weights, long n_weights, const uint32_t *shapes) {
    return init(dir, names, count, arena_floats, weights, n_weights, shapes);
}

void *yg_arena(void) { return arena.map; }

int yg_plan(int n, const int *pipe, const uint32_t *push, const uint32_t *groups) {
    int no_ts = getenv("YG_NO_TS") != NULL;   /* per-job timestamps off (they serialize the queue) */
    if (n + 1 > qcounts[cur]) {
        if (qpools[cur]) { vkQueueWaitIdle(queue); vkDestroyQueryPool(dev, qpools[cur], NULL); qpools[cur] = VK_NULL_HANDLE; }
        VkQueryPoolCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = n + 1 };
        CK(vkCreateQueryPool(dev, &qi, NULL, &qpools[cur]));
        qcounts[cur] = n + 1;
    }
    VkQueryPool qpool = qpools[cur];
    njobs_s[cur] = n;
    CK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, qpool, 0, n + 1);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[cur_set], 0, NULL);
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 0);
    for (int i = 0; i < n; i++) {
        if (pipe[i] < 0 || pipe[i] >= npipes) { fprintf(stderr, "ygpu: job %d bad pipe %d\n", i, pipe[i]); return -1; }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[pipe[i]]);
        vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32 * 4, push + (size_t)i * 32);
        vkCmdDispatch(cb, groups[i * 3], groups[i * 3 + 1], groups[i * 3 + 2]);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        if (!no_ts || i == n - 1) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, i + 1);
    }
    VkMemoryBarrier hb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    CK(vkEndCommandBuffer(cb));
    return 0;
}

/* Recorded plans (double-buffered conv histories 2026-10-14; a second graph 2026-10-18): yg_slot picks the command
   buffer that the next yg_plan records and yg_run submits. Slot 0 is the default, so single-plan callers are unchanged. */
int yg_slot(int s) {
    if (s < 0 || s >= MAXS) return -1;
    cur = s;
    cb = cbs[s];
    return 0;
}

int yg_run(double *gpu_ms) {
    VkSubmitInfo su = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
    double t = now();
    CK(vkQueueSubmit(queue, 1, &su, VK_NULL_HANDLE));
    CK(vkQueueWaitIdle(queue));
    if (gpu_ms) *gpu_ms = (now() - t) * 1e3;
    return 0;
}

int yg_times(double *ms) {
    int njobs = njobs_s[cur];
    uint64_t *ts = malloc(sizeof(uint64_t) * (njobs + 1));
    VkResult r = vkGetQueryPoolResults(dev, qpools[cur], 0, njobs + 1, sizeof(uint64_t) * (njobs + 1), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) { free(ts); return -1; }
    for (int i = 0; i < njobs; i++) ms[i] = (double)(ts[i + 1] - ts[i]) * ts_period * 1e-6;
    free(ts);
    return 0;
}
