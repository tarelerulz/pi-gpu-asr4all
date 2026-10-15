VK ?=
# default: the system Vulkan loader (Raspberry Pi OS: apt install libvulkan-dev); VK=/dir = a private one
VKLINK = $(if $(wildcard $(VK)/libvulkan.so*),-L$(VK) -lvulkan -Wl$(comma)-rpath$(comma)$(VK),-lvulkan)
comma := ,
EWOPS = 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
SHADERS = gemm12p_00 gemm12p_01 gemm12p_02 gemm12p_10 gemm12p_11 gemm12p_12 dwconvp dwconv1p gemm12_00 gemm12_01 gemm12_02 gemm12_10 gemm12_11 gemm12_12 dwconv dwconv1 shiftsum copy4 tcopy bmm conv1d softmax rmax $(EWOPS:%=ew%)
all: libygpu.so $(SHADERS:%=shaders/%.spv)
lib: libygpu.so
libygpu.so: ygpu.c
	gcc -O2 -fPIC -shared -Wall -o libygpu.so ygpu.c $(VKLINK)
shaders/%.spv: shaders/%.comp shaders/common.glsl
	glslc --target-env=vulkan1.1 -O $< -o $@
shaders/ew%.spv: shaders/ew.comp shaders/common.glsl
	glslc --target-env=vulkan1.1 -O -DOP=$* $< -o $@
shaders/gemm12_%.spv: shaders/gemm12.comp shaders/common.glsl
	glslc --target-env=vulkan1.1 -O -DAV4=$(word 1,$(subst _, ,$(shell echo $* | sed 's/./& /'))) -DOV4=$(word 2,$(shell echo $* | sed 's/./& /')) $< -o $@
shaders/gemm12p_%.spv: shaders/gemm12.comp shaders/common.glsl
	glslc --target-env=vulkan1.1 -O -DPOST -DAV4=$(word 1,$(subst _, ,$(shell echo $* | sed 's/./& /'))) -DOV4=$(word 2,$(shell echo $* | sed 's/./& /')) $< -o $@
shaders/dwconvp.spv: shaders/dwconv.comp shaders/common.glsl
	glslc --target-env=vulkan1.1 -O -DPOST $< -o $@
shaders/dwconv1p.spv: shaders/dwconv1.comp shaders/common.glsl
	glslc --target-env=vulkan1.1 -O -DPOST $< -o $@
