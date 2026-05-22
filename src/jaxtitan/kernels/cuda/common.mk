JAXTITAN_CUDA_MAKE_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
THUNDERKITTENS_ROOT ?= $(abspath $(JAXTITAN_CUDA_MAKE_DIR)/../../../../third_party/ThunderKittens)

include $(THUNDERKITTENS_ROOT)/kernels/common.mk
