// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "GS/Renderers/Vulkan/VKLoader.h"

#include "common/Assertions.h"
#include "common/Console.h"
#include "common/DynamicLibrary.h"
#include "common/Error.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" {

#define VULKAN_MODULE_ENTRY_POINT(name, required) PFN_##name name;
#define VULKAN_INSTANCE_ENTRY_POINT(name, required) PFN_##name name;
#define VULKAN_DEVICE_ENTRY_POINT(name, required) PFN_##name name;
#include "VKEntryPoints.inl"
#undef VULKAN_DEVICE_ENTRY_POINT
#undef VULKAN_INSTANCE_ENTRY_POINT
#undef VULKAN_MODULE_ENTRY_POINT
}

void Vulkan::ResetVulkanLibraryFunctionPointers()
{
#define VULKAN_MODULE_ENTRY_POINT(name, required) name = nullptr;
#define VULKAN_INSTANCE_ENTRY_POINT(name, required) name = nullptr;
#define VULKAN_DEVICE_ENTRY_POINT(name, required) name = nullptr;
#include "VKEntryPoints.inl"
#undef VULKAN_DEVICE_ENTRY_POINT
#undef VULKAN_INSTANCE_ENTRY_POINT
#undef VULKAN_MODULE_ENTRY_POINT
}

static DynamicLibrary s_vulkan_library;

bool Vulkan::IsVulkanLibraryLoaded()
{
	return s_vulkan_library.IsOpen();
}

bool Vulkan::LoadVulkanLibrary(Error* error)
{
	pxAssertRel(!s_vulkan_library.IsOpen(), "Vulkan module is not loaded.");

#ifdef __APPLE__
	// Check if a path to a specific Vulkan library has been specified.
	char* libvulkan_env = getenv("LIBVULKAN_PATH");
	if (libvulkan_env)
		s_vulkan_library.Open(libvulkan_env, error);

	if (!s_vulkan_library.IsOpen())
	{
		// A libretro core has no bundle of its own: it gets whatever the frontend
		// brought along. RetroArch ships MoltenVK inside its own bundle - as a plain
		// dylib in some builds, as a framework in others - and a Vulkan SDK install
		// leaves the loader on the library path instead, so try all of them rather
		// than only the name a bundled PCSX2 would use. DynamicLibrary::Open() looks
		// beside this module and in the running bundle's Frameworks directory for
		// each name, which is why the framework candidate is written as a relative
		// path.
		const std::string candidates[] = {
			DynamicLibrary::GetVersionedFilename("MoltenVK"),
			"MoltenVK.framework/MoltenVK",
			DynamicLibrary::GetVersionedFilename("vulkan", 1),
			DynamicLibrary::GetVersionedFilename("vulkan"),
		};

		for (const std::string& candidate : candidates)
		{
			if (s_vulkan_library.Open(candidate.c_str(), nullptr))
			{
				INFO_LOG("Vulkan: Loaded {}", candidate);
				break;
			}
		}

		if (!s_vulkan_library.IsOpen())
		{
			std::string tried;
			for (const std::string& candidate : candidates)
			{
				if (!tried.empty())
					tried += ", ";
				tried += candidate;
			}

			Error::SetStringFmt(error,
				"Failed to load a Vulkan library. Tried {} - beside the core, and in the "
				"Frameworks directory of the running application. Set LIBVULKAN_PATH to "
				"point at one.",
				tried);
			return false;
		}
	}
#else
	// try versioned first, then unversioned.
	if (!s_vulkan_library.Open(DynamicLibrary::GetVersionedFilename("vulkan", 1).c_str(), error) &&
		!s_vulkan_library.Open(DynamicLibrary::GetVersionedFilename("vulkan").c_str(), error))
	{
		return false;
	}
#endif

	bool required_functions_missing = false;
#define VULKAN_MODULE_ENTRY_POINT(name, required) \
	if (!s_vulkan_library.GetSymbol(#name, &name)) \
	{ \
		ERROR_LOG("Vulkan: Failed to load required module function {}", #name); \
		required_functions_missing = true; \
	}

#include "VKEntryPoints.inl"
#undef VULKAN_MODULE_ENTRY_POINT

	if (required_functions_missing)
	{
		ResetVulkanLibraryFunctionPointers();
		s_vulkan_library.Close();
		return false;
	}

	return true;
}

void Vulkan::UnloadVulkanLibrary()
{
	ResetVulkanLibraryFunctionPointers();
	s_vulkan_library.Close();
}

bool Vulkan::LoadVulkanInstanceFunctions(VkInstance instance)
{
	bool required_functions_missing = false;
	auto LoadFunction = [&required_functions_missing, instance](PFN_vkVoidFunction* func_ptr, const char* name, bool is_required) {
		*func_ptr = vkGetInstanceProcAddr(instance, name);
		if (!(*func_ptr) && is_required)
		{
			std::fprintf(stderr, "Vulkan: Failed to load required instance function %s\n", name);
			required_functions_missing = true;
		}
	};

#define VULKAN_INSTANCE_ENTRY_POINT(name, required) \
	LoadFunction(reinterpret_cast<PFN_vkVoidFunction*>(&name), #name, required);
#include "VKEntryPoints.inl"
#undef VULKAN_INSTANCE_ENTRY_POINT

	return !required_functions_missing;
}

bool Vulkan::LoadVulkanDeviceFunctions(VkDevice device)
{
	bool required_functions_missing = false;
	auto LoadFunction = [&required_functions_missing, device](PFN_vkVoidFunction* func_ptr, const char* name, bool is_required) {
		*func_ptr = vkGetDeviceProcAddr(device, name);
		if (!(*func_ptr) && is_required)
		{
			std::fprintf(stderr, "Vulkan: Failed to load required device function %s\n", name);
			required_functions_missing = true;
		}
	};

#define VULKAN_DEVICE_ENTRY_POINT(name, required) \
	LoadFunction(reinterpret_cast<PFN_vkVoidFunction*>(&name), #name, required);
#include "VKEntryPoints.inl"
#undef VULKAN_DEVICE_ENTRY_POINT

	return !required_functions_missing;
}
