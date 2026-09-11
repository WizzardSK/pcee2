// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// PCSX2 libretro core frontend.
//
// Threading model:
//  - retro_load_game() spawns a dedicated CPU thread which runs the usual
//    VMManager::Execute() loop (same shape as pcsx2-gsrunner's CPUThreadMain).
//  - Host::PumpMessagesOnCPUThread() is invoked by the core once per emulated
//    frame (at CPU vsync). We use it as the pacing point: the CPU thread grabs
//    the presented frame into a buffer, signals retro_run(), then blocks until
//    the frontend asks for the next frame.
//  - retro_run() hands one "run token" to the CPU thread, waits (with timeout,
//    so slow boots don't freeze the frontend) for the frame, and uploads it.
//
// v1 scope: software renderer + MTGS::SaveMemorySnapshot readback,
// null audio output, no pad input, no savestates. Enough to boot and render.

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "common/Assertions.h"
#include "common/Console.h"
#include "common/CrashHandler.h"
#include "common/Error.h"
#include "common/HostSys.h"
#include "common/FileSystem.h"
#include "common/MemorySettingsInterface.h"
#include "common/Path.h"
#include "common/ProgressCallback.h"
#include "common/SettingsWrapper.h"
#include "common/StringUtil.h"
#include "common/WindowInfo.h"

#include "EmbeddedResources.h"
#include "pcsx2/Achievements.h"
#include "pcsx2/CDVD/CDVDcommon.h"
#include "pcsx2/Config.h"
#include "pcsx2/GS.h"
#include "pcsx2/GS/GS.h"
#include "pcsx2/GS/GSExtra.h"
#include "pcsx2/GS/Renderers/Common/GSRenderer.h"
#include "pcsx2/Host/AudioStream.h"
#include "pcsx2/SIO/Pad/PadDualshock2.h"
#include "pcsx2/SPU2/spu2.h"
#include "pcsx2/Host.h"
#include "pcsx2/INISettingsInterface.h"
#include "pcsx2/ImGui/FullscreenUI.h"
#include "pcsx2/ImGui/ImGuiFullscreen.h"
#include "pcsx2/ImGui/ImGuiManager.h"
#include "pcsx2/Input/InputManager.h"
#include "pcsx2/GS/Renderers/Common/GSLibretro.h"
#ifdef ENABLE_VULKAN
#include "pcsx2/GS/Renderers/Vulkan/GSDeviceVK.h"
#include "pcsx2/GS/Renderers/Vulkan/VKLibretro.h"
#include "pcsx2/GS/Renderers/Vulkan/VKLoader.h"
#endif
#ifdef ENABLE_OPENGL
#include "pcsx2/GS/Renderers/OpenGL/GLLibretro.h"
#include "pcsx2/GS/Renderers/OpenGL/GSDeviceOGL.h"
#include "glad/gl.h"
#endif
#include "pcsx2/MTGS.h"
#include "pcsx2/MTVU.h"
#include "pcsx2/MemoryTypes.h"
#include "pcsx2/SIO/Pad/Pad.h"
#include "pcsx2/SaveState.h"
#include "pcsx2/USB/USB.h"
#include "pcsx2/VMManager.h"
#include "pcsx2/ps2/BiosTools.h"

#include "svnrev.h"

#include "libretro.h"
#ifdef ENABLE_VULKAN
#include "libretro_vulkan.h"
#endif
#include "LibretroVFS.h"

#include "fmt/format.h"

namespace LibretroHost
{
	// libretro callbacks
	static retro_environment_t s_environ_cb;
	static retro_video_refresh_t s_video_cb;
	static retro_audio_sample_batch_t s_audio_batch_cb;
	static retro_input_poll_t s_input_poll_cb;
	static retro_input_state_t s_input_state_cb;
	static retro_log_printf_t s_log_cb;

	// configuration
	static MemorySettingsInterface s_settings_interface;
	static std::string s_system_dir;

	// CPU thread management. The thread persists across game sessions (like
	// the Qt frontend's EmuThread) because VMManager::Internal::
	// CPUThreadInitialize may only run once per process (page fault handler,
	// COM init, ...). Games are booted in a request loop.
	static std::thread s_cpu_thread;
	static std::atomic_bool s_running{false};
	static VMBootParameters s_boot_params;
	static std::mutex s_session_mutex;
	static std::condition_variable s_session_cv;
	static bool s_boot_requested = false;
	static bool s_exit_requested = false;
	static bool s_session_active = false;

	// Disc list behind the libretro disk control interface. Holds one entry for
	// ordinary content and one per line for an m3u playlist; path is what the
	// frontend is shown, boot_path what CDVD opens (a cue resolves to the file
	// its data track lives in).
	struct DiscImage
	{
		std::string path;
		std::string boot_path;
	};
	static std::vector<DiscImage> s_discs;
	static unsigned s_disc_index = 0;
	static bool s_disc_ejected = false;
	// Remembered across loads by the frontend, so a multi-disc game resumes on
	// the disc it was left on.
	static unsigned s_disc_initial_index = 0;
	static std::string s_disc_initial_path;

	// frame pacing: retro_run() posts a run token, CPU thread posts frame-done
	static std::mutex s_frame_mutex;
	static std::condition_variable s_frame_cv;
	static bool s_run_token = false;
	static bool s_frame_ready = false;

	// Frames the GS dropped as duplicates (see GSSetDuplicateFrameCallback),
	// counted on the GS thread and spent in retro_run(): one dropped frame buys
	// one extra run token, so a frame the frontend would only have been told to
	// repeat becomes emulation time instead. Free-running counter and the
	// number of them retro_run() has spent - never reset independently.
	static std::atomic<u64> s_duplicate_frames{0};
	static u64 s_duplicate_frames_spent = 0;

	// Whether the VM paces itself (see the frame limiter core option).
	static std::atomic<bool> s_internal_limiter{false};

	// frame buffer handed from CPU thread to retro_run (guarded by s_frame_mutex)
	static std::vector<u32> s_frame_pixels;
	static u32 s_frame_width = 0;
	static u32 s_frame_height = 0;
	// ... and which readback retro_run() last presented, so it can tell a
	// waiting frame from an already-shown one (the HW paths have their own
	// serials, see VKLibretro/GLLibretro::HasFrame).
	static u64 s_frame_serial = 0;
	static u64 s_frame_serial_seen = 0;

	// Zero-copy HW-render present paths: instead of the GPU->CPU readback above,
	// the GS renders into something the frontend can read directly.
	//  - Vulkan (ported from yaps2): the GS shares the frontend's VkDevice
	//    through context negotiation and hands the rendered VkImage straight
	//    over via set_image.
	//  - OpenGL: the GS thread renders in a context sharing the frontend's
	//    objects, and retro_run blits the finished texture into the frontend's
	//    framebuffer (see GLLibretro).
	// Selected at retro_load_game from the renderer option; the readback path
	// stays as the fallback (software renderer, frontend refuses HW render, the
	// frontend's GL context can't be shared, or the PCEE2_READBACK=1 env
	// override for A/B testing).
	enum class HWRender
	{
		None,
		Vulkan,
		OpenGL,
	};
	static HWRender s_hw_render = HWRender::None;
	static bool HWRenderActive() { return s_hw_render != HWRender::None; }
	// Set by the frontend's context_reset once the retro_hw_render_interface is
	// available; the CPU thread parks on it before booting the VM so GSDeviceVK
	// adopts the shared instance during negotiation rather than creating its own.
	static std::atomic<bool> s_context_ready{false};
	// Raised once CPUThreadInitialize() has run, so the frontend's negotiation
	// callback (which opens MTGS) doesn't race the CPU-thread global setup.
	static std::atomic<bool> s_cpu_thread_initialized{false};
#ifdef ENABLE_OPENGL
	// The GL render callback outlives retro_load_game: retro_run calls
	// get_current_framebuffer() through it on every presented frame.
	static struct retro_hw_render_callback s_gl_hw_render = {};
	// Read framebuffer used to blit the GS's texture into the frontend's, on
	// the frontend's thread. FBOs are not shared between contexts, so this one
	// belongs to the frontend's context and dies with it.
	static GLuint s_gl_present_fbo = 0;
	// Set once a frontend GL context has been captured, so a second
	// context_reset can be told apart from the first: the second means the
	// frontend threw its context away and everything the GS built is orphaned.
	static bool s_gl_context_seen = false;
	// Raised by context_reset when that has happened, and acted on by
	// retro_run. See OnGLContextReset for why it cannot be acted on in place.
	static std::atomic<bool> s_gl_needs_reopen{false};
#endif

	// Last geometry announced to the frontend on a HW-render path. The aspect is
	// part of it: a widescreen patch or a 480p switch can change the shape of
	// the picture without moving the native resolution at all.
	static u32 s_hw_geom_width = 0;
	static u32 s_hw_geom_height = 0;
	static float s_hw_geom_aspect = 0.0f;

	// Size the last real HW frame was handed over at. A duplicate frame has to
	// repeat it: the frontend reads the size from the video callback on every
	// frame, and a number that disagrees with the frame before it is a resize
	// as far as it is concerned.
	static u32 s_hw_last_width = 0;
	static u32 s_hw_last_height = 0;

	// HW-render counterpart of "s_frame_width != 0": the readback callback that
	// used to set s_frame_width isn't wired in HW mode, so UpdateInput's
	// VM-is-up gate needs this instead. Only touched on the retro_run thread.
	static bool s_hw_frame_seen = false;

	// deferred work queue for Host::RunOnCPUThread
	static std::mutex s_cpu_work_mutex;
	static std::deque<std::function<void()>> s_cpu_work;
	static std::atomic_bool s_cpu_work_pending{false};
	static std::thread::id s_cpu_thread_id;

	static constexpr u32 DEFAULT_WIDTH = 640;
	static constexpr u32 DEFAULT_HEIGHT = 480;
	static constexpr u32 MAX_UPSCALE = 4;
	static constexpr u32 MAX_WIDTH = DEFAULT_WIDTH * MAX_UPSCALE;
	static constexpr u32 MAX_HEIGHT = DEFAULT_HEIGHT * MAX_UPSCALE;

	// current output (readback) resolution; follows the upscale option
	static std::atomic<u32> s_out_width{DEFAULT_WIDTH};
	static std::atomic<u32> s_out_height{DEFAULT_HEIGHT};

	// core option state
	static std::vector<std::string> s_bios_names; // backing storage for option values
	static std::vector<std::string> s_memcard_names; // backing storage for option values
	static u32 s_opt_upscale = 1;

	// libretro port -> PCSX2 pad index (see sioConvertPadToPortAndSlot: 0=1A,
	// 1=2A, 2..4=1B..1D, 5..7=2B..2D), built from the multitap option
	static std::vector<u32> s_pad_map = {0, 1};

	// GunCon2 lightguns on USB ports (bit 0 = USB1, bit 1 = USB2)
	static u32 s_lightgun_mask = 0;

	// rumble: written by the InputManager callback (CPU thread), consumed in
	// retro_run; packed as (large << 16) | small, each 0..65535
	static std::array<std::atomic<u32>, 8> s_pad_rumble;
	static bool s_rumble_enabled = true;
	static retro_rumble_interface s_rumble_interface = {};

	static void PadVibrationCallback(u32 pad_index, float large_motor, float small_motor)
	{
		if (pad_index >= s_pad_rumble.size())
			return;
		const u32 large = static_cast<u32>(std::clamp(large_motor, 0.0f, 1.0f) * 65535.0f);
		const u32 small = static_cast<u32>(std::clamp(small_motor, 0.0f, 1.0f) * 65535.0f);
		s_pad_rumble[pad_index].store((large << 16) | small, std::memory_order_relaxed);
	}
	static constexpr u32 SAMPLE_RATE = 48000;
	static constexpr u32 MAX_AUDIO_FRAMES_PER_RUN = 2048;

	// What the frontend last said about its audio buffer. A write into a full
	// one blocks until something drains it, and a frontend whose audio device
	// is not running - a menu is open, the VM is paused - drains nothing, so
	// that write does not return and retro_run never does either. The CPU
	// thread is then parked waiting for a frame token the main thread cannot
	// give it, which is the deadlock in issue #36.
	//
	// Registering this is what lets the core see that coming: at a high
	// occupancy it hands over nothing that frame rather than risk the block.
	// The frontend repeats the last buffer, which is audible as a moment of
	// stutter and is a great deal better than a hang.
	static std::atomic_bool s_audio_status_known{false};
	static std::atomic<unsigned> s_audio_occupancy{0};

	// VM timing reported to the frontend (PAL games run at 50Hz, PSX mode at
	// 44.1kHz); updated from the CPU/audio-factory threads, consumed in retro_run
	static std::atomic<u32> s_vm_fps_bits{0};
	static std::atomic<u32> s_audio_sample_rate{SAMPLE_RATE};

	// set once the frontend has been given the EE memory map for this session
	static bool s_memory_map_sent = false;

	// display aspect ratio reported to the frontend (bit_cast'd float); follows
	// the aspect option / widescreen patches
	static std::atomic<u32> s_aspect_bits{0};

	// SPU2 output stream that the frontend pulls samples from in retro_run().
	// Reads happen while the CPU thread is parked in PumpMessagesOnCPUThread(),
	// and the ring buffer itself is SPSC-atomic, so no locking is needed for
	// steady-state reads/writes against a live stream object.
	//
	// That assumption breaks around VMManager::Reset(): SPU2::CreateOutputStream()
	// (called from hwReset() by way of SPU2::Reset() -> UpdateSampleRate(), or
	// via ApplySettings()) does `s_output_stream.reset()` then constructs a
	// brand new stream, and s_audio_stream is just an observer of whichever
	// object SPU2 currently owns - it isn't updated until the new stream's
	// constructor runs. Between those two points the old AudioStream is fully
	// destroyed while s_audio_stream still points at it. retro_run() keeps
	// getting called on its own schedule the whole time (nothing about
	// retro_reset() pauses it), so OutputAudio() can - and did - dereference
	// a freed stream via s_audio_stream mid-teardown. s_audio_stream_mutex
	// closes that window: OutputAudio() holds it only for the quick
	// PullFrames() call, and retro_reset() holds it for the full
	// VMManager::Reset(), so the two can never overlap regardless of which
	// thread ends up calling what.
	static std::mutex s_audio_stream_mutex;

	class LibretroAudioStream final : public AudioStream
	{
	public:
		LibretroAudioStream(u32 sample_rate, const AudioStreamParameters& parameters)
			: AudioStream(sample_rate, parameters)
		{
		}

		void Initialize()
		{
			BaseInitialize(&StereoSampleReaderImpl, false);
		}

		u32 PullFrames(SampleType* dest, u32 max_frames)
		{
			const u32 frames = std::min(GetBufferedFramesRelaxed(), max_frames);
			if (frames > 0)
				ReadFrames(dest, frames);
			return frames;
		}
	};

	// observer only; the stream is owned by SPU2 (s_output_stream)
	static LibretroAudioStream* s_audio_stream = nullptr;
	static std::atomic<u64> s_audio_frames_output{0};

	// Runs on the GS thread once per vsync with the previous frame's pixels
	// (see GSSetFramebufferReadback). Swizzles RGBA -> XRGB8888 into the
	// buffer retro_run() presents.
	static void FramebufferReadbackCallback(const u32* pixels, u32 pitch_px, u32 width, u32 height)
	{
		std::unique_lock lock(s_frame_mutex);
		s_frame_pixels.resize(static_cast<size_t>(width) * height);
		for (u32 y = 0; y < height; y++)
		{
			const u32* src = pixels + static_cast<size_t>(y) * pitch_px;
			u32* dst = s_frame_pixels.data() + static_cast<size_t>(y) * width;
			for (u32 x = 0; x < width; x++)
			{
				const u32 px = src[x];
				dst[x] = (px & 0xFF00FF00u) | ((px & 0xFFu) << 16) | ((px >> 16) & 0xFFu);
			}
		}
		s_frame_width = width;
		s_frame_height = height;
		s_frame_serial++;
	}

	// Runs on the GS thread whenever SkipDuplicateFrames drops a vsync, i.e.
	// whenever there will be no frame for retro_run() to present.
	static void DuplicateFrameCallback()
	{
		s_duplicate_frames.fetch_add(1, std::memory_order_release);
	}

	static std::unique_ptr<AudioStream> CreateLibretroAudioStream(u32 sample_rate, const AudioStreamParameters& parameters)
	{
		std::unique_ptr<LibretroAudioStream> stream = std::make_unique<LibretroAudioStream>(sample_rate, parameters);
		stream->Initialize();
		s_audio_stream = stream.get();
		s_audio_sample_rate.store(sample_rate, std::memory_order_release);
		return stream;
	}

	static bool InitializeConfig();
	static void SettingsOverride();
	static void CPUThreadMain();
	static void DrainCPUWork();
	static void RegisterCoreOptions();
	static void ReadCoreOptions(bool startup);

	// Forward PCSX2's log to the frontend's log interface. Keeps Windows from
	// popping up a console window, and the messages land in RetroArch's log.
	static void HostLogCallback(LOGLEVEL level, ConsoleColors color, std::string_view message)
	{
		if (!s_log_cb)
		{
			std::fprintf(stderr, "%.*s\n", static_cast<int>(message.size()), message.data());
			return;
		}

		retro_log_level rl = RETRO_LOG_INFO;
		if (level == LOGLEVEL_ERROR)
			rl = RETRO_LOG_ERROR;
		else if (level == LOGLEVEL_WARNING)
			rl = RETRO_LOG_WARN;
		else if (level >= LOGLEVEL_DEV)
			rl = RETRO_LOG_DEBUG;
		s_log_cb(rl, "[PCSX2] %.*s\n", static_cast<int>(message.size()), message.data());
	}
} // namespace LibretroHost

using namespace LibretroHost;

// Tears the core down when the process exits or the library is unloaded:
// stops the persistent CPU thread (running VMManager's CPU-thread shutdown on
// it), and removes the process-wide page fault handler. Registered via
// atexit() from retro_init, which makes it run BEFORE any of this library's
// static destructors (atexit/destructor handlers execute in reverse
// registration order, and retro_init runs after all static initializers) —
// MTGS's global thread object asserts if it's still alive at destruction.
static void ShutdownCoreAtExit()
{
	if (s_cpu_thread.joinable())
	{
		{
			std::unique_lock lock(s_session_mutex);
			s_exit_requested = true;
		}
		s_session_cv.notify_all();
		s_cpu_thread.join();
	}

	PageFaultHandler::Uninstall();
}

//////////////////////////////////////////////////////////////////////////
// Config / boot
//////////////////////////////////////////////////////////////////////////

bool LibretroHost::InitializeConfig()
{
	// Map PCSX2's folder layout into <retro_system_directory>/pcsx2/.
	// BIOS goes to <system>/pcsx2/bios, resources to <system>/pcsx2/resources.
	const char* system_dir = nullptr;
	if (s_environ_cb && s_environ_cb(RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY, &system_dir) && system_dir)
		s_system_dir = Path::Combine(system_dir, "pcsx2");
	else
		s_system_dir = "pcsx2";

	EmuFolders::AppRoot = s_system_dir;
	EmuFolders::DataRoot = s_system_dir;
	EmuFolders::Resources = Path::Combine(s_system_dir, "resources");
	EmuFolders::UserResources = EmuFolders::Resources;
	EmuFolders::Settings = Path::Combine(s_system_dir, "inis");

	// crash dumps belong in our data directory, not the frontend's cwd
	CrashHandler::SetWriteDirectory(EmuFolders::DataRoot);

	if (!FileSystem::DirectoryExists(EmuFolders::Resources.c_str()))
	{
		// Not fatal any more: the shaders and fonts the core cannot start without
		// are built into it. What is left in that directory - the game database,
		// the patches - only costs the feature that wants it.
		Console.WarningFmt("No resources directory at '{}'. The core will run, but the game database and "
						   "patches will be unavailable; copy the 'resources' directory from a matching "
						   "PCSX2 build there to get them.",
			EmuFolders::Resources);
	}

	const char* error;
	if (!VMManager::PerformEarlyHardwareChecks(&error))
	{
		Console.ErrorFmt("Hardware check failed: {}", error);
		return false;
	}

	{
		// ImGui keeps the span, so whatever backs it has to outlive this scope:
		// the mapping stays for the process, and the embedded copy is static data.
		std::span<const u8> roboto_data;

#ifdef PCSX2_EMBEDDED_RESOURCES
		if (EmbeddedResourcesPreferred())
		{
			if (const std::optional<std::string_view> embedded = GetEmbeddedResource("fonts/Roboto-Regular.ttf"))
				roboto_data = std::span<const u8>(reinterpret_cast<const u8*>(embedded->data()), embedded->size());
		}
#endif

		if (roboto_data.empty())
		{
			const std::string roboto_path =
				EmuFolders::GetOverridableResourcePath("fonts" FS_OSPATH_SEPARATOR_STR "Roboto-Regular.ttf");
			roboto_data = FileSystem::MapBinaryFileForRead(roboto_path.c_str());
			if (roboto_data.empty())
			{
				Console.ErrorFmt("Failed to load font file '{}'.", roboto_path);
				return false;
			}
		}

		std::vector<ImGuiManager::FontInfo> fonts;
		ImGuiManager::FontInfo fi{};
		fi.data = roboto_data;
		fi.exclude_ranges = {};
		fi.face_name = nullptr;
		fi.is_emoji_font = false;
		fonts.push_back(fi);

		ImGuiManager::SetFonts(std::move(fonts));
	}

	MemorySettingsInterface& si = s_settings_interface;

	// content can be loaded repeatedly in one core session (RetroArch's
	// "Close Content" keeps the core resident); the base layer may only be
	// registered once
	static bool s_settings_layer_registered = false;
	if (!s_settings_layer_registered)
	{
		Host::Internal::SetBaseSettingsLayer(&si);
		s_settings_layer_registered = true;
	}

	VMManager::SetDefaultSettings(si, true, true, true, true, true);

	// If the user dropped a standalone PCSX2.ini into <system>/pcsx2/inis,
	// adopt its emulation settings as the baseline. Core options and the
	// libretro-specific overrides still apply on top, and host-side sections
	// (folders, UI, input bindings, audio device, logging, ...) are ignored.
	{
		const std::string ini_path = Path::Combine(EmuFolders::Settings, "PCSX2.ini");
		INISettingsInterface ini(ini_path);
		if (FileSystem::FileExists(ini_path.c_str()) && ini.Load())
		{
			static constexpr const char* merge_sections[] = {
				"EmuCore",
				"EmuCore/Speedhacks",
				"EmuCore/CPU",
				"EmuCore/CPU/Recompiler",
				"EmuCore/GS",
				"EmuCore/Gamefixes",
				"MemoryCards",
				"DEV9/Eth",
				"DEV9/Hdd",
			};

			u32 merged = 0;
			for (const char* section : merge_sections)
			{
				for (const auto& [key, value] : ini.GetKeyValueList(section))
				{
					s_settings_interface.SetStringValue(section, key.c_str(), value.c_str());
					merged++;
				}
			}
			Console.WriteLnFmt("Adopted {} settings from standalone config '{}'.", merged, ini_path);
		}
	}

	VMManager::Internal::LoadStartupSettings();

	EmuFolders::EnsureFoldersExist();
	return true;
}

void LibretroHost::SettingsOverride()
{
	// Host vsync is never ours to wait on - the frontend owns the window. The
	// frame limiter is a choice: normally the frontend paces us by blocking on
	// the audio we hand it, and a second limiter underneath that would only
	// fight it. Where that blocking does not happen - an audio driver that
	// underruns instead of waiting, which is what an Android device with no
	// headroom does - nothing paces the VM at all and its own speed variation
	// reaches the screen as jitter. The option puts PCSX2's limiter back for
	// that case.
	s_settings_interface.SetBoolValue("EmuCore/GS", "FrameLimitEnable", s_internal_limiter.load(std::memory_order_acquire));
	s_settings_interface.SetIntValue("EmuCore/GS", "VsyncEnable", false);

	// Renderer comes from the core options (Vulkan or SW-on-Vulkan); Vulkan is
	// the only Linux backend that supports a Surfaceless (swapchain-less)
	// device, and we read frames back anyway.

	// All input comes through the libretro API, not host devices. The default
	// keyboard bindings are left in place — we never feed host key events, so
	// they are inert, and their presence suppresses the "controller not
	// configured" OSD warning.
	s_settings_interface.SetBoolValue("InputSources", "SDL", false);
	s_settings_interface.SetBoolValue("InputSources", "XInput", false);
	s_settings_interface.ClearSection("Hotkeys");

	// v1: no audio output yet
	s_settings_interface.SetStringValue("SPU2/Output", "OutputModule", "nullout");

	// no system console: it would open a real console window on Windows; logs
	// flow through the libretro log interface instead (see HostLogCallback)
	s_settings_interface.SetBoolValue("Logging", "EnableSystemConsole", false);

	// savestates go through retro_serialize as uncompressed zips; speed over
	// size, the frontend can compress its state files itself
	s_settings_interface.SetIntValue("EmuCore", "SavestateCompressionType",
		static_cast<int>(SavestateCompressionMethod::Uncompressed));

	// pick the first valid BIOS image from <system>/pcsx2/bios if none is configured
	if (s_settings_interface.GetStringValue("Filenames", "BIOS").empty())
	{
		FileSystem::FindResultsArray files;
		FileSystem::FindFiles(EmuFolders::Bios.c_str(), "*", FILESYSTEM_FIND_FILES, &files);
		for (const FILESYSTEM_FIND_DATA& fd : files)
		{
			u32 version, region;
			std::string description, zone;
			if (IsBIOS(fd.FileName.c_str(), version, description, region, zone))
			{
				const std::string filename(Path::GetFileName(fd.FileName));
				Console.WriteLnFmt("Auto-selected BIOS: {} ({})", filename, description);
				s_settings_interface.SetStringValue("Filenames", "BIOS", filename.c_str());
				break;
			}
		}
	}
}

void LibretroHost::RegisterCoreOptions()
{
	// scan for BIOS images so the option can list them
	s_bios_names.clear();
	s_bios_names.push_back("auto");
	{
		const char* system_dir = nullptr;
		if (s_environ_cb(RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY, &system_dir) && system_dir)
		{
			FileSystem::FindResultsArray files;
			FileSystem::FindFiles(Path::Combine(Path::Combine(system_dir, "pcsx2"), "bios").c_str(), "*",
				FILESYSTEM_FIND_FILES, &files);
			for (const FILESYSTEM_FIND_DATA& fd : files)
			{
				u32 version, region;
				std::string description, zone;
				if (IsBIOS(fd.FileName.c_str(), version, description, region, zone))
					s_bios_names.push_back(std::string(Path::GetFileName(fd.FileName)));
			}
		}
	}

	// Scan only real .ps2 files. This happens before InitializeConfig(), so the
	// existing FileMcd_GetAvailableCards() helper cannot be used here: it reads
	// EmuFolders::MemoryCards, which is populated during startup configuration.
	s_memcard_names.clear();
	{
		const char* system_dir = nullptr;
		if (s_environ_cb(RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY, &system_dir) && system_dir)
		{
			FileSystem::FindResultsArray files;
			const std::string memcards_dir = Path::Combine(Path::Combine(system_dir, "pcsx2"), "memcards");
			FileSystem::FindFiles(memcards_dir.c_str(), "*", FILESYSTEM_FIND_FILES, &files);
			for (const FILESYSTEM_FIND_DATA& fd : files)
			{
				const std::string filename(Path::GetFileName(fd.FileName));
				if (filename.ends_with(".ps2"))
					s_memcard_names.push_back(filename);
			}
		}
	}
	std::sort(s_memcard_names.begin(), s_memcard_names.end());
	s_memcard_names.erase(std::unique(s_memcard_names.begin(), s_memcard_names.end()), s_memcard_names.end());

	static retro_core_option_v2_category categories[] = {
		{"system", "System", "BIOS and boot behaviour."},
		{"memory_cards", "Memory Cards", "PS2 memory card slot settings."},
		{"graphics", "Graphics", "Renderer, resolution and image quality."},
		{"patches", "Patches", "Built-in game patches (widescreen, no-interlacing)."},
		{"performance", "Performance", "Speed hacks. May break games."},
		{"audio", "Audio", "Sound output and buffering."},
		{nullptr, nullptr, nullptr},
	};

	retro_core_option_v2_definition definitions[] = {
		// system
		{"pcsx2_bios", "BIOS", nullptr, "BIOS image to use, from <system>/pcsx2/bios. Requires restart.", nullptr,
			"system", {{nullptr, nullptr}}, "auto"},
		{"pcsx2_fast_boot", "Fast Boot", nullptr, "Skip the BIOS boot animation. Requires restart.", nullptr,
			"system", {{"enabled", nullptr}, {"disabled", nullptr}, {nullptr, nullptr}}, "enabled"},
		// graphics
		// Offering an API this build has no renderer for would leave GSCreateDevice
		// with nothing to construct (Android turns OpenGL off entirely), so the
		// list only carries what is actually compiled in.
		{"pcsx2_renderer", "Renderer", nullptr,
			"Hardware renderer API, or the software renderer. Switching to or from Software applies "
			"on the fly; switching between hardware APIs takes effect when the content is restarted.",
			nullptr, "graphics",
			{
#ifdef ENABLE_VULKAN
			{"vulkan", "Vulkan (Hardware)"},
#endif
#ifdef ENABLE_OPENGL
				{"opengl", "OpenGL (Hardware)"},
#endif
				{"software", "Software"}, {nullptr, nullptr}},
			"vulkan"},
		{"pcsx2_upscale_multiplier", "Internal Resolution", nullptr,
			"Internal rendering resolution multiplier for the hardware renderer. Also scales the output framebuffer. Applies on the fly.",
			nullptr, "graphics",
			{{"1", "1x Native (640x480)"}, {"2", "2x Native (1280x960)"}, {"3", "3x Native (1920x1440)"},
				{"4", "4x Native (2560x1920)"}, {nullptr, nullptr}},
			"1"},
		{"pcsx2_hw_download_mode", "Hardware Download Mode", nullptr,
			"How GPU->CPU readbacks are handled when a game reads rendered data back (GT3 heat haze, "
			"photo modes...). Accurate stalls the whole pipeline on tiler GPUs; Unsynchronized returns "
			"stale data without stalling (big speedup, may glitch those effects); Disabled skips them.",
			nullptr, "graphics",
			{{"accurate", "Accurate (Default)"}, {"unsynchronized", "Unsynchronized (Fast)"},
				{"disabled", "Disabled (Fastest)"}, {nullptr, nullptr}},
			"accurate"},
		{"pcsx2_blending_accuracy", "Blending Accuracy", nullptr,
			"Higher levels emulate more PS2 blending effects correctly at a GPU cost.", nullptr, "graphics",
			{{"minimum", "Minimum"}, {"basic", "Basic (Recommended)"}, {"medium", "Medium"}, {"high", "High"},
				{"full", "Full (Slow)"}, {"maximum", "Maximum (Very Slow)"}, {nullptr, nullptr}},
			"basic"},
		{"pcsx2_texture_filtering", "Texture Filtering", nullptr,
			"Bilinear (PS2) replicates the console; forced modes smooth all textures.", nullptr, "graphics",
			{{"nearest", "Nearest"}, {"bilinear_ps2", "Bilinear (PS2)"}, {"bilinear_forced", "Bilinear (Forced)"},
				{"bilinear_forced_sprite", "Bilinear (Forced excluding sprites)"}, {nullptr, nullptr}},
			"bilinear_ps2"},
		{"pcsx2_trilinear_filtering", "Trilinear Filtering", nullptr, nullptr, nullptr, "graphics",
			{{"auto", "Automatic (Default)"}, {"off", "Off"}, {"ps2", "Trilinear (PS2)"}, {"forced", "Trilinear (Forced)"},
				{nullptr, nullptr}},
			"auto"},
		{"pcsx2_anisotropic_filtering", "Anisotropic Filtering", nullptr,
			"Reduces texture aliasing at steep angles.", nullptr, "graphics",
			{{"0", "Off"}, {"2", "2x"}, {"4", "4x"}, {"8", "8x"}, {"16", "16x"}, {nullptr, nullptr}}, "0"},
		{"pcsx2_dithering", "Dithering", nullptr,
			"Unscaled (default) replicates PS2 dithering; Off can reduce banding artifacts at high resolutions.",
			nullptr, "graphics",
			{{"0", "Off"}, {"1", "Scaled"}, {"2", "Unscaled (Default)"}, {nullptr, nullptr}}, "2"},
		{"pcsx2_mipmapping", "Hardware Mipmapping", nullptr, nullptr, nullptr, "graphics",
			{{"enabled", nullptr}, {"disabled", nullptr}, {nullptr, nullptr}}, "enabled"},
		{"pcsx2_deinterlace_mode", "Deinterlacing", nullptr,
			"Automatic uses the GameDB-recommended mode per game.", nullptr, "graphics",
			{{"0", "Automatic (Default)"}, {"1", "Off"}, {"2", "Weave (TFF)"}, {"3", "Weave (BFF)"},
				{"4", "Bob (TFF)"}, {"5", "Bob (BFF)"}, {"6", "Blend (TFF)"}, {"7", "Blend (BFF)"},
				{"8", "Adaptive (TFF)"}, {"9", "Adaptive (BFF)"}, {nullptr, nullptr}},
			"0"},
		{"pcsx2_fxaa", "FXAA", nullptr, "Cheap post-process anti-aliasing.", nullptr, "graphics",
			{{"disabled", nullptr}, {"enabled", nullptr}, {nullptr, nullptr}}, "disabled"},
		{"pcsx2_cas_mode", "Contrast Adaptive Sharpening", nullptr, nullptr, nullptr, "graphics",
			{{"disabled", "Disabled"}, {"sharpen", "Sharpen Only"}, {nullptr, nullptr}}, "disabled"},
		{"pcsx2_audio_buffer_ms", "Audio Buffer", nullptr,
			"How much audio the emulator keeps ahead of the frontend. More of it rides out a "
			"stall that would otherwise be heard as a break in the sound, at the cost of that "
			"much delay before you hear anything. PCSX2's own default is 50 ms, which assumes it "
			"is feeding an audio device directly rather than a frontend that pulls once a frame.",
			nullptr, "audio",
			{{"50", "50 ms"}, {"75", "75 ms"}, {"100", "100 ms"}, {"150", "150 ms"}, {"200", "200 ms"},
				{nullptr, nullptr}}, "100"},
		{"pcsx2_frame_limiter", "Frame Limiter", nullptr,
			"What holds the emulator to full speed. Frontend leaves the pacing to RetroArch, which "
			"throttles by making the core wait on the audio it hands over. Internal uses PCSX2's own "
			"limiter instead - for a device where that wait never happens, because the audio driver "
			"underruns rather than blocking, and the emulator's own speed variation then reaches the "
			"screen as uneven frame pacing.",
			nullptr, "system",
			{{"frontend", "Frontend (RetroArch)"}, {"internal", "Internal (PCSX2)"}, {nullptr, nullptr}}, "frontend"},
		{"pcsx2_skip_duplicate_frames", "Skip Presenting Duplicate Frames", nullptr,
			"Don't hand the frontend a frame the GS never redrew - a 30fps game then delivers 30 "
			"unique frames instead of 60 with every second one repeated. Turn this off if a frame "
			"generation or interpolation filter needs every frame delivered as its own.",
			nullptr, "graphics",
			{{"enabled", nullptr}, {"disabled", nullptr}, {nullptr, nullptr}}, "enabled"},
		{"pcsx2_cas_sharpness", "CAS Sharpness", nullptr, nullptr, nullptr, "graphics",
			{{"10", nullptr}, {"20", nullptr}, {"30", nullptr}, {"40", nullptr}, {"50", nullptr}, {"60", nullptr},
				{"70", nullptr}, {"80", nullptr}, {"90", nullptr}, {"100", nullptr}, {nullptr, nullptr}},
			"50"},
		{"pcsx2_aspect_ratio", "Aspect Ratio", nullptr,
			"Automatic reports 16:9 when widescreen patches are enabled, 4:3 otherwise.", nullptr, "graphics",
			{{"auto", "Automatic"}, {"4:3", nullptr}, {"16:9", nullptr}, {nullptr, nullptr}}, "auto"},
		// system (continued)
		{"pcsx2_multitap", "Multitap", nullptr,
			"Enable the multitap adapter for up to 8 controllers. Player order follows the physical slots "
			"(port 1: 1A-1D, then port 2: 2A-2D). Restart recommended.",
			nullptr, "system",
			{{"disabled", "Disabled (2 players)"}, {"port1", "Port 1 (5 players)"}, {"port2", "Port 2 (5 players)"},
				{"both", "Both Ports (8 players)"}, {nullptr, nullptr}},
			"disabled"},
		{"pcsx2_lightgun", "Lightgun (GunCon 2)", nullptr,
			"Emulate a Namco GunCon 2 on a USB port, aimed with the frontend's lightgun (or mouse mapped as "
			"lightgun) on the matching controller port. Requires restart.",
			nullptr, "system",
			{{"disabled", "Disabled"}, {"usb1", "USB Port 1"}, {"usb2", "USB Port 2"}, {"both", "Both Ports"},
				{nullptr, nullptr}},
			"disabled"},
		{"pcsx2_rumble", "Rumble", nullptr, "Forward DualShock 2 vibration to the frontend's rumble support.",
			nullptr, "system", {{"enabled", nullptr}, {"disabled", nullptr}, {nullptr, nullptr}}, "enabled"},
		{"pcsx2_axis_scale", "Analog Axis Scale", nullptr,
			"Scales stick input like a real DualShock 2 (PCSX2 default 133%). Lower if diagonals feel clamped.",
			nullptr, "system",
			{{"100", "100%"}, {"115", "115%"}, {"133", "133% (Default)"}, {"150", "150%"}, {nullptr, nullptr}},
			"133"},
		{"pcsx2_axis_deadzone", "Analog Deadzone", nullptr,
			"Stick deadzone applied inside the emulated controller, on top of any frontend deadzone.", nullptr,
			"system",
			{{"0", "0% (Default)"}, {"5", "5%"}, {"10", "10%"}, {"15", "15%"}, {"20", "20%"}, {"30", "30%"},
				{nullptr, nullptr}},
			"0"},
		// patches
		{"pcsx2_widescreen_patches", "Widescreen Patches", nullptr,
			"Enable built-in 16:9 widescreen patches where available. Best applied before starting a game.", nullptr,
			"patches", {{"disabled", nullptr}, {"enabled", nullptr}, {nullptr, nullptr}}, "disabled"},
		{"pcsx2_no_interlacing_patches", "No-Interlacing Patches", nullptr,
			"Enable built-in progressive-output patches where available. Best applied before starting a game.", nullptr,
			"patches", {{"disabled", nullptr}, {"enabled", nullptr}, {nullptr, nullptr}}, "disabled"},
		// performance
		{"pcsx2_mtvu", "MTVU (Multi-Threaded VU1)", nullptr,
			"Runs VU1 on its own thread. Large speedup on multi-core CPUs; a small number of games hang with it.",
			nullptr, "performance",
			{{"enabled", nullptr}, {"disabled", nullptr}, {nullptr, nullptr}}, "enabled"},
		{"pcsx2_instant_vu1", "Instant VU1", nullptr,
			"Runs VU1 to completion immediately (ignored while MTVU is enabled). Usually a speedup.",
			nullptr, "performance",
			{{"enabled", nullptr}, {"disabled", nullptr}, {nullptr, nullptr}}, "enabled"},
		{"pcsx2_ee_cycle_rate", "EE Cycle Rate", nullptr,
			"Underclock or overclock the emulated Emotion Engine. Default 100%. May break games.", nullptr,
			"performance",
			{{"-3", "50% (Underclock)"}, {"-2", "60% (Underclock)"}, {"-1", "75% (Underclock)"},
				{"0", "100% (Default)"}, {"1", "130% (Overclock)"}, {"2", "180% (Overclock)"},
				{"3", "300% (Overclock)"}, {nullptr, nullptr}},
			"0"},
		{"pcsx2_ee_cycle_skip", "EE Cycle Skip", nullptr,
			"Makes the EE skip cycles. Helps some games with high VU activity, breaks others.", nullptr,
			"performance",
			{{"0", "Disabled (Default)"}, {"1", "Mild"}, {"2", "Moderate"}, {"3", "Maximum"}, {nullptr, nullptr}},
			"0"},
		{"pcsx2_cpu_recompiler", "CPU Recompiler (JIT)", nullptr,
			"Diagnostic master switch. Enabled runs the EE, IOP and VU0/VU1 dynarecs (JIT, fast, default). "
			"Disabled forces every CPU to an interpreter, which is far slower but isolates JIT bugs: if a crash "
			"still happens with this off, the recompiler is not the cause. The four per-CPU switches below only "
			"take effect while this is Enabled. Requires restart.",
			nullptr, "performance",
			{{"enabled", "Enabled (JIT, Default)"}, {"disabled", "Disabled (Interpreter)"}, {nullptr, nullptr}},
			"enabled"},
		{"pcsx2_rec_ee", "  - EE Recompiler", nullptr,
			"Diagnostic. Disable just the Emotion Engine (EE) dynarec while leaving the others on, to bisect "
			"which recompiler causes a crash. Requires restart.",
			nullptr, "performance",
			{{"enabled", "Enabled (Default)"}, {"disabled", "Disabled (Interpreter)"}, {nullptr, nullptr}},
			"enabled"},
		{"pcsx2_rec_iop", "  - IOP Recompiler", nullptr,
			"Diagnostic. Disable just the IOP (R3000) dynarec while leaving the others on, to bisect which "
			"recompiler causes a crash. Requires restart.",
			nullptr, "performance",
			{{"enabled", "Enabled (Default)"}, {"disabled", "Disabled (Interpreter)"}, {nullptr, nullptr}},
			"enabled"},
		{"pcsx2_rec_vu0", "  - VU0 Recompiler", nullptr,
			"Diagnostic. Disable just the VU0 microVU dynarec while leaving the others on, to bisect which "
			"recompiler causes a crash. Requires restart.",
			nullptr, "performance",
			{{"enabled", "Enabled (Default)"}, {"disabled", "Disabled (Interpreter)"}, {nullptr, nullptr}},
			"enabled"},
		{"pcsx2_rec_vu1", "  - VU1 Recompiler", nullptr,
			"Diagnostic. Disable just the VU1 microVU dynarec while leaving the others on, to bisect which "
			"recompiler causes a crash. Requires restart.",
			nullptr, "performance",
			{{"enabled", "Enabled (Default)"}, {"disabled", "Disabled (Interpreter)"}, {nullptr, nullptr}},
			"enabled"},
		// memory cards
		{"pcsx2_memcard_slot1_enable", "Slot 1 Enabled", nullptr,
			"Enable the Slot 1 PS2 memory card. Changes apply immediately while content is running.",
			nullptr, "memory_cards", {{"enabled", nullptr}, {"disabled", nullptr}, {nullptr, nullptr}}, "enabled"},
		{"pcsx2_memcard_slot2_enable", "Slot 2 Enabled", nullptr,
			"Enable the Slot 2 PS2 memory card. Changes apply immediately while content is running.",
			nullptr, "memory_cards", {{"enabled", nullptr}, {"disabled", nullptr}, {nullptr, nullptr}}, "enabled"},
		{"pcsx2_memcard_slot1_file", "Slot 1 Card", nullptr,
			"Select an existing .ps2 card from <system>/pcsx2/memcards. Changes apply immediately while content is running.",
			nullptr, "memory_cards", {{nullptr, nullptr}}, "Mcd001.ps2"},
		{"pcsx2_memcard_slot2_file", "Slot 2 Card", nullptr,
			"Select an existing .ps2 card from <system>/pcsx2/memcards. Changes apply immediately while content is running.",
			nullptr, "memory_cards", {{nullptr, nullptr}}, "Mcd002.ps2"},
		{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, {{nullptr, nullptr}}, nullptr},
	};

	// Do not invent Mcd001.ps2/Mcd002.ps2 entries when the directory is empty.
	// The final two definitions are deliberately last so a null key here only
	// omits the Card selectors while retaining the Enabled options above them.
	//
	// definitions[] is static, so a key cleared on one call is still cleared on
	// the next - and cannot be found by key any more either. Note where the two
	// selectors are while their keys are intact, and put the keys back before
	// deciding again, or a first run with an empty memcards directory would
	// hide the selectors for the rest of the process.
	static constexpr const char* SLOT_FILE_KEYS[2] = {"pcsx2_memcard_slot1_file", "pcsx2_memcard_slot2_file"};
	static size_t slot_file_defs[2] = {std::size(definitions), std::size(definitions)};
	for (size_t k = 0; k < std::size(SLOT_FILE_KEYS); k++)
	{
		if (slot_file_defs[k] == std::size(definitions))
		{
			for (size_t i = 0; i < std::size(definitions); i++)
			{
				if (definitions[i].key && std::strcmp(definitions[i].key, SLOT_FILE_KEYS[k]) == 0)
				{
					slot_file_defs[k] = i;
					break;
				}
			}
		}

		if (slot_file_defs[k] != std::size(definitions))
			definitions[slot_file_defs[k]].key = s_memcard_names.empty() ? nullptr : SLOT_FILE_KEYS[k];
	}

	// fill in the discovered BIOS list (bounded by the option value array size)
	for (retro_core_option_v2_definition& def : definitions)
	{
		if (!def.key || std::strcmp(def.key, "pcsx2_bios") != 0)
			continue;
		const size_t max_bios = std::min(s_bios_names.size(), std::size(def.values) - 1);
		for (size_t i = 0; i < max_bios; i++)
			def.values[i] = {s_bios_names[i].c_str(), nullptr};
		def.values[max_bios] = {nullptr, nullptr};
		break;
	}

	// fill in the discovered memory card list (bounded by the option value array size)
	bool warned_about_memcard_limit = false;
	for (retro_core_option_v2_definition& def : definitions)
	{
		if (!def.key)
			break;
		if (std::strcmp(def.key, "pcsx2_memcard_slot1_file") != 0 &&
			std::strcmp(def.key, "pcsx2_memcard_slot2_file") != 0)
			continue;

		const size_t max_cards = std::min(s_memcard_names.size(), std::size(def.values) - 1);
		for (size_t i = 0; i < max_cards; i++)
			def.values[i] = {s_memcard_names[i].c_str(), nullptr};
		def.values[max_cards] = {nullptr, nullptr};

		// The declared default (Mcd001.ps2) only exists if the user happens to
		// have that card; naming a value that is not in the list leaves the
		// frontend to pick for us. Fall back to the first card we found.
		bool default_present = false;
		for (size_t i = 0; i < max_cards && !default_present; i++)
			default_present = (std::strcmp(def.default_value, def.values[i].value) == 0);
		if (!default_present && max_cards > 0)
			def.default_value = def.values[0].value;
		if (!warned_about_memcard_limit && s_memcard_names.size() > max_cards)
		{
			Console.WarningFmt("{} memory cards found, but only {} can be exposed as a core option.",
				s_memcard_names.size(), max_cards);
			warned_about_memcard_limit = true;
		}
	}

	unsigned version = 0;
	if (s_environ_cb(RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION, &version) && version >= 2)
	{
		retro_core_options_v2 options = {categories, definitions};
		s_environ_cb(RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2, &options);
		return;
	}

	// legacy fallback: "Description; value1|value2" strings
	static std::vector<std::string> legacy_storage;
	legacy_storage.clear();
	// legacy[] holds c_str() pointers into these strings, so the storage must not
	// reallocate while it is being filled (a moved short string changes address)
	legacy_storage.reserve(std::size(definitions));
	std::vector<retro_variable> legacy;
	for (const retro_core_option_v2_definition& def : definitions)
	{
		if (!def.key)
			break;

		std::string str = fmt::format("{}; ", def.desc);
		// default first, as required by the legacy API
		str += def.default_value;
		for (const retro_core_option_value& v : def.values)
		{
			if (!v.value)
				break;
			if (std::strcmp(v.value, def.default_value) != 0)
				str += fmt::format("|{}", v.value);
		}
		legacy_storage.push_back(std::move(str));
		legacy.push_back({def.key, legacy_storage.back().c_str()});
	}
	legacy.push_back({nullptr, nullptr});
	s_environ_cb(RETRO_ENVIRONMENT_SET_VARIABLES, legacy.data());
}

void LibretroHost::ReadCoreOptions(bool startup)
{
	const auto get_option = [](const char* key, const char* fallback) -> const char* {
		retro_variable var = {key, nullptr};
		if (s_environ_cb && s_environ_cb(RETRO_ENVIRONMENT_GET_VARIABLE, &var) && var.value)
			return var.value;
		return fallback;
	};

	// The default has to name a renderer this build actually has: a core without
	// Vulkan still reads a stale pcsx2_renderer = "vulkan" out of the frontend's
	// config, and on the readback path there is no HW type to clamp it back to.
#if defined(ENABLE_VULKAN)
	const char* renderer = get_option("pcsx2_renderer", "vulkan");
	GSRendererType renderer_type = GSRendererType::VK;
#elif defined(ENABLE_OPENGL)
	const char* renderer = get_option("pcsx2_renderer", "opengl");
	GSRendererType renderer_type = GSRendererType::OGL;
#else
	const char* renderer = get_option("pcsx2_renderer", "software");
	GSRendererType renderer_type = GSRendererType::SW;
#endif
	if (std::strcmp(renderer, "software") == 0)
		renderer_type = GSRendererType::SW;
#ifdef ENABLE_OPENGL
	else if (std::strcmp(renderer, "opengl") == 0)
		renderer_type = GSRendererType::OGL;
#endif

	// The frontend's HW-render context type is negotiated once, when the
	// content loads, and cannot change under a running session — so on a
	// HW-render path the graphics API is fixed for the session even though the
	// option can still be changed. Switching to and from the software renderer
	// stays live: it presents through whichever device is already up.
	if (HWRenderActive() && renderer_type != GSRendererType::SW)
	{
		const GSRendererType hw_type =
			(s_hw_render == HWRender::Vulkan) ? GSRendererType::VK : GSRendererType::OGL;
		if (renderer_type != hw_type)
		{
			static bool warned = false;
			if (!warned)
			{
				warned = true;
				Console.WarningFmt("Renderer API can only be changed by restarting the content; "
								   "staying on {}.",
					(hw_type == GSRendererType::VK) ? "Vulkan" : "OpenGL");
			}
			renderer_type = hw_type;
		}
	}

	s_settings_interface.SetIntValue("EmuCore/GS", "Renderer", static_cast<int>(renderer_type));

	const u32 upscale = std::clamp<u32>(StringUtil::FromChars<u32>(get_option("pcsx2_upscale_multiplier", "1")).value_or(1), 1, MAX_UPSCALE);
	s_settings_interface.SetFloatValue("EmuCore/GS", "upscale_multiplier", static_cast<float>(upscale));
	s_opt_upscale = upscale;
	s_out_width.store(DEFAULT_WIDTH * upscale, std::memory_order_release);
	s_out_height.store(DEFAULT_HEIGHT * upscale, std::memory_order_release);
	// The HW-render paths hand the frontend the rendered image directly; only
	// wire the GPU->CPU readback when we're actually on the readback path.
	if (!HWRenderActive())
		GSSetFramebufferReadback(&FramebufferReadbackCallback, DEFAULT_WIDTH * upscale, DEFAULT_HEIGHT * upscale);

	// graphics quality
	const auto get_int_option = [&get_option](const char* key, const char* fallback) {
		return StringUtil::FromChars<int>(get_option(key, fallback)).value_or(StringUtil::FromChars<int>(fallback).value_or(0));
	};

	static constexpr std::pair<const char*, AccBlendLevel> blend_levels[] = {
		{"minimum", AccBlendLevel::Minimum}, {"basic", AccBlendLevel::Basic}, {"medium", AccBlendLevel::Medium},
		{"high", AccBlendLevel::High}, {"full", AccBlendLevel::Full}, {"maximum", AccBlendLevel::Maximum}};
	// GSHardwareDownloadMode: Enabled=0, Unsynchronized=3, Disabled=4
	{
		const char* dl = get_option("pcsx2_hw_download_mode", "accurate");
		int dl_mode = 0;
		if (std::strcmp(dl, "unsynchronized") == 0)
			dl_mode = 3;
		else if (std::strcmp(dl, "disabled") == 0)
			dl_mode = 4;
		s_settings_interface.SetIntValue("EmuCore/GS", "HWDownloadMode", dl_mode);
	}

	const char* blend = get_option("pcsx2_blending_accuracy", "basic");
	for (const auto& [name, level] : blend_levels)
	{
		if (std::strcmp(blend, name) == 0)
		{
			s_settings_interface.SetIntValue("EmuCore/GS", "accurate_blending_unit", static_cast<int>(level));
			break;
		}
	}

	static constexpr std::pair<const char*, BiFiltering> bi_filters[] = {
		{"nearest", BiFiltering::Nearest}, {"bilinear_ps2", BiFiltering::PS2},
		{"bilinear_forced", BiFiltering::Forced}, {"bilinear_forced_sprite", BiFiltering::Forced_But_Sprite}};
	const char* bi = get_option("pcsx2_texture_filtering", "bilinear_ps2");
	for (const auto& [name, mode] : bi_filters)
	{
		if (std::strcmp(bi, name) == 0)
		{
			s_settings_interface.SetIntValue("EmuCore/GS", "filter", static_cast<int>(mode));
			break;
		}
	}

	static constexpr std::pair<const char*, TriFiltering> tri_filters[] = {
		{"auto", TriFiltering::Automatic}, {"off", TriFiltering::Off}, {"ps2", TriFiltering::PS2},
		{"forced", TriFiltering::Forced}};
	const char* tri = get_option("pcsx2_trilinear_filtering", "auto");
	for (const auto& [name, mode] : tri_filters)
	{
		if (std::strcmp(tri, name) == 0)
		{
			s_settings_interface.SetIntValue("EmuCore/GS", "TriFilter", static_cast<int>(mode));
			break;
		}
	}

	s_settings_interface.SetIntValue("EmuCore/GS", "MaxAnisotropy", get_int_option("pcsx2_anisotropic_filtering", "0"));
	s_settings_interface.SetIntValue("EmuCore/GS", "dithering_ps2", get_int_option("pcsx2_dithering", "2"));
	s_settings_interface.SetBoolValue("EmuCore/GS", "hw_mipmap",
		std::strcmp(get_option("pcsx2_mipmapping", "enabled"), "enabled") == 0);
	s_settings_interface.SetIntValue("EmuCore/GS", "deinterlace_mode", get_int_option("pcsx2_deinterlace_mode", "0"));
	s_settings_interface.SetBoolValue("EmuCore/GS", "fxaa",
		std::strcmp(get_option("pcsx2_fxaa", "disabled"), "enabled") == 0);
	s_settings_interface.SetIntValue("EmuCore/GS", "CASMode",
		std::strcmp(get_option("pcsx2_cas_mode", "disabled"), "sharpen") == 0 ?
			static_cast<int>(GSCASMode::SharpenOnly) :
			static_cast<int>(GSCASMode::Disabled));
	s_settings_interface.SetIntValue("EmuCore/GS", "CASSharpness", get_int_option("pcsx2_cas_sharpness", "50"));
	s_settings_interface.SetBoolValue("EmuCore/GS", "SkipDuplicateFrames",
		std::strcmp(get_option("pcsx2_skip_duplicate_frames", "enabled"), "enabled") == 0);
	s_internal_limiter.store(std::strcmp(get_option("pcsx2_frame_limiter", "frontend"), "internal") == 0,
		std::memory_order_release);

	// How much audio the emulator keeps buffered before the frontend takes it.
	// PCSX2's own default is 50ms, chosen for a native backend it hands samples
	// to directly; a core sits behind a frontend that adds its own latency and
	// pulls once per retro_run, so the same 50ms empties on any hitch. Changing
	// it recreates the output stream (SPU2::CheckForConfigChanges), so it
	// applies without a restart.
	s_settings_interface.SetIntValue("SPU2/Output", "BufferMS", get_int_option("pcsx2_audio_buffer_ms", "100"));

	// patches
	const bool widescreen = std::strcmp(get_option("pcsx2_widescreen_patches", "disabled"), "enabled") == 0;
	s_settings_interface.SetBoolValue("EmuCore", "EnableWideScreenPatches", widescreen);
	s_settings_interface.SetBoolValue("EmuCore", "EnableNoInterlacingPatches",
		std::strcmp(get_option("pcsx2_no_interlacing_patches", "disabled"), "enabled") == 0);

	// reported display aspect
	const char* aspect = get_option("pcsx2_aspect_ratio", "auto");
	float aspect_value = 4.0f / 3.0f;
	if (std::strcmp(aspect, "16:9") == 0 || (std::strcmp(aspect, "auto") == 0 && widescreen))
		aspect_value = 16.0f / 9.0f;
	s_aspect_bits.store(std::bit_cast<u32>(aspect_value), std::memory_order_release);

	// performance
	// PCEE2_OSD=1: performance overlay (fps/speed + EE/GS/VU thread loads +
	// GPU usage) for diagnosing whether a heavy scene is CPU- or GPU-bound.
	if (std::getenv("PCEE2_OSD"))
	{
		s_settings_interface.SetBoolValue("EmuCore/GS", "OsdShowSpeed", true);
		s_settings_interface.SetBoolValue("EmuCore/GS", "OsdShowFPS", true);
		s_settings_interface.SetBoolValue("EmuCore/GS", "OsdShowResolution", true);
		s_settings_interface.SetBoolValue("EmuCore/GS", "OsdShowCPU", true);
		s_settings_interface.SetBoolValue("EmuCore/GS", "OsdShowGPU", true);
	}

	// MTVU: VU1 on its own thread — the single biggest speedup on multi-core
	// ARM (pcee2's base defaults it off; yaps2's libretro core defaults it on).
	s_settings_interface.SetBoolValue("EmuCore/Speedhacks", "vuThread",
		std::strcmp(get_option("pcsx2_mtvu", "enabled"), "enabled") == 0);
	s_settings_interface.SetBoolValue("EmuCore/Speedhacks", "vu1Instant",
		std::strcmp(get_option("pcsx2_instant_vu1", "enabled"), "enabled") == 0);
	s_settings_interface.SetIntValue("EmuCore/Speedhacks", "EECycleRate", get_int_option("pcsx2_ee_cycle_rate", "0"));
	s_settings_interface.SetIntValue("EmuCore/Speedhacks", "EECycleSkip", get_int_option("pcsx2_ee_cycle_skip", "0"));

	// multitap: enable adapters and build the libretro-port -> pad-index map
	{
		const char* multitap = get_option("pcsx2_multitap", "disabled");
		const bool mt1 = (std::strcmp(multitap, "port1") == 0 || std::strcmp(multitap, "both") == 0);
		const bool mt2 = (std::strcmp(multitap, "port2") == 0 || std::strcmp(multitap, "both") == 0);
		s_settings_interface.SetBoolValue("Pad", "MultitapPort1", mt1);
		s_settings_interface.SetBoolValue("Pad", "MultitapPort2", mt2);

		s_pad_map.clear();
		s_pad_map.push_back(0); // 1A
		if (mt1)
		{
			s_pad_map.push_back(2); // 1B
			s_pad_map.push_back(3); // 1C
			s_pad_map.push_back(4); // 1D
		}
		s_pad_map.push_back(1); // 2A
		if (mt2)
		{
			s_pad_map.push_back(5); // 2B
			s_pad_map.push_back(6); // 2C
			s_pad_map.push_back(7); // 2D
		}

		// all mapped pads are DualShock 2s, with the configured analog response
		const float axis_scale = static_cast<float>(get_int_option("pcsx2_axis_scale", "133")) / 100.0f;
		const float axis_deadzone = static_cast<float>(get_int_option("pcsx2_axis_deadzone", "0")) / 100.0f;
		for (const u32 pad : s_pad_map)
		{
			const std::string section = fmt::format("Pad{}", pad + 1);
			s_settings_interface.SetStringValue(section.c_str(), "Type", "DualShock2");
			s_settings_interface.SetFloatValue(section.c_str(), "AxisScale", axis_scale);
			s_settings_interface.SetFloatValue(section.c_str(), "Deadzone", axis_deadzone);
		}

		s_rumble_enabled = std::strcmp(get_option("pcsx2_rumble", "enabled"), "enabled") == 0;
	}

	// lightguns: configure GunCon2 USB devices; the presence of the Relative*
	// binding keys switches the device to relative-axis aiming, which we feed
	// from the frontend's lightgun coordinates
	if (startup)
	{
		const char* lightgun = get_option("pcsx2_lightgun", "disabled");
		s_lightgun_mask = 0;
		if (std::strcmp(lightgun, "usb1") == 0 || std::strcmp(lightgun, "both") == 0)
			s_lightgun_mask |= 1;
		if (std::strcmp(lightgun, "usb2") == 0 || std::strcmp(lightgun, "both") == 0)
			s_lightgun_mask |= 2;

		for (u32 usb_port = 0; usb_port < 2; usb_port++)
		{
			const std::string section = fmt::format("USB{}", usb_port + 1);
			if (s_lightgun_mask & (1u << usb_port))
			{
				s_settings_interface.SetStringValue(section.c_str(), "Type", "guncon2");
				for (const char* bind : {"RelativeLeft", "RelativeRight", "RelativeUp", "RelativeDown"})
					s_settings_interface.SetStringValue(section.c_str(), fmt::format("guncon2_{}", bind).c_str(), "None");
			}
			else
			{
				s_settings_interface.SetStringValue(section.c_str(), "Type", "None");
			}
		}
	}

	if (startup)
	{
		const char* bios = get_option("pcsx2_bios", "auto");
		if (std::strcmp(bios, "auto") != 0)
			s_settings_interface.SetStringValue("Filenames", "BIOS", bios);

		s_settings_interface.SetBoolValue("EmuCore", "EnableFastBoot",
			std::strcmp(get_option("pcsx2_fast_boot", "enabled"), "enabled") == 0);

		// Diagnostic toggles: the master switch forces every CPU to its
		// interpreter when off; the four per-CPU switches then let a tester
		// re-enable the dynarecs one at a time to bisect which recompiler
		// causes a crash. Effective enable = master AND per-CPU. Recompiler
		// enables are latched when the CPUs are created, so this only takes
		// effect on a fresh boot.
		const bool jit = std::strcmp(get_option("pcsx2_cpu_recompiler", "enabled"), "enabled") == 0;
		const auto rec_on = [&](const char* key) {
			return jit && std::strcmp(get_option(key, "enabled"), "enabled") == 0;
		};
		const bool ee = rec_on("pcsx2_rec_ee");
		const bool iop = rec_on("pcsx2_rec_iop");
		const bool vu0 = rec_on("pcsx2_rec_vu0");
		const bool vu1 = rec_on("pcsx2_rec_vu1");
		s_settings_interface.SetBoolValue("EmuCore/CPU/Recompiler", "EnableEE", ee);
		s_settings_interface.SetBoolValue("EmuCore/CPU/Recompiler", "EnableIOP", iop);
		s_settings_interface.SetBoolValue("EmuCore/CPU/Recompiler", "EnableVU0", vu0);
		s_settings_interface.SetBoolValue("EmuCore/CPU/Recompiler", "EnableVU1", vu1);
		if (!(ee && iop && vu0 && vu1))
			Console.WriteLnFmt("Recompiler state via core options: EE={} IOP={} VU0={} VU1={} (off = interpreter).",
				ee, iop, vu0, vu1);
	}

	// Memory Card options are read at startup and whenever RetroArch reports a
	// Core Option change. Keep all four values in the same settings layer before
	// the caller's single VMManager::ApplySettings() so PCSX2 can perform its
	// native change detection and eject/reopen path.
	s_settings_interface.SetBoolValue("MemoryCards", "Slot1_Enable",
		std::strcmp(get_option("pcsx2_memcard_slot1_enable", "enabled"), "enabled") == 0);
	s_settings_interface.SetBoolValue("MemoryCards", "Slot2_Enable",
		std::strcmp(get_option("pcsx2_memcard_slot2_enable", "enabled"), "enabled") == 0);
	if (startup)
	{
		// The discovered list controls which selectors are registered, not
		// whether RetroArch has a current value for those selectors.
		if (!s_memcard_names.empty())
		{
			s_settings_interface.SetStringValue("MemoryCards", "Slot1_Filename",
				get_option("pcsx2_memcard_slot1_file", "Mcd001.ps2"));
			s_settings_interface.SetStringValue("MemoryCards", "Slot2_Filename",
				get_option("pcsx2_memcard_slot2_file", "Mcd002.ps2"));
		}
	}
	else
	{
		// Runtime reads must not be gated by the registration-time candidate
		// list. Query both keys every time and leave the existing setting alone
		// if the frontend does not expose one of them.
		if (const char* filename = get_option("pcsx2_memcard_slot1_file", nullptr))
			s_settings_interface.SetStringValue("MemoryCards", "Slot1_Filename", filename);
		if (const char* filename = get_option("pcsx2_memcard_slot2_file", nullptr))
			s_settings_interface.SetStringValue("MemoryCards", "Slot2_Filename", filename);
	}
}

void LibretroHost::CPUThreadMain()
{
	s_cpu_thread_id = std::this_thread::get_id();

	const bool init_ok = VMManager::Internal::CPUThreadInitialize();
	if (!init_ok)
		Console.Error("CPUThreadInitialize() failed.");
	s_cpu_thread_initialized.store(true, std::memory_order_release);

	for (;;)
	{
		{
			std::unique_lock lock(s_session_mutex);
			s_session_cv.wait(lock, []() { return s_boot_requested || s_exit_requested; });
			if (s_exit_requested)
				break;
			s_boot_requested = false;
		}

		// HW render: the GS can only open once the frontend's context exists —
		// on Vulkan its context negotiation opens MTGS itself (GSDeviceVK adopts
		// the shared instance) and then fires context_reset; on GL context_reset
		// is where the frontend's context gets captured to share from. Booting
		// the VM before that would open the GS against a context of our own.
		while (init_ok && HWRenderActive() && !s_context_ready.load(std::memory_order_acquire) &&
			s_running.load(std::memory_order_acquire))
			std::this_thread::sleep_for(std::chrono::milliseconds(1));

		if (init_ok)
		{
			VMManager::ApplySettings();

			if (VMManager::Initialize(s_boot_params) == VMBootResult::StartupSuccess)
			{
				VMManager::SetState(VMState::Running);
				// Nominal means the VM throttles itself to the console's rate;
				// Unlimited leaves the pacing to the frontend, which is the default.
				VMManager::SetLimiterMode(s_internal_limiter.load(std::memory_order_acquire) ?
					LimiterModeType::Nominal : LimiterModeType::Unlimited);
				while (s_running.load(std::memory_order_acquire))
				{
					const VMState state = VMManager::GetState();
					if (state == VMState::Resetting)
					{
						// retro_reset() (via Host::RunOnCPUThread) already called
						// VMManager::Reset() once to get here - that call only
						// flagged VMState::Resetting and returned, because it ran
						// while state was Running (see VMManager::Reset()'s own
						// comment: it exits the rec's tight execution loop first,
						// then expects to be called again to do the real work).
						// Calling it again now, with state == Resetting, bypasses
						// that early-out and actually runs hwReset()/SPU2::Reset()/
						// etc., ending by flipping state back to Running - exactly
						// the pattern every other PCSX2 frontend's main loop uses
						// (see QtHost's `case VMState::Resetting: VMManager::Reset();`).
						// Without this case here, this loop's condition just saw a
						// non-Running state and fell through to VMManager::Shutdown()
						// below - i.e. "Reštart" silently tore the whole VM down
						// instead of resetting it, out from under retro_run() still
						// being called every frame by the frontend. This is also why
						// the lock has to be here and not in retro_reset(): this is
						// where SPU2's output stream is actually torn down and
						// recreated, not there.
						std::unique_lock lock(s_audio_stream_mutex);
						VMManager::Reset();
						continue;
					}

					if (state != VMState::Running)
						break;

					VMManager::Execute();
				}
				VMManager::Shutdown(false);
			}
			else
			{
				Console.Error("VMManager::Initialize() failed.");
			}
		}

		s_running.store(false, std::memory_order_release);

		// wake up a potentially waiting retro_run()
		{
			std::unique_lock lock(s_frame_mutex);
			s_frame_ready = true;
		}
		s_frame_cv.notify_all();

		// let retro_unload_game() proceed
		{
			std::unique_lock lock(s_session_mutex);
			s_session_active = false;
		}
		s_session_cv.notify_all();
	}

	if (init_ok)
		VMManager::Internal::CPUThreadShutdown();
}

void LibretroHost::DrainCPUWork()
{
	std::deque<std::function<void()>> work;
	{
		std::unique_lock lock(s_cpu_work_mutex);
		work.swap(s_cpu_work);
		s_cpu_work_pending.store(false, std::memory_order_release);
	}
	for (auto& fn : work)
		fn();
}

//////////////////////////////////////////////////////////////////////////
// Vulkan context negotiation (ported from yaps2). create_device runs on the
// frontend thread while it builds its Vulkan context: it stashes the shared
// instance/GPU + the frontend's required extensions/layers/features into
// VKLibretro::Init, then opens MTGS so GSDeviceVK constructs against the shared
// device (its vkCreateDevice is intercepted by the VKLibretro wraps, which
// capture the resulting VkDevice for the reply below).
//////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_VULKAN
static const VkApplicationInfo* GetVulkanApplicationInfo(void)
{
	static VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app_info.pApplicationName = "PCEE2";
	app_info.applicationVersion = VK_MAKE_VERSION(2, 0, 0);
	app_info.pEngineName = "PCEE2";
	app_info.engineVersion = VK_MAKE_VERSION(2, 0, 0);
	app_info.apiVersion = VK_API_VERSION_1_1;
	return &app_info;
}

static bool CreateVulkanDevice(retro_vulkan_context* context, VkInstance instance, VkPhysicalDevice gpu,
	VkSurfaceKHR surface, PFN_vkGetInstanceProcAddr get_instance_proc_addr, const char** required_device_extensions,
	unsigned num_required_device_extensions, const char** required_device_layers,
	unsigned num_required_device_layers, const VkPhysicalDeviceFeatures* required_features)
{
	VKLibretro::Init.instance = instance;
	VKLibretro::Init.gpu = gpu;
	VKLibretro::Init.get_instance_proc_addr = get_instance_proc_addr;
	VKLibretro::Init.required_device_extensions = required_device_extensions;
	VKLibretro::Init.num_required_device_extensions = num_required_device_extensions;
	VKLibretro::Init.required_device_layers = required_device_layers;
	VKLibretro::Init.num_required_device_layers = num_required_device_layers;
	VKLibretro::Init.required_features = required_features;

	// Bring up the GS thread now: GSDeviceVK adopts Init.instance/gpu and the
	// wrapped vkCreateDevice fills Init.device with the shared device.
	if (!MTGS::IsOpen() && !MTGS::WaitForOpen())
	{
		Console.Error("MTGS::WaitForOpen failed during Vulkan negotiation.");
		return false;
	}

	GSDeviceVK* dev = GSDeviceVK::GetInstance();
	if (!dev || VKLibretro::Init.device == VK_NULL_HANDLE)
	{
		Console.Error("GS device missing after negotiation open.");
		return false;
	}

	context->gpu = dev->GetPhysicalDevice();
	context->device = VKLibretro::Init.device;
	context->queue = dev->GetGraphicsQueue();
	context->queue_family_index = dev->GetGraphicsQueueFamilyIndex();
	context->presentation_queue = context->queue;
	context->presentation_queue_family_index = context->queue_family_index;
	return true;
}
#endif

static void OnContextReset(void)
{
	retro_hw_render_interface* iface = nullptr;

#ifdef ENABLE_VULKAN
	if (!s_environ_cb(RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE, &iface) || !iface ||
		iface->interface_type != RETRO_HW_RENDER_INTERFACE_VULKAN)
	{
		Console.Error("Failed to get Vulkan HW render interface.");
		return;
	}
	VKLibretro::SetHWRenderInterface(iface);
#endif

	s_context_ready.store(true, std::memory_order_release);
}

static void OnContextDestroy(void)
{
	// Called by the frontend on its own thread whenever it tears the video
	// driver down under us - a fullscreen toggle (RetroArch's F) does it just
	// like an av_info change. The GS thread keeps running and every queue
	// submit goes through the interface that is about to be retired, so stop
	// presenting, let the GS thread finish what it already has, and only then
	// drop the interface (SetHWRenderInterface waits out a submit that is
	// already inside the wrapper).
#ifdef ENABLE_VULKAN
	VKLibretro::AbortPacing();
#endif
	s_context_ready.store(false, std::memory_order_release);
	if (MTGS::IsOpen())
	{
		std::atomic_bool gs_drained{false};
		MTGS::RunOnGSThread([&gs_drained]() { gs_drained.store(true, std::memory_order_release); });
		for (int i = 0; i < 1000 && !gs_drained.load(std::memory_order_acquire); i++)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
#ifdef ENABLE_VULKAN
	VKLibretro::SetHWRenderInterface(nullptr);
#endif
}

#ifdef ENABLE_OPENGL
//////////////////////////////////////////////////////////////////////////
// OpenGL context handoff.
//
// There is no retro_hw_render_interface for GL to ask the frontend for: its
// context_reset, running on the frontend thread with the context current, is
// the only moment the core can see that context at all. Grab it there; the GS
// thread then builds a context sharing its objects (GLLibretro) and the CPU
// thread is released to boot the VM against it.
//////////////////////////////////////////////////////////////////////////

static void OnGLContextReset(void)
{
	Error error;
	if (!GLLibretro::CaptureFrontendContext(&error))
	{
		// Nothing to share from, so drop to the readback path: the GL device
		// will bring up a surfaceless context of its own and the frame comes
		// back through the CPU. Slower, but it boots — leaving s_context_ready
		// clear here would park the CPU thread on a context that never arrives.
		Console.ErrorFmt("Failed to capture the frontend's GL context: {}", error.GetDescription());
		Console.Error("  Falling back to the readback present path.");
		GLLibretro::Active = false;
		s_hw_render = HWRender::None;
		GSLibretro::Active = false;
		GSSetFramebufferReadback(
			&FramebufferReadbackCallback, DEFAULT_WIDTH * s_opt_upscale, DEFAULT_HEIGHT * s_opt_upscale);
		s_context_ready.store(true, std::memory_order_release);
		return;
	}

	// Second time through: the frontend destroyed its context and built a new
	// one (its fullscreen toggle does exactly that), so every GL object the GS
	// holds belongs to a share group nothing can reach any more. The device has
	// to be rebuilt against the new context — but NOT from here.
	//
	// This callback runs part-way through the frontend rebuilding its video
	// driver, with its thread inside EGL. Recreating the device means the GS
	// thread calls eglMakeCurrent to drop its old context, which blocks on the
	// same driver-wide lock the frontend thread is holding; waiting here for
	// the GS thread to finish therefore deadlocks the two against each other
	// (observed: GS thread parked in GLContextEGL::DoneCurrent, frontend parked
	// in this callback). Flag it instead and let retro_run do the work, once
	// the frontend is back in its normal frame loop and out of EGL.
	if (s_gl_context_seen)
		s_gl_needs_reopen.store(true, std::memory_order_release);

	s_gl_context_seen = true;
	s_gl_present_fbo = 0;

	s_context_ready.store(true, std::memory_order_release);
}

static void OnGLContextDestroy(void)
{
	// Runs on the frontend thread, before the frontend destroys its context,
	// while the GS thread keeps going. Stop the handoff first so the GS thread
	// can't park waiting for a retro_run that is not coming, then have it let
	// go of its own context.
	//
	// That last part has to happen HERE, and not when the device is torn down
	// later. The GS thread's context shares an EGL display with the frontend's,
	// and the frontend ends by calling eglTerminate on it — after which every
	// handle the GS thread holds points into freed driver state, and simply
	// unbinding the context segfaults inside Mesa (observed: SIGSEGV in
	// eglMakeCurrent, from GSDeviceOGL::Destroy). This is the last moment those
	// handles are still good.
	GLLibretro::AbortPacing();
	s_context_ready.store(false, std::memory_order_release);
	if (MTGS::IsOpen())
	{
		std::atomic_bool gs_released{false};
		MTGS::RunOnGSThread([&gs_released]() {
			// Retire the queued primitives first. GSreopen() would flush them
			// itself, but that runs after the context is gone, and a draw
			// issued then writes through a vertex buffer whose mapping went
			// with it (SIGSEGV in IASetVertexBuffer). Here the context is
			// still live and the draws land normally.
			if (g_gs_renderer)
				g_gs_renderer->Flush(GSState::GSFlushReason::GSREOPEN);

			if (g_gs_device && g_gs_device->GetRenderAPI() == RenderAPI::OpenGL)
				static_cast<GSDeviceOGL*>(g_gs_device.get())->AbandonContext(true);
			gs_released.store(true, std::memory_order_release);
		});
		for (int i = 0; i < 2000 && !gs_released.load(std::memory_order_acquire); i++)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		if (!gs_released.load(std::memory_order_acquire))
			Console.Error("GL: The GS thread did not let go of its context in time.");
	}

	// The FBO went with the context; a new one gets made against whatever
	// context_reset hands over next.
	s_gl_present_fbo = 0;
	GLLibretro::ReleaseFrontendContext();
}
#endif // ENABLE_OPENGL

//////////////////////////////////////////////////////////////////////////
// libretro entry points
//////////////////////////////////////////////////////////////////////////

void retro_set_environment(retro_environment_t cb)
{
	s_environ_cb = cb;

	bool no_game = true;
	cb(RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME, &no_game);

	retro_log_callback log_cb{};
	if (cb(RETRO_ENVIRONMENT_GET_LOG_INTERFACE, &log_cb))
		s_log_cb = log_cb.log;

	// Has to happen before anything touches a file - on Android the content,
	// and possibly the system and save directories too, are only reachable
	// through the frontend.
	InitializeVFS(cb);

	RegisterCoreOptions();
}

void retro_set_video_refresh(retro_video_refresh_t cb) { s_video_cb = cb; }
void retro_set_audio_sample(retro_audio_sample_t cb) {}
void retro_set_audio_sample_batch(retro_audio_sample_batch_t cb) { s_audio_batch_cb = cb; }
void retro_set_input_poll(retro_input_poll_t cb) { s_input_poll_cb = cb; }
void retro_set_input_state(retro_input_state_t cb) { s_input_state_cb = cb; }

unsigned retro_api_version(void)
{
	return RETRO_API_VERSION;
}

void retro_get_system_info(struct retro_system_info* info)
{
	std::memset(info, 0, sizeof(*info));
	info->library_name = "PCEE2";
	info->library_version = GIT_REV;
	info->valid_extensions = "iso|chd|cue|m3u|cso|zso|gz|bin|mdf|nrg|elf|irx";
	info->need_fullpath = true;
	info->block_extract = true;
}

void retro_get_system_av_info(struct retro_system_av_info* info)
{
	const u32 fps_bits = s_vm_fps_bits.load(std::memory_order_acquire);
	const float fps = fps_bits ? std::bit_cast<float>(fps_bits) : 59.94f;

	std::memset(info, 0, sizeof(*info));
	// On a HW-render path the geometry is the native resolution rather than the
	// canvas, for the reasons in SyncHWRenderGeometry; before the first frame
	// there is no native size to report yet and the configured output stands in.
	const u64 native = HWRenderActive() ? GSLibretro::NativeSize.load(std::memory_order_acquire) : 0;
	info->geometry.base_width = native ? static_cast<u32>(native >> 32) :
										 s_out_width.load(std::memory_order_acquire);
	info->geometry.base_height = native ? static_cast<u32>(native & 0xffffffffu) :
										  s_out_height.load(std::memory_order_acquire);
	// The HW-render canvas is aspect-expanded and can exceed the plain upscale
	// rectangle, so advertise the frontend the larger backbuffer bound.
	info->geometry.max_width = HWRenderActive() ? GSLibretro::kMaxCanvasWidth : MAX_WIDTH;
	info->geometry.max_height = HWRenderActive() ? GSLibretro::kMaxCanvasHeight : MAX_HEIGHT;
	const u32 aspect_bits = s_aspect_bits.load(std::memory_order_acquire);
	info->geometry.aspect_ratio = aspect_bits ? std::bit_cast<float>(aspect_bits) : (4.0f / 3.0f);
	info->timing.fps = static_cast<double>(fps);
	info->timing.sample_rate = static_cast<double>(s_audio_sample_rate.load(std::memory_order_acquire));
}

void retro_init(void)
{
	// Both of these are idempotent, and retro_deinit takes them back down in
	// the reverse order, so a frontend cycling retro_deinit/retro_init ends up
	// with the same chain (page fault handler in front, crash handler behind
	// it) rather than one clobbering the other.
	CrashHandler::Install();

	static bool s_atexit_registered = false;
	if (!s_atexit_registered)
	{
		std::atexit(&ShutdownCoreAtExit);
		s_atexit_registered = true;
	}

	Log::SetHostOutputLevel(LOGLEVEL_INFO, &HostLogCallback);

	LogVFSStatus();
}

void retro_deinit(void)
{
	// Stop the CPU thread completely: leaving anything running past deinit
	// breaks the frontend's process teardown on Windows (threads are killed
	// before DLL atexit handlers run, deadlocking joins under the loader
	// lock). The idempotent signal-handler installs make the next
	// retro_init/load cycle safe again.
	if (s_cpu_thread.joinable())
	{
		{
			std::unique_lock lock(s_session_mutex);
			s_exit_requested = true;
		}
		s_session_cv.notify_all();

		// The exit request is only looked at between sessions. If we are called
		// without retro_unload_game() having wound one down, the CPU thread is
		// parked in PumpMessagesOnCPUThread() waiting for a run token nobody is
		// going to post, and the join below would wait on it forever. Break the
		// pacing handshake too, so that park always has a way out.
		s_running.store(false, std::memory_order_release);
		{
			std::unique_lock lock(s_frame_mutex);
			s_run_token = true;
		}
		s_frame_cv.notify_all();

		s_cpu_thread.join();
		{
			std::unique_lock lock(s_session_mutex);
			s_exit_requested = false;
		}
	}

	// MTVU's VU1 thread is a second background thread the core owns, and
	// nothing has joined it up to this point - the CPU thread above only ever
	// waits on it (WaitVU()), it never closes it. Left alone, the only thing
	// that would ever join it is VU_Thread's own destructor, and for the
	// global `vu1Thread` that lives in this DLL, that destructor only runs at
	// DLL_PROCESS_DETACH inside FreeLibrary() - under the loader lock. That is
	// exactly the deadlock class this function was already patched to avoid
	// for the CPU thread (see the pacing-handshake break above): a frontend
	// that calls retro_deinit() and then unloads the module joins the VU1 thread
	// from within DllMain, and the VU1 thread's own exit can need that same
	// lock to come back. Close it here instead, while we're still safely
	// outside FreeLibrary().
	vu1Thread.Close();

	// Hand the process' fault handling back to the frontend. Nothing of ours
	// runs from here on, and the frontend unloads this module while it keeps
	// going: a SIGSEGV/exception filter still pointing in here would report the
	// frontend's own crashes as ours - writing a dump of an address space that
	// was the size of a PS2 - and, after the unload, jump into freed code.
	// Reverse order of retro_init: the page fault handler chains to the crash
	// handler, so it has to be taken down first.
	PageFaultHandler::Uninstall();
	CrashHandler::Uninstall();
}

// Cue sheets and m3u playlists both name their files relative to themselves.
// Composing "directory of the list" + "name in the list" is a guess: Redump
// sheets routinely disagree with the file's actual case, which only bites on a
// case-sensitive filesystem, and a frontend serving the list over Android's
// SAF hands us a path whose siblings we cannot spell at all. So fall back to
// the directory listing and match without regard to case - FindFiles() goes
// through the same VFS hooks that read the list, so wherever the list itself
// could be opened, the files it names can be found.
static std::string ResolveSiblingFile(const std::string_view dir, const std::string_view name)
{
	// Lists authored on Windows spell the separator as a backslash, which is a
	// path separator nowhere else.
	std::string relative(name);
	std::replace(relative.begin(), relative.end(), '\\', '/');

	std::string path = Path::IsAbsolute(relative) ? relative : Path::Combine(dir, relative);
	if (FileSystem::FileExists(path.c_str()))
		return path;

	const std::string_view wanted = Path::GetFileName(relative);
	FileSystem::FindResultsArray results;
	if (FileSystem::FindFiles(std::string(dir).c_str(), "*", FILESYSTEM_FIND_FILES, &results))
	{
		for (const FILESYSTEM_FIND_DATA& fd : results)
		{
			if (StringUtil::compareNoCase(Path::GetFileName(fd.FileName), wanted))
				return fd.FileName;
		}
	}

	// Nothing matched - hand back the composed path so the caller can name it.
	return path;
}

// Redump dumps the PS2's CD-ROM titles - and most demo discs - as a cue sheet
// plus one or more binary tracks, so that is what a lot of libraries hold.
// CDVD has no cue parser (it opens the image directly, auto-detecting 2048,
// 2352 and 2448 byte sectors), so resolve the sheet here and boot the file its
// data track lives in. Audio tracks are lost in the process, which is no worse
// than the same disc converted to CHD: the ISO path reports a single-track TOC
// for every image format it supports.
static std::string ResolveCueSheet(const std::string& cue_path)
{
	const std::optional<std::string> sheet = FileSystem::ReadFileToString(cue_path.c_str());
	if (!sheet.has_value())
	{
		Console.Error(fmt::format("Failed to read cue sheet '{}'.", cue_path));
		return {};
	}

	// FILE "<name>" <format>, then the TRACK lines that belong to it. The
	// first data track is the one to boot; the first file is the fallback for
	// sheets whose first track is audio and which we therefore cannot serve
	// properly anyway.
	std::string_view first_file;
	std::string_view current_file;
	std::string_view data_file;
	for (const std::string_view line : StringUtil::SplitString(sheet.value(), '\n'))
	{
		const std::string_view trimmed = StringUtil::StripWhitespace(line);
		if (StringUtil::StartsWithNoCase(trimmed, "FILE"))
		{
			const std::string_view::size_type open_quote = trimmed.find('"');
			const std::string_view::size_type close_quote =
				(open_quote != std::string_view::npos) ? trimmed.find('"', open_quote + 1) : std::string_view::npos;
			if (close_quote == std::string_view::npos)
				continue;

			current_file = trimmed.substr(open_quote + 1, close_quote - open_quote - 1);
			if (first_file.empty())
				first_file = current_file;
		}
		else if (StringUtil::StartsWithNoCase(trimmed, "TRACK") && data_file.empty())
		{
			// "TRACK 01 MODE1/2352" - anything but AUDIO carries the filesystem.
			if (trimmed.find("AUDIO") == std::string_view::npos)
				data_file = current_file;
		}
	}

	const std::string_view chosen = data_file.empty() ? first_file : data_file;
	if (chosen.empty())
	{
		Console.Error(fmt::format("Cue sheet '{}' names no track file.", cue_path));
		return {};
	}

	const std::string path = ResolveSiblingFile(Path::GetDirectory(cue_path), chosen);
	if (!FileSystem::FileExists(path.c_str()))
	{
		Console.Error(fmt::format("Cue sheet '{}' points at '{}', which does not exist.", cue_path, path));
		return {};
	}

	Console.WriteLnFmt("Cue sheet '{}': booting track file '{}'.", cue_path, path);
	return path;
}

// The frontend hands us whatever the user picked; everything except a cue
// sheet goes to the VM as-is.
static std::string ResolveContentPath(const char* path)
{
	if (StringUtil::compareNoCase(Path::GetExtension(path), "cue"))
	{
		std::string resolved = ResolveCueSheet(path);
		if (!resolved.empty())
			return resolved;
	}

	return path;
}

// ---------------------------------------------------------------- disc list
//
// Multi-disc games are shipped as an m3u playlist naming one image per line.
// RetroArch can expand those itself through add_image_index/replace_image_index,
// but only for a core that offers the disk control interface, and a playlist
// handed straight to the core has to work too - so parse it here and let the
// same list serve both.

// One image path per line, relative to the playlist. Blank lines are skipped,
// and so are comments, which is where RetroArch keeps its own #EXTM3U tags.
static std::vector<std::string> ParseM3U(const std::string& m3u_path)
{
	std::vector<std::string> entries;

	const std::optional<std::string> list = FileSystem::ReadFileToString(m3u_path.c_str());
	if (!list.has_value())
	{
		Console.Error(fmt::format("Failed to read playlist '{}'.", m3u_path));
		return entries;
	}

	const std::string_view dir = Path::GetDirectory(m3u_path);
	for (const std::string_view line : StringUtil::SplitString(list.value(), '\n'))
	{
		const std::string_view trimmed = StringUtil::StripWhitespace(line);
		if (trimmed.empty() || trimmed.front() == '#')
			continue;

		std::string path = ResolveSiblingFile(dir, trimmed);
		if (!FileSystem::FileExists(path.c_str()))
		{
			Console.Error(fmt::format("Playlist '{}' names '{}', which does not exist - skipping it.", m3u_path, path));
			continue;
		}

		entries.push_back(std::move(path));
	}

	return entries;
}

// Swaps the disc in a running VM. ChangeDisc() drives the tray state the game
// watches for a media change, so one call per swap is the whole story.
static bool InsertDisc(unsigned index)
{
	if (index >= s_discs.size())
		return false;

	// Nothing is booted yet during retro_load_game - the index picked there is
	// what boots, and there is no VM to swap under.
	if (!s_running.load(std::memory_order_acquire) || !VMManager::HasValidVM())
		return true;

	const std::string path = s_discs[index].boot_path;
	bool changed = false;
	Host::RunOnCPUThread([&changed, &path]() { changed = VMManager::ChangeDisc(CDVD_SourceType::Iso, path); }, true);
	return changed;
}

static bool DiskSetEjectState(bool ejected)
{
	if (s_discs.empty())
		return false;

	if (ejected == s_disc_ejected)
		return true;

	s_disc_ejected = ejected;

	// Closing the tray is where the swap happens. Opening it is left alone on
	// purpose: ChangeDisc() already opens and closes the tray for the guest, so
	// removing the disc here as well would show it two media changes for one.
	return ejected ? true : InsertDisc(s_disc_index);
}

static bool DiskGetEjectState()
{
	return s_disc_ejected;
}

static unsigned DiskGetImageIndex()
{
	return s_disc_index;
}

static bool DiskSetImageIndex(unsigned index)
{
	if (index >= s_discs.size())
		return false;

	s_disc_index = index;

	// The frontend ejects, picks, then closes, and the disc goes in on close.
	// Should it pick with the tray shut, honour that straight away.
	return s_disc_ejected ? true : InsertDisc(index);
}

static unsigned DiskGetNumImages()
{
	return static_cast<unsigned>(s_discs.size());
}

static bool DiskReplaceImageIndex(unsigned index, const struct retro_game_info* info)
{
	if (index >= s_discs.size())
		return false;

	// A null info means "drop this entry", which is how a frontend shortens the
	// list it built with add_image_index.
	if (!info || !info->path)
	{
		s_discs.erase(s_discs.begin() + index);
		if (s_disc_index >= s_discs.size() && !s_discs.empty())
			s_disc_index = static_cast<unsigned>(s_discs.size() - 1);
		return true;
	}

	s_discs[index].path = info->path;
	s_discs[index].boot_path = ResolveContentPath(info->path);
	return true;
}

static bool DiskAddImageIndex()
{
	s_discs.emplace_back();
	return true;
}

// Called before retro_load_game with the disc the frontend last had inserted.
static bool DiskSetInitialImage(unsigned index, const char* path)
{
	s_disc_initial_index = index;
	s_disc_initial_path = path ? path : "";
	return true;
}

static bool CopyOut(std::string_view text, char* s, size_t len)
{
	if (!s || len == 0)
		return false;

	const size_t count = std::min(text.size(), len - 1);
	std::memcpy(s, text.data(), count);
	s[count] = 0;
	return true;
}

static bool DiskGetImagePath(unsigned index, char* s, size_t len)
{
	if (index >= s_discs.size())
		return false;

	return CopyOut(s_discs[index].path, s, len);
}

static bool DiskGetImageLabel(unsigned index, char* s, size_t len)
{
	if (index >= s_discs.size())
		return false;

	// The file name without its extension - "Game (Disc 2)" reads better in the
	// disc menu than the whole path does.
	return CopyOut(Path::GetFileTitle(s_discs[index].path), s, len);
}

static void RegisterDiskControl()
{
	static const struct retro_disk_control_ext_callback ext = {
		&DiskSetEjectState,
		&DiskGetEjectState,
		&DiskGetImageIndex,
		&DiskSetImageIndex,
		&DiskGetNumImages,
		&DiskReplaceImageIndex,
		&DiskAddImageIndex,
		&DiskSetInitialImage,
		&DiskGetImagePath,
		&DiskGetImageLabel,
	};
	if (s_environ_cb(RETRO_ENVIRONMENT_SET_DISK_CONTROL_EXT_INTERFACE, (void*)&ext))
		return;

	// Pre-1.7.6 frontends only know the original interface, which has no labels
	// and cannot restore the disc that was in the drive last time.
	static const struct retro_disk_control_callback legacy = {
		&DiskSetEjectState,
		&DiskGetEjectState,
		&DiskGetImageIndex,
		&DiskSetImageIndex,
		&DiskGetNumImages,
		&DiskReplaceImageIndex,
		&DiskAddImageIndex,
	};
	s_environ_cb(RETRO_ENVIRONMENT_SET_DISK_CONTROL_INTERFACE, (void*)&legacy);
}

// Clears only the libretro-facing disc list state. PCSX2's CDVD/NoDisc state
// remains owned by VMManager and is intentionally not touched here.
static void ResetDiscList()
{
	s_discs.clear();
	s_disc_index = 0;
	s_disc_ejected = false;
}

// Fills the disc list from whatever the frontend loaded and returns the image
// to boot, or an empty string if there is nothing bootable in it.
static std::string BuildDiscList(const char* content_path)
{
	ResetDiscList();

	if (StringUtil::compareNoCase(Path::GetExtension(content_path), "m3u"))
	{
		for (std::string& entry : ParseM3U(content_path))
		{
			DiscImage disc;
			disc.boot_path = ResolveContentPath(entry.c_str());
			disc.path = std::move(entry);
			s_discs.push_back(std::move(disc));
		}

		if (s_discs.empty())
		{
			Console.Error(fmt::format("Playlist '{}' holds no usable image.", content_path));
			return {};
		}

		// Restore the disc the frontend had inserted when it last ran this
		// playlist. The path is checked because the list may have been edited
		// since, in which case the index means something else now.
		if (s_disc_initial_index < s_discs.size() &&
			(s_disc_initial_path.empty() || s_disc_initial_path == s_discs[s_disc_initial_index].path))
		{
			s_disc_index = s_disc_initial_index;
		}

		Console.WriteLnFmt("Playlist '{}': {} disc(s), booting #{} ('{}').", content_path, s_discs.size(),
			s_disc_index + 1, s_discs[s_disc_index].path);
	}
	else
	{
		// Single image, but still offered through the disk control interface:
		// that is what lets a frontend build its own playlist on top of it.
		DiscImage disc;
		disc.path = content_path;
		disc.boot_path = ResolveContentPath(content_path);
		s_discs.push_back(std::move(disc));
	}

	return s_discs[s_disc_index].boot_path;
}

// Defined with the audio output below; declared here because retro_load_game
// registers it before that point in the file.
static void RETRO_CALLCONV AudioBufferStatus(bool active, unsigned occupancy, bool underrun_likely);

bool retro_load_game(const struct retro_game_info* game)
{
	// No content means boot the BIOS. A frontend signals that with a null
	// info pointer, but not every one of them does: some hand over a zeroed
	// retro_game_info instead, and an empty path there means the same thing,
	// not a load to refuse.
	const bool no_content = (game == nullptr) || !game->path || game->path[0] == '\0';

	enum retro_pixel_format fmt = RETRO_PIXEL_FORMAT_XRGB8888;
	if (!s_environ_cb(RETRO_ENVIRONMENT_SET_PIXEL_FORMAT, &fmt))
	{
		Console.Error("XRGB8888 pixel format not supported by frontend.");
		return false;
	}

	// Ask to be told how full the frontend's audio buffer is. Not every
	// frontend answers, and one that does not leaves the core behaving as it
	// did before, which is why OutputAudio only acts on the figure once it has
	// actually been given one.
	{
		struct retro_audio_buffer_status_callback audio_status{AudioBufferStatus};
		if (!s_environ_cb(RETRO_ENVIRONMENT_SET_AUDIO_BUFFER_STATUS_CALLBACK, &audio_status))
			Console.WriteLn("libretro: the frontend does not report audio buffer status");
	}

	if (!InitializeConfig())
		return false;

	// Decide the present path before SettingsOverride (which wires the readback
	// callback only when we're NOT on a HW-render path). Which one we get
	// follows the renderer the user picked in the frontend; the software
	// renderer has nothing on the GPU to hand over, and PCEE2_READBACK=1 forces
	// the readback path for A/B testing.
	{
		retro_variable var = {"pcsx2_renderer", nullptr};
#if defined(ENABLE_VULKAN)
		const char* const fallback_renderer = "vulkan";
#elif defined(ENABLE_OPENGL)
		const char* const fallback_renderer = "opengl";
#else
		const char* const fallback_renderer = "software";
#endif
		const char* const renderer =
			(s_environ_cb(RETRO_ENVIRONMENT_GET_VARIABLE, &var) && var.value) ? var.value : fallback_renderer;
		if (std::getenv("PCEE2_READBACK"))
			s_hw_render = HWRender::None;
		else if (std::strcmp(renderer, "opengl") == 0)
			s_hw_render = HWRender::OpenGL;
		else if (std::strcmp(renderer, "software") != 0)
			s_hw_render = HWRender::Vulkan;
	}

#ifndef ENABLE_OPENGL
	if (s_hw_render == HWRender::OpenGL)
	{
		// The renderer option offers OpenGL only where it was built in, so this
		// means a stale option value from a build that had it.
		Console.Warning("This build has no OpenGL renderer; falling back to readback present.");
		s_hw_render = HWRender::None;
	}
#endif

#ifdef ENABLE_VULKAN
	if (s_hw_render == HWRender::Vulkan)
	{
		static struct retro_hw_render_callback hw_render = {};
		hw_render.context_type = RETRO_HW_CONTEXT_VULKAN;
		hw_render.version_major = 1;
		hw_render.version_minor = 1;
		hw_render.context_reset = OnContextReset;
		hw_render.context_destroy = OnContextDestroy;
		hw_render.cache_context = true;
		if (!s_environ_cb(RETRO_ENVIRONMENT_SET_HW_RENDER, &hw_render))
		{
			Console.Warning("Frontend refused Vulkan HW context; falling back to readback present.");
			s_hw_render = HWRender::None;
		}
		else
		{
			static const struct retro_hw_render_context_negotiation_interface_vulkan neg_iface = {
				RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN,
				RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN_VERSION,
				GetVulkanApplicationInfo,
				CreateVulkanDevice,
				nullptr, // destroy_device
			};
			s_environ_cb(RETRO_ENVIRONMENT_SET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE, (void*)&neg_iface);

			Error vk_error;
			if (!Vulkan::IsVulkanLibraryLoaded() && !Vulkan::LoadVulkanLibrary(&vk_error))
			{
				Console.Error(fmt::format("LoadVulkanLibrary: {}", vk_error.GetDescription()));
				return false;
			}
			VKLibretro::InstallWraps();
			VKLibretro::Active = true;
			s_context_ready.store(false, std::memory_order_release);
		}
	}
#endif

#ifdef ENABLE_OPENGL
	if (s_hw_render == HWRender::OpenGL)
	{
		// A shared context is the whole basis of the GL path: the GS thread
		// needs a context of its own, and the only way to get one that can
		// hand textures to the frontend is for the frontend's context to be
		// shareable in the first place.
		if (!s_environ_cb(RETRO_ENVIRONMENT_SET_HW_SHARED_CONTEXT, nullptr))
			Console.Warning("Frontend does not advertise shared HW contexts; trying anyway.");

		// Desktop GL first, GL ES second: the GS renderer is happier on desktop
		// GL (dual-source blending and clip control are core there, extensions
		// at best on ES), but a mobile or embedded frontend has only ES to
		// offer and refuses the desktop request outright. PCEE2_GLES=1 flips
		// the order, for testing and for drivers whose desktop GL is worse than
		// their ES.
		const bool prefer_gles = (std::getenv("PCEE2_GLES") != nullptr);
		static constexpr struct
		{
			unsigned context_type;
			unsigned major, minor;
			const char* name;
		} kGLContexts[] = {
			{RETRO_HW_CONTEXT_OPENGL_CORE, 3, 3, "OpenGL 3.3 core"},
			{RETRO_HW_CONTEXT_OPENGLES3, 3, 2, "OpenGL ES 3.2"},
		};

		s_gl_hw_render = {};
		s_gl_hw_render.context_reset = OnGLContextReset;
		s_gl_hw_render.context_destroy = OnGLContextDestroy;
		// The GS renders into a texture and retro_run blits it; the frontend's
		// framebuffer never needs a depth or stencil attachment of its own.
		s_gl_hw_render.depth = false;
		s_gl_hw_render.stencil = false;
		s_gl_hw_render.bottom_left_origin = true;
		// Deliberately false, and it matters more than it looks. Asking
		// RetroArch to preserve the context makes it skip context_destroy when
		// it rebuilds its video driver anyway — and context_destroy is the only
		// notice that arrives while the context is still usable. Without it the
		// GS thread's context can never be released (see the comment there),
		// so the picture is stuck until the content is reloaded. Saying the
		// context is expendable gets the callback, and with it a clean rebuild.
		s_gl_hw_render.cache_context = false;

		bool got_context = false;
		for (int i = 0; i < 2 && !got_context; i++)
		{
			const auto& want = kGLContexts[prefer_gles ? (1 - i) : i];
			s_gl_hw_render.context_type = static_cast<retro_hw_context_type>(want.context_type);
			s_gl_hw_render.version_major = want.major;
			s_gl_hw_render.version_minor = want.minor;
			if (s_environ_cb(RETRO_ENVIRONMENT_SET_HW_RENDER, &s_gl_hw_render))
			{
				Console.WriteLnFmt("Frontend gave us a {} context.", want.name);
				got_context = true;
			}
			else
			{
				Console.WarningFmt("Frontend refused a {} context.", want.name);
			}
		}

		if (!got_context)
		{
			Console.Warning("No OpenGL HW context available; falling back to readback present.");
			s_hw_render = HWRender::None;
		}
		else
		{
			GLLibretro::Active = true;
			s_context_ready.store(false, std::memory_order_release);
		}
	}
#endif

	// Tells the present path there is no window behind it, whichever API ends
	// up driving it.
	GSLibretro::Active = HWRenderActive();

	ReadCoreOptions(true);
	SettingsOverride();

	SPU2::CustomOutputStreamFactory = &CreateLibretroAudioStream;
	InputManager::SetPadVibrationCallback(&PadVibrationCallback);
	for (auto& r : s_pad_rumble)
		r.store(0, std::memory_order_relaxed);
	s_environ_cb(RETRO_ENVIRONMENT_GET_RUMBLE_INTERFACE, &s_rumble_interface);

	s_boot_params = VMBootParameters();
	if (no_content)
	{
		ResetDiscList();
	}
	else
	{
		s_boot_params.filename = BuildDiscList(game->path);
		if (s_boot_params.filename.empty())
			return false;
	}

	RegisterDiskControl();

	{
		std::unique_lock lock(s_frame_mutex);
		s_run_token = false;
		s_frame_ready = false;
		s_frame_width = 0;
		s_frame_height = 0;
		s_frame_serial = 0;
		s_frame_serial_seen = 0;
	}
	s_hw_frame_seen = false;

	// Wired for every present path - the readback one included, which also has
	// nothing to hand over for a dropped frame.
	s_duplicate_frames.store(0, std::memory_order_release);
	s_duplicate_frames_spent = 0;
	GSSetDuplicateFrameCallback(&DuplicateFrameCallback);

	s_running.store(true, std::memory_order_release);
	{
		std::unique_lock lock(s_session_mutex);
		s_boot_requested = true;
		s_session_active = true;
	}
	if (!s_cpu_thread.joinable())
		s_cpu_thread = std::thread(CPUThreadMain);
	s_session_cv.notify_all();

	// The frontend calls the negotiation create_device (which opens MTGS) after
	// this returns; it depends on CPUThreadInitialize having run. Wait for it so
	// the two threads don't race on the CPU-thread global setup. (GL has no
	// negotiation step: its context_reset only captures the context, and the
	// CPU thread opens MTGS itself once it sees that happen.)
	if (s_hw_render == HWRender::Vulkan)
	{
		while (!s_cpu_thread_initialized.load(std::memory_order_acquire))
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return true;
}

bool retro_load_game_special(unsigned game_type, const struct retro_game_info* info, size_t num_info)
{
	return false;
}

void retro_unload_game(void)
{
	if (!s_cpu_thread.joinable())
		return;

	// The frontend replays the last set_image indefinitely (menu background,
	// duped frames) — retract it and wait for the GPU before the VM teardown
	// below destroys the textures it points at. Abort pacing first so the GS
	// thread can't stay parked in PublishFrame.
#ifdef ENABLE_VULKAN
	if (s_hw_render == HWRender::Vulkan)
	{
		VKLibretro::AbortPacing();
		if (auto* vulkan = static_cast<retro_hw_render_interface_vulkan*>(VKLibretro::GetHWRenderInterface()))
		{
			vulkan->set_image(vulkan->handle, nullptr, 0, nullptr, vulkan->queue_index);
			vulkan->wait_sync_index(vulkan->handle);
		}
	}
#endif

#ifdef ENABLE_OPENGL
	// GL hands over a texture retro_run copies out of, so there is nothing for
	// the frontend to keep replaying — just stop the handoff before the VM
	// teardown deletes the textures the last published frame points at.
	else if (s_hw_render == HWRender::OpenGL)
	{
		GLLibretro::AbortPacing();
	}
#endif

	s_running.store(false, std::memory_order_release);
	if (VMManager::HasValidVM())
		VMManager::SetState(VMState::Stopping);

	// release the CPU thread if it's blocked waiting for a run token
	{
		std::unique_lock lock(s_frame_mutex);
		s_run_token = true;
		s_frame_cv.notify_all();
	}

	// wait for the session to wind down; the CPU thread itself stays alive
	// for the next retro_load_game
	{
		std::unique_lock lock(s_session_mutex);
		s_session_cv.wait(lock, []() { return !s_session_active; });
	}

	s_audio_stream = nullptr;
	GSSetDuplicateFrameCallback(nullptr);
	if (!HWRenderActive())
		GSSetFramebufferReadback(nullptr, 0, 0);

#ifdef ENABLE_VULKAN
	if (s_hw_render == HWRender::Vulkan)
	{
		VKLibretro::Shutdown();
		VKLibretro::Active = false;
	}
#endif

#ifdef ENABLE_OPENGL
	if (s_hw_render == HWRender::OpenGL)
	{
		GLLibretro::Shutdown();
		GLLibretro::Active = false;
		s_gl_context_seen = false;
		s_gl_present_fbo = 0;
	}
#endif
	if (HWRenderActive())
	{
		s_hw_render = HWRender::None;
		s_context_ready.store(false, std::memory_order_release);
	}
	GSLibretro::Active = false;
	s_hw_geom_width = 0;
	s_hw_geom_height = 0;
	s_memory_map_sent = false;
}

void retro_reset(void)
{
	if (!s_running.load(std::memory_order_acquire))
		return;

	// VMManager::Reset() only *requests* a reset from here: called while the
	// VM is Running (always, since this can only run against a live session),
	// it just flags VMState::Resetting and returns immediately - see its own
	// comment ("we tell the rec to exit execution, _then_ reset"). The actual
	// reset (hwReset(), SPU2::Reset(), ...) only happens once something calls
	// VMManager::Reset() again while state == Resetting; every other PCSX2
	// frontend's main loop has a case for that (see QtHost's `case
	// VMState::Resetting: VMManager::Reset(); continue;`) and drives the real
	// reset from there. CPUThreadMain() below does the same now - that's
	// also where s_audio_stream_mutex actually needs to be held (see its
	// declaration, above OutputAudio()), not here: locking around *this*
	// call only serializes the flag-set against its own caller, and the real
	// teardown/rebuild of the audio stream happens later, back in
	// CPUThreadMain(), on the CPU thread's own schedule.
	//
	// block=true still matters on its own merits (matches ChangeDiscIndex):
	// it stops retro_reset() from returning - and a second reset or a
	// save-state op from being accepted - before the request has even been
	// queued.
	Host::RunOnCPUThread([]() { VMManager::Reset(); }, true);
}

// Translate libretro joypad/analog state into DualShock2 binds. Called at the
// start of retro_run(), while the CPU thread is parked, so Pad state writes
// don't race the SIO reads.
static void UpdateInput()
{
	if (!s_input_poll_cb || !s_input_state_cb)
		return;

	// don't touch Pad state until the VM is fully up (first frame produced);
	// during VMManager::Initialize() the CPU thread is still constructing it
	if (s_frame_width == 0 && !s_hw_frame_seen)
		return;

	s_input_poll_cb();

	static constexpr std::pair<unsigned, u32> button_map[] = {
		{RETRO_DEVICE_ID_JOYPAD_UP, PadDualshock2::Inputs::PAD_UP},
		{RETRO_DEVICE_ID_JOYPAD_RIGHT, PadDualshock2::Inputs::PAD_RIGHT},
		{RETRO_DEVICE_ID_JOYPAD_DOWN, PadDualshock2::Inputs::PAD_DOWN},
		{RETRO_DEVICE_ID_JOYPAD_LEFT, PadDualshock2::Inputs::PAD_LEFT},
		{RETRO_DEVICE_ID_JOYPAD_X, PadDualshock2::Inputs::PAD_TRIANGLE},
		{RETRO_DEVICE_ID_JOYPAD_A, PadDualshock2::Inputs::PAD_CIRCLE},
		{RETRO_DEVICE_ID_JOYPAD_B, PadDualshock2::Inputs::PAD_CROSS},
		{RETRO_DEVICE_ID_JOYPAD_Y, PadDualshock2::Inputs::PAD_SQUARE},
		{RETRO_DEVICE_ID_JOYPAD_SELECT, PadDualshock2::Inputs::PAD_SELECT},
		{RETRO_DEVICE_ID_JOYPAD_START, PadDualshock2::Inputs::PAD_START},
		{RETRO_DEVICE_ID_JOYPAD_L, PadDualshock2::Inputs::PAD_L1},
		{RETRO_DEVICE_ID_JOYPAD_L2, PadDualshock2::Inputs::PAD_L2},
		{RETRO_DEVICE_ID_JOYPAD_R, PadDualshock2::Inputs::PAD_R1},
		{RETRO_DEVICE_ID_JOYPAD_R2, PadDualshock2::Inputs::PAD_R2},
		{RETRO_DEVICE_ID_JOYPAD_L3, PadDualshock2::Inputs::PAD_L3},
		{RETRO_DEVICE_ID_JOYPAD_R3, PadDualshock2::Inputs::PAD_R3},
	};

	for (u32 port = 0; port < static_cast<u32>(s_pad_map.size()); port++)
	{
		const u32 pad = s_pad_map[port];

		for (const auto& [retro_id, ds2_bind] : button_map)
		{
			const int16_t state = s_input_state_cb(port, RETRO_DEVICE_JOYPAD, 0, retro_id);
			Pad::SetControllerState(pad, ds2_bind, state ? 1.0f : 0.0f);
		}

		// analog sticks: split each axis into the two directional binds
		static constexpr auto axis_value = [](int16_t v, bool negative) {
			const float f = static_cast<float>(v) / 32767.0f;
			return negative ? std::max(-f, 0.0f) : std::max(f, 0.0f);
		};

		const int16_t lx = s_input_state_cb(port, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_X);
		const int16_t ly = s_input_state_cb(port, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_Y);
		const int16_t rx = s_input_state_cb(port, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X);
		const int16_t ry = s_input_state_cb(port, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y);

		Pad::SetControllerState(pad, PadDualshock2::Inputs::PAD_L_LEFT, axis_value(lx, true));
		Pad::SetControllerState(pad, PadDualshock2::Inputs::PAD_L_RIGHT, axis_value(lx, false));
		Pad::SetControllerState(pad, PadDualshock2::Inputs::PAD_L_UP, axis_value(ly, true));
		Pad::SetControllerState(pad, PadDualshock2::Inputs::PAD_L_DOWN, axis_value(ly, false));
		Pad::SetControllerState(pad, PadDualshock2::Inputs::PAD_R_LEFT, axis_value(rx, true));
		Pad::SetControllerState(pad, PadDualshock2::Inputs::PAD_R_RIGHT, axis_value(rx, false));
		Pad::SetControllerState(pad, PadDualshock2::Inputs::PAD_R_UP, axis_value(ry, true));
		Pad::SetControllerState(pad, PadDualshock2::Inputs::PAD_R_DOWN, axis_value(ry, false));
	}

	// GunCon2 lightguns (bind indices from usb-lightgun/guncon2.cpp)
	for (u32 usb_port = 0; usb_port < 2; usb_port++)
	{
		if (!(s_lightgun_mask & (1u << usb_port)))
			continue;

		const auto gun = [&](unsigned id) {
			return s_input_state_cb(usb_port, RETRO_DEVICE_LIGHTGUN, 0, id);
		};

		// aim: [-0x8000,0x7fff] across the visible video -> relative half-axes
		const float x = std::clamp(static_cast<float>(gun(RETRO_DEVICE_ID_LIGHTGUN_SCREEN_X)) / 32767.0f, -1.0f, 1.0f);
		const float y = std::clamp(static_cast<float>(gun(RETRO_DEVICE_ID_LIGHTGUN_SCREEN_Y)) / 32767.0f, -1.0f, 1.0f);
		USB::SetDeviceBindValue(usb_port, 18 /* BID_RELATIVE_LEFT  */, (x < 0.0f) ? -x : 0.0f);
		USB::SetDeviceBindValue(usb_port, 19 /* BID_RELATIVE_RIGHT */, (x > 0.0f) ? x : 0.0f);
		USB::SetDeviceBindValue(usb_port, 20 /* BID_RELATIVE_UP    */, (y < 0.0f) ? -y : 0.0f);
		USB::SetDeviceBindValue(usb_port, 21 /* BID_RELATIVE_DOWN  */, (y > 0.0f) ? y : 0.0f);

		static constexpr std::pair<unsigned, u32> gun_buttons[] = {
			{RETRO_DEVICE_ID_LIGHTGUN_TRIGGER, 13 /* BID_TRIGGER */},
			{RETRO_DEVICE_ID_LIGHTGUN_RELOAD, 16 /* BID_SHOOT_OFFSCREEN */},
			{RETRO_DEVICE_ID_LIGHTGUN_AUX_A, 3 /* BID_A */},
			{RETRO_DEVICE_ID_LIGHTGUN_AUX_B, 2 /* BID_B */},
			{RETRO_DEVICE_ID_LIGHTGUN_AUX_C, 1 /* BID_C */},
			{RETRO_DEVICE_ID_LIGHTGUN_START, 15 /* BID_START */},
			{RETRO_DEVICE_ID_LIGHTGUN_SELECT, 14 /* BID_SELECT */},
			{RETRO_DEVICE_ID_LIGHTGUN_DPAD_UP, 4 /* BID_DPAD_UP */},
			{RETRO_DEVICE_ID_LIGHTGUN_DPAD_DOWN, 6 /* BID_DPAD_DOWN */},
			{RETRO_DEVICE_ID_LIGHTGUN_DPAD_LEFT, 7 /* BID_DPAD_LEFT */},
			{RETRO_DEVICE_ID_LIGHTGUN_DPAD_RIGHT, 5 /* BID_DPAD_RIGHT */},
		};
		for (const auto& [retro_id, bid] : gun_buttons)
			USB::SetDeviceBindValue(usb_port, bid, gun(retro_id) ? 1.0f : 0.0f);
	}
}

// pad_when_empty keeps the frontend's pipeline fed while nothing is being
// produced yet; the pacing loop in retro_run() drains without it, since a
// silent block there would be inserted into a stream that is running fine.
static void RETRO_CALLCONV AudioBufferStatus(bool active, unsigned occupancy, bool underrun_likely)
{
	s_audio_status_known.store(active, std::memory_order_release);
	s_audio_occupancy.store(occupancy, std::memory_order_release);
}

static void OutputAudio(bool pad_when_empty = true)
{
	// Leave the frontend alone while its buffer is nearly full: what it does
	// with an over-full one is block, and on some drivers block for good.
	if (s_audio_status_known.load(std::memory_order_acquire) &&
		s_audio_occupancy.load(std::memory_order_acquire) >= 90)
		return;

	if (!s_audio_batch_cb)
		return;

	static float float_buffer[MAX_AUDIO_FRAMES_PER_RUN * 2];
	static int16_t s16_buffer[MAX_AUDIO_FRAMES_PER_RUN * 2];

	u32 frames = 0;
	{
		std::unique_lock lock(s_audio_stream_mutex);
		if (s_audio_stream)
			frames = s_audio_stream->PullFrames(float_buffer, MAX_AUDIO_FRAMES_PER_RUN);
	}

	if (frames == 0)
	{
		if (!pad_when_empty)
			return;

		// keep the frontend's audio pipeline fed during boot
		std::memset(s16_buffer, 0, (SAMPLE_RATE / 60) * 2 * sizeof(int16_t));
		s_audio_batch_cb(s16_buffer, SAMPLE_RATE / 60);
		return;
	}

	for (u32 i = 0; i < frames * 2; i++)
	{
		const float v = std::clamp(float_buffer[i], -1.0f, 1.0f);
		s16_buffer[i] = static_cast<int16_t>(v * 32767.0f);
	}
	s_audio_batch_cb(s16_buffer, frames);
	s_audio_frames_output.fetch_add(frames, std::memory_order_relaxed);
}

// Re-announce av_info whenever the VM's timing or our output size changes
// (PAL 50Hz detection after boot, PSX-mode 44.1kHz, upscale option, ...).
static void UpdateAVInfoIfChanged()
{
	// Initialized to what retro_get_system_av_info reported at startup, so the
	// first VM report only announces when it genuinely differs.
	static u32 last_fps_bits = 0; // 0 = "still the 59.94f startup default"
	static u32 last_sample_rate = SAMPLE_RATE;
	static u32 last_width = 0;
	static u32 last_height = 0;
	static u32 last_aspect_bits = 0;

	const u32 fps_bits = s_vm_fps_bits.load(std::memory_order_acquire);
	const u32 sample_rate = s_audio_sample_rate.load(std::memory_order_acquire);
	const u32 width = s_out_width.load(std::memory_order_acquire);
	const u32 height = s_out_height.load(std::memory_order_acquire);
	const u32 aspect_bits = s_aspect_bits.load(std::memory_order_acquire);

	// SET_SYSTEM_AV_INFO makes the frontend reinit the whole video driver
	// (context_destroy + re-negotiation on the HW-render path) — only worth it
	// for a real timing change. In HW-render mode geometry/aspect are handled
	// per-frame via SET_GEOMETRY (no reinit), so ignore them here. The fps
	// compare needs a tolerance: NTSC reports 59.94005994Hz vs the 59.94f
	// startup default, and that 0.00006Hz delta must not trigger a reinit.
	const float fps = fps_bits ? std::bit_cast<float>(fps_bits) : 0.0f;
	const float last_fps = last_fps_bits ? std::bit_cast<float>(last_fps_bits) : 59.94f;
	const bool timing_changed = std::abs(fps - last_fps) > 0.25f || sample_rate != last_sample_rate;
	const bool geometry_changed = width != last_width || height != last_height || aspect_bits != last_aspect_bits;
	if (!timing_changed && (HWRenderActive() || !geometry_changed))
		return;

	// don't announce anything until the VM has reported a real frame rate
	if (fps_bits == 0)
		return;

	last_fps_bits = fps_bits;
	last_sample_rate = sample_rate;
	last_width = width;
	last_height = height;
	last_aspect_bits = aspect_bits;

	// The frontend's video reinit runs inside this environment call while the
	// GS thread may still be chewing queued work that submits to the shared
	// queue — drain it first so the reinit doesn't race those submits. (The CPU
	// thread is parked here: retro_run hasn't posted its run token.)
	if (HWRenderActive() && MTGS::IsOpen())
	{
		std::atomic_bool gs_drained{false};
		MTGS::RunOnGSThread([&gs_drained]() { gs_drained.store(true, std::memory_order_release); });
		for (int i = 0; i < 1000 && !gs_drained.load(std::memory_order_acquire); i++)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	retro_system_av_info av_info;
	retro_get_system_av_info(&av_info);
	s_environ_cb(RETRO_ENVIRONMENT_SET_SYSTEM_AV_INFO, &av_info);
	INFO_LOG("libretro av_info: {}x{} @ {:.2f}Hz, {}Hz audio", av_info.geometry.base_width,
		av_info.geometry.base_height, av_info.timing.fps, sample_rate);
}

// Whether a frame the frontend has not seen yet is waiting to be presented.
// Call with s_frame_mutex held (the readback path publishes under it).
static bool PresentFramePending()
{
#ifdef ENABLE_VULKAN
	if (s_hw_render == HWRender::Vulkan)
		return VKLibretro::HasFrame();
#endif
#ifdef ENABLE_OPENGL
	if (s_hw_render == HWRender::OpenGL)
		return GLLibretro::HasFrame();
#endif
	return s_frame_serial != s_frame_serial_seen;
}

// What the frontend is told about the picture, as opposed to what it is handed.
//
// The canvas handed over is the merged frame at the size the GS drew it, so it
// moves with the internal-resolution setting and with every field/frame merge -
// 640x448 one frame and 640x224 the next is ordinary for an interlaced game.
// Geometry is not the place for that. A frontend recomputes its viewport from
// the geometry, and one that is scaling by integers has to pick a different
// multiple every time the base moves, which is a picture that grows and shrinks
// by a whole step several times a second (issue #33).
//
// So the geometry carries the native resolution, which is stable across both,
// and the per-frame canvas size goes where it belongs: in the video callback,
// which is what tells the frontend how much of the render target to sample.
static void SyncHWRenderGeometry(u32 canvas_width, u32 canvas_height)
{
	const u64 native = GSLibretro::NativeSize.load(std::memory_order_acquire);
	const u32 width = native ? static_cast<u32>(native >> 32) : canvas_width;
	const u32 height = native ? static_cast<u32>(native & 0xffffffffu) : canvas_height;

	// The GS's aspect, not this file's reading of the option, so the two cannot
	// disagree about 480p or a widescreen patch.
	const float gs_aspect = GSLibretro::DisplayAspect.load(std::memory_order_acquire);
	const u32 aspect_bits = s_aspect_bits.load(std::memory_order_acquire);
	const float aspect = gs_aspect > 0.0f ?
		gs_aspect :
		(aspect_bits ? std::bit_cast<float>(aspect_bits) : (4.0f / 3.0f));

	if (width == s_hw_geom_width && height == s_hw_geom_height && aspect == s_hw_geom_aspect)
		return;

	s_hw_geom_width = width;
	s_hw_geom_height = height;
	s_hw_geom_aspect = aspect;

	retro_game_geometry geometry = {};
	geometry.base_width = width;
	geometry.base_height = height;
	geometry.max_width = GSLibretro::kMaxCanvasWidth;
	geometry.max_height = GSLibretro::kMaxCanvasHeight;
	geometry.aspect_ratio = aspect;
	s_environ_cb(RETRO_ENVIRONMENT_SET_GEOMETRY, &geometry);
}

void retro_run(void)
{
	// apply core option changes on the fly
	bool options_updated = false;
	if (s_environ_cb(RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE, &options_updated) && options_updated &&
		s_running.load(std::memory_order_acquire))
	{
		ReadCoreOptions(false);
		Host::RunOnCPUThread([]() {
			VMManager::SetLimiterMode(s_internal_limiter.load(std::memory_order_acquire) ?
				LimiterModeType::Nominal : LimiterModeType::Unlimited);
			VMManager::ApplySettings();
		});
	}

	UpdateAVInfoIfChanged();

	// announce the memory map once the VM has its memory allocated, so the
	// frontend's achievements/cheats can read EE RAM
	if (!s_memory_map_sent && eeMem && s_running.load(std::memory_order_acquire))
	{
		static retro_memory_descriptor descs[2];
		descs[0] = {RETRO_MEMDESC_SYSTEM_RAM, eeMem->Main, 0, 0x00000000u, 0, 0, Ps2MemSize::MainRam, "EE RAM"};
		descs[1] = {RETRO_MEMDESC_SYSTEM_RAM, eeMem->Scratch, 0, 0x70000000u, 0, 0, sizeof(eeMem->Scratch), "Scratchpad"};
		retro_memory_map mmap = {descs, 2};
		s_memory_map_sent = s_environ_cb(RETRO_ENVIRONMENT_SET_MEMORY_MAPS, &mmap);
	}

	if (s_running.load(std::memory_order_acquire))
		UpdateInput();

	// forward DS2 vibration to the frontend
	if (s_rumble_interface.set_rumble_state)
	{
		for (u32 port = 0; port < static_cast<u32>(s_pad_map.size()); port++)
		{
			const u32 packed = s_rumble_enabled ? s_pad_rumble[s_pad_map[port]].load(std::memory_order_relaxed) : 0;
			s_rumble_interface.set_rumble_state(port, RETRO_RUMBLE_STRONG, static_cast<u16>(packed >> 16));
			s_rumble_interface.set_rumble_state(port, RETRO_RUMBLE_WEAK, static_cast<u16>(packed & 0xFFFF));
		}
	}

	if (!s_running.load(std::memory_order_acquire))
	{
		// VM is gone (failed boot or shutdown); present black
		static std::vector<u32> black(DEFAULT_WIDTH * DEFAULT_HEIGHT, 0);
		if (s_video_cb)
			s_video_cb(black.data(), DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_WIDTH * sizeof(u32));
		return;
	}

#ifdef ENABLE_OPENGL
	// The frontend replaced its GL context (see OnGLContextReset): every object
	// the GS device holds belongs to a share group that no longer exists, so
	// rebuild it against the new one. Deliberately here rather than in
	// context_reset — by now the frontend is out of EGL, so the GS thread can
	// swap contexts without blocking against it — and before the run token
	// below, so the CPU thread is still parked and this is the only thing
	// writing to the GS ring.
	if (s_gl_needs_reopen.load(std::memory_order_acquire))
	{
		s_gl_needs_reopen.store(false, std::memory_order_release);
		if (MTGS::IsOpen())
		{
			// -1 still running, 0 the GS thread cannot take a new context, 1 done.
			std::atomic<int> result{-1};
			MTGS::RunOnGSThread([&result]() {
				// Normally already done from context_destroy — but RetroArch
				// never calls it for a GL core when it rebuilds its video
				// driver (the same going-behind-the-core's-back the Vulkan path
				// works around by re-fetching its interface every frame), so
				// this is usually the first the GS thread hears of it, with the
				// context already dead.
				auto* dev = (g_gs_device && g_gs_device->GetRenderAPI() == RenderAPI::OpenGL) ?
					static_cast<GSDeviceOGL*>(g_gs_device.get()) : nullptr;
				if (dev && !dev->AbandonContext(false))
				{
					result.store(0, std::memory_order_release);
					return;
				}

				GSreopen(true, false, GSConfig.Renderer, std::nullopt);
				result.store(1, std::memory_order_release);
			});

			for (int i = 0; i < 5000 && result.load(std::memory_order_acquire) < 0; i++)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));

			if (result.load(std::memory_order_acquire) == 0)
			{
				// Not a failure we can retry: this GS thread can never be given
				// another context, and building one anyway parks it inside the
				// driver for good, which would also hang the frontend the next
				// time it unloads the content. Leave the picture where it is
				// and say what fixes it.
				Console.Error("GL: The frontend replaced its context without announcing it, and the "
							  "old one cannot be released. Reload the content to get the picture back.");
				Host::AddKeyedOSDMessage("GLContextLost",
					TRANSLATE_STR("GS", "The video output was reset. Reload the content to resume, or "
									   "switch to the Vulkan renderer."),
					Host::OSD_CRITICAL_ERROR_DURATION);
			}
			else if (result.load(std::memory_order_acquire) < 0)
			{
				Console.Error("GL: The GS device did not come back; the picture will stay black.");
			}
		}
	}
#endif

	// Hand the CPU thread a frame of execution and wait for the result.
	//
	// When the GS drops a frame as a duplicate there is no image to hand over
	// and the frontend is told to repeat the last one, so its frame rate stays
	// at the VM's vertical frequency while the game's is a fraction of it -
	// which is what frame generation and interpolation filters trip over.
	// Spend the drop on another frame of emulation instead: retro_run() then
	// returns once per frame that carries a new image, and the cadence the
	// frontend sees becomes the game's internal one. Announcing the lower rate
	// through av_info would be the other way to do it, but that reset tears the
	// video driver down mid-run.
	//
	// Only drops the GS has actually reported are spent, at most as many in a
	// row as the GS itself will drop. When it is merely a frame behind, no
	// credit is waiting and the frontend gets the dupe it would have got
	// before - guessing from "nothing arrived" instead would run the VM ahead
	// every time the GPU is the slow end. A frame already waiting is presented
	// as it is: the HW paths park the GS thread in PublishFrame until
	// retro_run consumes, so running on past an unconsumed frame would stall
	// this loop against its own pacing.
	//
	// What holds the speed to 100% is the frontend blocking on the audio it is
	// handed, the same as it already is for this core - nothing here limits by
	// wall clock, and SettingsOverride() turns the VM's own limiter off.
	std::unique_lock lock(s_frame_mutex);
	bool got_frame = false;
	for (u32 run = 0;; run++)
	{
		s_run_token = true;
		s_frame_cv.notify_all();

		got_frame = s_frame_cv.wait_for(lock, std::chrono::milliseconds(200), []() { return s_frame_ready; });
		s_frame_ready = false;

		if (!got_frame || run >= MAX_SKIPPED_DUPLICATE_FRAMES || PresentFramePending())
			break;

		const u64 dropped = s_duplicate_frames.load(std::memory_order_acquire);
		if (dropped <= s_duplicate_frames_spent)
			break;

		s_duplicate_frames_spent++;

		// Hand this frame's audio over before running the next one: the stream
		// keeps ~50ms of ring and four frames of a 50Hz game are 80ms, so
		// draining once at the end would overrun it. It is also what keeps the
		// extra frames from running the VM fast - the frontend's write blocks
		// for as long as the audio it just took needs to play. The CPU thread
		// stays parked meanwhile; it has no run token until the next pass.
		lock.unlock();
		OutputAudio(false);
		lock.lock();
	}

#ifdef ENABLE_VULKAN
	if (s_hw_render == HWRender::Vulkan)
	{
		// Zero-copy present: consume the frame the GS just published and hand
		// its VkImage to the frontend. The retro_vulkan_image storage must
		// outlive this call (the frontend replays it for cached/duped frames).
		VKLibretro::Frame frame;

		// Ask the frontend for the interface every frame rather than trusting
		// the pointer context_reset handed over. RetroArch rebuilds its video
		// driver behind the core's back - toggling fullscreen (its F binding)
		// does exactly that - and frees the retro_hw_render_interface_vulkan it
		// gave us without calling context_destroy or context_reset, so the
		// cached pointer becomes freed memory and calling set_image through it
		// jumps into whatever now lives there.
		retro_hw_render_interface* hw_iface = nullptr;
		if (!s_environ_cb(RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE, &hw_iface) || !hw_iface ||
			hw_iface->interface_type != RETRO_HW_RENDER_INTERFACE_VULKAN)
		{
			hw_iface = nullptr;
		}
		auto* vulkan = reinterpret_cast<retro_hw_render_interface_vulkan*>(hw_iface);

		// Keep the GS thread's copy in step: it takes the frontend's queue lock
		// through this same interface on every submit.
		if (vulkan != VKLibretro::GetHWRenderInterface())
			VKLibretro::SetHWRenderInterface(vulkan);

		if (vulkan && vulkan->set_image && VKLibretro::ConsumeFrame(&frame))
		{
			s_hw_frame_seen = true;
			s_hw_last_width = frame.width;
			s_hw_last_height = frame.height;
			SyncHWRenderGeometry(frame.width, frame.height);
			static retro_vulkan_image vkimage;
			vkimage = {};
			vkimage.image_view = frame.view;
			vkimage.image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			vkimage.create_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, frame.image,
				VK_IMAGE_VIEW_TYPE_2D, frame.format,
				{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
					VK_COMPONENT_SWIZZLE_IDENTITY},
				{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
			vulkan->set_image(vulkan->handle, &vkimage, 0, nullptr, vulkan->queue_index);
			if (s_video_cb)
				s_video_cb(RETRO_HW_FRAME_BUFFER_VALID, frame.width, frame.height, 0);
		}
		else if (s_video_cb)
		{
			// Nothing new (still booting, or a duplicate frame) - dupe, at the
			// size the last real frame was handed over at. s_frame_width belongs
			// to the readback path and is never set here, so the old fallback
			// announced 640x480 in between upscaled frames, and a frontend that
			// scales by integers recomputed its viewport on every one of them:
			// a picture that grows and shrinks by a whole step several times a
			// second (issue #33).
			s_video_cb(nullptr, s_hw_last_width ? s_hw_last_width : DEFAULT_WIDTH,
				s_hw_last_height ? s_hw_last_height : DEFAULT_HEIGHT, 0);
		}
	}
#endif

#ifdef ENABLE_OPENGL
	if (s_hw_render == HWRender::OpenGL)
	{
		// The frontend's context is current on this thread and the GS thread's
		// context shares its objects, so the texture the GS just published can
		// be read straight into the frontend's framebuffer.
		GLLibretro::Frame frame;
		if (s_gl_hw_render.get_current_framebuffer && GLLibretro::ConsumeFrame(&frame))
		{
			s_hw_frame_seen = true;
			s_hw_last_width = frame.width;
			s_hw_last_height = frame.height;
			SyncHWRenderGeometry(frame.width, frame.height);

			// Sharing objects does not share ordering: the fence is what says
			// the GS thread's rendering has actually landed, rather than just
			// been queued. Waiting on the server side costs this thread
			// nothing — the blit below is what ends up waiting.
			if (frame.fence)
			{
				glWaitSync(frame.fence, 0, GL_TIMEOUT_IGNORED);
				glDeleteSync(frame.fence);
			}

			if (s_gl_present_fbo == 0)
				glGenFramebuffers(1, &s_gl_present_fbo);

			const GLuint target_fbo = static_cast<GLuint>(s_gl_hw_render.get_current_framebuffer());
			glBindFramebuffer(GL_READ_FRAMEBUFFER, s_gl_present_fbo);
			glFramebufferTexture2D(
				GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frame.texture, 0);
			glReadBuffer(GL_COLOR_ATTACHMENT0);
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_fbo);

			// A blit is masked by both of these, and this context is shared
			// with the GS thread's — whatever either side last set is still
			// set, so say what this one needs rather than assuming.
			glDisable(GL_SCISSOR_TEST);
			glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
			glBlitFramebuffer(0, 0, frame.width, frame.height, 0, 0, frame.width, frame.height,
				GL_COLOR_BUFFER_BIT, GL_NEAREST);

			// hw_render.bottom_left_origin is set, so the frontend reads the
			// lower-left frame.width x frame.height of its framebuffer — which
			// is exactly where a GL blit to (0,0) lands.
			glBindFramebuffer(GL_READ_FRAMEBUFFER, target_fbo);
			if (s_video_cb)
				s_video_cb(RETRO_HW_FRAME_BUFFER_VALID, frame.width, frame.height, 0);
		}
		else if (s_video_cb)
		{
			// Nothing new (still booting, or a duplicate frame) - dupe, at the
			// size the last real frame was handed over at. s_frame_width belongs
			// to the readback path and is never set here, so the old fallback
			// announced 640x480 in between upscaled frames, and a frontend that
			// scales by integers recomputed its viewport on every one of them:
			// a picture that grows and shrinks by a whole step several times a
			// second (issue #33).
			s_video_cb(nullptr, s_hw_last_width ? s_hw_last_width : DEFAULT_WIDTH,
				s_hw_last_height ? s_hw_last_height : DEFAULT_HEIGHT, 0);
		}
	}
#endif
	// Readback path. Written as its own condition rather than an else on the blocks
	// above, which compile out one by one as the renderers are switched off.
	if (!HWRenderActive())
	{
		if (got_frame && s_frame_width > 0 && s_frame_height > 0 && s_video_cb)
		{
			s_frame_serial_seen = s_frame_serial;
			s_video_cb(s_frame_pixels.data(), s_frame_width, s_frame_height, s_frame_width * sizeof(u32));
		}
		else if (s_video_cb)
		{
			// no frame produced yet (still booting); duplicate previous frame
			s_video_cb(nullptr, s_frame_width ? s_frame_width : DEFAULT_WIDTH,
				s_frame_height ? s_frame_height : DEFAULT_HEIGHT,
				(s_frame_width ? s_frame_width : DEFAULT_WIDTH) * sizeof(u32));
		}
	}

	// CPU thread is parked again at this point; safe to drain the audio buffer
	OutputAudio();
}

// Fixed upper bound for uncompressed PS2 state data (EE 32MB + IOP 2MB + GS +
// SPU2 + VU + zip overhead). libretro requires a stable serialize size, while
// the real state size varies per frame, so we pad up to this and store the
// actual length in a small header.
static constexpr size_t SERIALIZE_BUFFER_SIZE = 96 * 1024 * 1024;
static constexpr u32 SERIALIZE_MAGIC = 0x50325253; // 'P2RS'

struct SerializeHeader
{
	u32 magic;
	u32 reserved;
	u64 zip_size;
};

size_t retro_serialize_size(void)
{
	return SERIALIZE_BUFFER_SIZE;
}

bool retro_serialize(void* data, size_t size)
{
	if (!s_running.load(std::memory_order_acquire) || size < sizeof(SerializeHeader))
		return false;

	bool result = false;
	Host::RunOnCPUThread(
		[data, size, &result]() {
			if (VMManager::GetState() != VMState::Running && VMManager::GetState() != VMState::Paused)
				return;

			Error error;
			std::unique_ptr<ArchiveEntryList> entries = SaveState_DownloadState(&error);
			if (!entries)
			{
				ERROR_LOG("retro_serialize: DownloadState failed: {}", error.GetDescription());
				return;
			}

			std::vector<u8> buffer;
			if (!SaveState_ZipToBuffer(std::move(entries), &buffer, &error))
			{
				ERROR_LOG("retro_serialize: ZipToBuffer failed: {}", error.GetDescription());
				return;
			}

			if (sizeof(SerializeHeader) + buffer.size() > size)
			{
				ERROR_LOG("retro_serialize: state too large ({} bytes)", buffer.size());
				return;
			}

			SerializeHeader header = {SERIALIZE_MAGIC, 0, buffer.size()};
			std::memcpy(data, &header, sizeof(header));
			std::memcpy(static_cast<u8*>(data) + sizeof(header), buffer.data(), buffer.size());
			result = true;
		},
		true);

	return result;
}

bool retro_unserialize(const void* data, size_t size)
{
	if (!s_running.load(std::memory_order_acquire) || size < sizeof(SerializeHeader))
		return false;

	SerializeHeader header;
	std::memcpy(&header, data, sizeof(header));
	if (header.magic != SERIALIZE_MAGIC || sizeof(SerializeHeader) + header.zip_size > size)
		return false;

	bool result = false;
	Host::RunOnCPUThread(
		[data, &header, &result]() {
			if (VMManager::GetState() != VMState::Running && VMManager::GetState() != VMState::Paused)
				return;

			Error error;
			if (!SaveState_UnzipFromBuffer(
					static_cast<const u8*>(data) + sizeof(SerializeHeader), header.zip_size, &error))
			{
				ERROR_LOG("retro_unserialize: {}", error.GetDescription());
				return;
			}

			result = true;
		},
		true);

	return result;
}

void retro_cheat_reset(void) {}
void retro_cheat_set(unsigned index, bool enabled, const char* code) {}

unsigned retro_get_region(void)
{
	const u32 fps_bits = s_vm_fps_bits.load(std::memory_order_acquire);
	return (fps_bits && std::bit_cast<float>(fps_bits) < 55.0f) ? RETRO_REGION_PAL : RETRO_REGION_NTSC;
}

void retro_set_controller_port_device(unsigned port, unsigned device) {}

// EE main memory, exposed so the frontend's achievements, cheats and memory
// inspection work. eeMem is allocated during CPU thread startup; until then
// (or after shutdown) report no memory.
void* retro_get_memory_data(unsigned id)
{
	if (id == RETRO_MEMORY_SYSTEM_RAM && eeMem && s_running.load(std::memory_order_acquire))
		return eeMem->Main;
	return nullptr;
}

size_t retro_get_memory_size(unsigned id)
{
	if (id == RETRO_MEMORY_SYSTEM_RAM)
		return Ps2MemSize::MainRam;
	return 0;
}

//////////////////////////////////////////////////////////////////////////
// Host implementation
//////////////////////////////////////////////////////////////////////////

void Host::CommitBaseSettingChanges()
{
	// in-memory settings; nothing to persist
}

void Host::LoadSettings(SettingsInterface& si, std::unique_lock<std::mutex>& lock)
{
}

void Host::CheckForSettingsChanges(const Pcsx2Config& old_config)
{
}

bool Host::RequestResetSettings(bool folders, bool core, bool controllers, bool hotkeys, bool ui)
{
	return false;
}

void Host::SetDefaultUISettings(SettingsInterface& si)
{
}

bool Host::LocaleCircleConfirm()
{
	return false;
}

std::unique_ptr<ProgressCallback> Host::CreateHostProgressCallback()
{
	return ProgressCallback::CreateNullProgressCallback();
}

void Host::ReportInfoAsync(const std::string_view title, const std::string_view message)
{
	if (!title.empty() && !message.empty())
		INFO_LOG("ReportInfoAsync: {}: {}", title, message);
	else if (!message.empty())
		INFO_LOG("ReportInfoAsync: {}", message);
}

void Host::ReportErrorAsync(const std::string_view title, const std::string_view message)
{
	if (!title.empty() && !message.empty())
		ERROR_LOG("ReportErrorAsync: {}: {}", title, message);
	else if (!message.empty())
		ERROR_LOG("ReportErrorAsync: {}", message);
}

void Host::OpenURL(const std::string_view url)
{
}

bool Host::CopyTextToClipboard(const std::string_view text)
{
	return false;
}

std::string Host::GetTextFromClipboard()
{
	// No host clipboard access from within the libretro core.
	return std::string();
}

void Host::BeginTextInput()
{
}

void Host::EndTextInput()
{
}

std::optional<WindowInfo> Host::GetTopLevelWindowInfo()
{
	WindowInfo wi;
	wi.type = WindowInfo::Type::Surfaceless;
	wi.surface_width = DEFAULT_WIDTH;
	wi.surface_height = DEFAULT_HEIGHT;
	wi.surface_scale = 1.0f;
	return wi;
}

void Host::OnInputDeviceConnected(const std::string_view identifier, const std::string_view device_name)
{
}

void Host::OnInputDeviceDisconnected(const InputBindingKey key, const std::string_view identifier)
{
}

void Host::SetMouseMode(bool relative_mode, bool hide_cursor)
{
}

void Host::SetMouseLock(bool state)
{
}

std::optional<WindowInfo> Host::AcquireRenderWindow(bool recreate_window)
{
	// v1: surfaceless; the SW renderer draws into memory and we read it back
	WindowInfo wi;
	wi.type = WindowInfo::Type::Surfaceless;
	wi.surface_width = DEFAULT_WIDTH;
	wi.surface_height = DEFAULT_HEIGHT;
	wi.surface_scale = 1.0f;
	return wi;
}

void Host::ReleaseRenderWindow()
{
}

void Host::BeginPresentFrame()
{
}

void Host::RequestResizeHostDisplay(s32 width, s32 height)
{
}

void Host::OnVMStarting()
{
}

void Host::OnVMStarted()
{
}

void Host::OnVMDestroyed()
{
}

void Host::OnVMPaused()
{
}

void Host::OnVMResumed()
{
}

void Host::OnGameChanged(const std::string& title, const std::string& elf_override, const std::string& disc_path,
	const std::string& disc_serial, u32 disc_crc, u32 current_crc)
{
	INFO_LOG("Game changed: {} ({})", title, disc_serial);
}

void Host::OnPerformanceMetricsUpdated()
{
}

void Host::OnSaveStateLoading(const std::string_view filename)
{
}

void Host::OnSaveStateLoaded(const std::string_view filename, bool was_successful)
{
}

void Host::OnSaveStateSaved(const std::string_view filename)
{
}

void Host::RunOnCPUThread(std::function<void()> function, bool block /* = false */)
{
	if (std::this_thread::get_id() == s_cpu_thread_id)
	{
		function();
		return;
	}

	if (!block)
	{
		{
			std::unique_lock lock(s_cpu_work_mutex);
			s_cpu_work.push_back(std::move(function));
			s_cpu_work_pending.store(true, std::memory_order_release);
		}
		s_frame_cv.notify_all(); // wake the CPU thread if it's parked
		return;
	}

	// blocking variant: wait for the CPU thread to drain the queue
	std::mutex done_mutex;
	std::condition_variable done_cv;
	bool done = false;
	{
		std::unique_lock lock(s_cpu_work_mutex);
		s_cpu_work.push_back([&]() {
			function();
			std::unique_lock dlock(done_mutex);
			done = true;
			done_cv.notify_all();
		});
		s_cpu_work_pending.store(true, std::memory_order_release);
	}
	s_frame_cv.notify_all();
	std::unique_lock dlock(done_mutex);
	done_cv.wait(dlock, [&]() { return done; });
}

void Host::RunOnGSThread(std::function<void()> function)
{
	MTGS::RunOnGSThread(std::move(function));
}

void Host::RefreshGameListAsync(bool invalidate_cache)
{
}

void Host::CancelGameListRefresh()
{
}

bool Host::IsFullscreen()
{
	return false;
}

void Host::SetFullscreen(bool enabled)
{
}

void Host::OnCaptureStarted(const std::string& filename)
{
}

void Host::OnCaptureStopped()
{
}

void Host::RequestExitApplication(bool allow_confirm)
{
	VMManager::SetState(VMState::Stopping);
}

void Host::RequestExitBigPicture()
{
}

void Host::RequestVMShutdown(bool allow_confirm, bool allow_save_state, bool default_save_state)
{
	VMManager::SetState(VMState::Stopping);
}

void Host::OnAchievementsLoginSuccess(const char* username, u32 points, u32 sc_points, u32 unread_messages)
{
}

void Host::OnAchievementsLoginRequested(Achievements::LoginRequestReason reason)
{
}

void Host::OnAchievementsHardcoreModeChanged(bool enabled)
{
}

void Host::OnAchievementsRefreshed()
{
}

void Host::OnCoverDownloaderOpenRequested()
{
}

void Host::OnCreateMemoryCardOpenRequested()
{
}

bool Host::InBatchMode()
{
	return true;
}

bool Host::InNoGUIMode()
{
	return true;
}

bool Host::ShouldPreferHostFileSelector()
{
	return false;
}

void Host::OpenHostFileSelectorAsync(std::string_view title, bool select_directory, FileSelectorCallback callback,
	FileSelectorFilters filters, std::string_view initial_directory)
{
	callback(std::string());
}

int Host::LocaleSensitiveCompare(std::string_view lhs, std::string_view rhs)
{
	const int res = std::strncmp(lhs.data(), rhs.data(), std::min(lhs.size(), rhs.size()));
	if (res != 0)
		return res;
	return lhs.size() > rhs.size() ? 1 : (lhs.size() < rhs.size() ? -1 : 0);
}

void Host::PumpMessagesOnCPUThread()
{
	// run deferred work first (settings changes, reset requests, ...)
	DrainCPUWork();

	if (!s_running.load(std::memory_order_acquire))
		return;

	// track the VM's vertical frequency for PAL/NTSC av_info reporting
	const float fps = VMManager::GetFrameRate();
	if (fps > 0.0f)
		s_vm_fps_bits.store(std::bit_cast<u32>(fps), std::memory_order_release);

	// frames arrive asynchronously via FramebufferReadbackCallback on the GS
	// thread; nothing to read back here
	static u32 s_frame_counter = 0;
	if ((s_frame_counter++ % 120) == 0)
	{
		u32 width, height;
		{
			std::unique_lock lock(s_frame_mutex);
			width = s_frame_width;
			height = s_frame_height;
		}
		INFO_LOG("libretro frame {}: {}x{} audio_frames={}", s_frame_counter - 1, width, height,
			s_audio_frames_output.load(std::memory_order_relaxed));
	}

	// pacing: signal frame done, then wait for the next run token, servicing
	// queued CPU work (savestates, resets, ...) while parked
	std::unique_lock lock(s_frame_mutex);
	s_frame_ready = true;
	s_frame_cv.notify_all();
	for (;;)
	{
		s_frame_cv.wait(lock, []() {
			return s_run_token || s_cpu_work_pending.load(std::memory_order_acquire) ||
				   !s_running.load(std::memory_order_acquire);
		});

		if (s_cpu_work_pending.load(std::memory_order_acquire))
		{
			lock.unlock();
			DrainCPUWork();
			lock.lock();
			continue;
		}

		break;
	}
	s_run_token = false;
}

s32 Host::Internal::GetTranslatedStringImpl(
	const std::string_view context, const std::string_view msg, char* tbuf, size_t tbuf_space)
{
	if (msg.size() > tbuf_space)
		return -1;
	else if (msg.empty())
		return 0;

	std::memcpy(tbuf, msg.data(), msg.size());
	return static_cast<s32>(msg.size());
}

std::string Host::TranslatePluralToString(const char* context, const char* msg, const char* disambiguation, int count)
{
	TinyString count_str = TinyString::from_format("{}", count);

	std::string ret(msg);
	for (;;)
	{
		std::string::size_type pos = ret.find("%n");
		if (pos == std::string::npos)
			break;

		ret.replace(pos, pos + 2, count_str.view());
	}

	return ret;
}

std::optional<u32> InputManager::ConvertHostKeyboardStringToCode(const std::string_view str)
{
	return std::nullopt;
}

std::optional<std::string> InputManager::ConvertHostKeyboardCodeToString(u32 code)
{
	return std::nullopt;
}

const char* InputManager::ConvertHostKeyboardCodeToIcon(u32 code)
{
	return nullptr;
}

BEGIN_HOTKEY_LIST(g_host_hotkeys)
END_HOTKEY_LIST()
