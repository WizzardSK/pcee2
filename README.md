# PCEE2 — PCSX2 libretro core

![Libretro Core Builds](https://img.shields.io/github/actions/workflow/status/WizzardSK/pcee2-libretro/libretro_builds.yml?branch=libretro&label=core%20builds)

A [libretro](https://www.libretro.com/) core frontend for the current PCSX2
codebase, letting RetroArch (and other libretro frontends) run PS2 games with
an up-to-date emulation core.

Unlike [LRPS2](https://github.com/libretro/ps2) — a hard fork of an older
PCSX2 snapshot — this port keeps the libretro layer *additive*: the emulation
core is tracked from [upstream PCSX2](https://github.com/PCSX2/pcsx2) with a
minimal set of hooks, so rebasing onto new upstream releases stays cheap.

The core reports the upstream PCSX2 version it is built from, so the version
RetroArch shows names the standalone PCSX2 the emulation code corresponds to —
currently **v2.7.523**. See [Upstream sync and versioning](#upstream-sync-and-versioning).

This project is not affiliated with or endorsed by the PCSX2 team.

## Status

| Area | Status |
|---|---|
| Boot + video (Vulkan, surfaceless) | ✅ working |
| Zero-copy Vulkan output (libretro HW render) | ✅ working, default on Vulkan |
| Software renderer (presented via Vulkan) | ✅ working |
| Audio (48 kHz / 44.1 kHz PSX mode) | ✅ working |
| Pad input (DualShock 2, 2 ports, analogs) | ✅ working |
| Savestates (`retro_serialize`) | ✅ working, deterministic |
| Memory cards | ✅ working (`<system>/pcsx2/memcards`) |
| Core options (renderer, resolution, BIOS, fast boot) | ✅ working |
| PAL (50 Hz) / NTSC av_info | ✅ working |
| Fast-forward | ✅ working |
| OpenGL renderer | ✅ working (surfaceless EGL) |
| D3D11 / D3D12 renderers | ⚠️ in the Windows build, untested |
| Metal renderer | ⚠️ in the macOS build, untested (Vulkan via MoltenVK is the default) |
| RetroAchievements | ✅ via RetroArch (EE RAM exposed; log in to RetroAchievements in RetroArch settings) |
| Multitap (up to 8 controllers) | ✅ core option |
| Lightgun (GunCon 2 via USB) | ✅ core option, aimed by frontend lightgun/mouse |
| Other USB devices (wheels, mic, EyeToy) | ❌ not wired up |
| Start Core without content | ✅ opens the PS2 BIOS / System Menu and Memory Card Browser |
| Content reload / Close Content | ✅ core survives RetroArch's deinit/init cycles |
| Windows x64 build (MSVC, via CI) | ✅ community-tested (WRC 4, GTA SA, Killzone on Vulkan) |
| macOS x86_64 build (via CI) | ⚠️ compiles + links, untested — feedback welcome |

On the Vulkan renderer the core shares the frontend's `VkDevice` through
libretro context negotiation and hands over the rendered image directly, with
no GPU readback in the way. Everything else (OpenGL, software, D3D, Metal, or a
frontend that refuses HW render) falls back to the per-frame readback path,
which is double-buffered on the GS thread and costs one frame of latency;
`PCEE2_READBACK=1` forces it on Vulkan too, for A/B testing.

## Download

Through RetroArch: **Online Updater → Core Downloader → PCEE2**. The libretro
buildbot builds this branch nightly for Windows x64, Linux x64 and aarch64,
macOS x64 and arm64, and Android arm64-v8a and x86_64, so what it hands out is
always the current tree. There are no releases here on purpose: another place
to download from is another place to be out of date.

macOS needs the quarantine flag cleared once:
`xattr -cr ~/Library/Application\ Support/RetroArch/cores`.

Per-commit builds are available as artifacts of the
[Libretro Core Builds](https://github.com/WizzardSK/pcee2-libretro/actions/workflows/libretro_builds.yml)
workflow — that is also where to get a build of a branch, or one for a target
the buildbot does not cover.

## Setup

1. Put a PS2 BIOS dump into `<retroarch system dir>/pcsx2/bios/`.
2. Optional: copy the `resources` directory from a PCSX2 installation of the same version (or from `bin/resources` of this repo) to `<retroarch system dir>/pcsx2/resources/`. The shaders and fonts the core cannot start without, and the `GameIndex.yaml` game database that carries the per-game fixes, are all compiled into it, so this only adds the remaining optional files; `PCEE2_EXTERNAL_RESOURCES=1` makes the on-disk copies win over the built-in ones, which is how you use a newer game database than the one the core was built with.
3. For built-in game patches (including the widescreen / no-interlacing options), download [`patches.zip`](https://github.com/PCSX2/pcsx2_patches/releases/latest/download/patches.zip) into `<retroarch system dir>/pcsx2/resources/patches.zip`.
4. Start the core without content to open the PS2 BIOS / System Menu, or load a disc image (`.iso`, `.chd`, `.cso`, `.gz`, `.bin`, `.mdf`, `.nrg`, `.elf`) with the core.
5. Optional: copy your standalone `PCSX2.ini` to `<retroarch system dir>/pcsx2/inis/PCSX2.ini` — the core adopts its emulation settings (EmuCore, speed hacks, CPU, GS, game fixes, memory cards, DEV9 network/HDD) as the baseline. Core options still override their respective settings.

Memory cards, savestates metadata, cache, etc. live under
`<retroarch system dir>/pcsx2/`.

## Building (Linux)

```sh
# distro packages (Ubuntu/KDE neon)
sudo apt install -y cmake ninja-build clang liblz4-dev libwebp-dev \
  libcurl4-openssl-dev libpcap-dev libfontconfig-dev libudev-dev \
  libx11-dev libxrandr-dev extra-cmake-modules libwayland-dev libegl-dev libdbus-1-dev

# deps not packaged usably by distros (SDL3, plutovg, plutosvg, rapidyaml,
# libbacktrace, shaderc — Ubuntu's libshaderc_combined.a is not self-contained):
bash pcee2-libretro/scripts/build-deps-linux.sh "$PWD/deps"

cmake -B build-libretro -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DENABLE_QT_UI=OFF -DENABLE_TESTS=OFF -DENABLE_LIBRETRO=ON \
  -DCMAKE_PREFIX_PATH=$PWD/deps \
  -DSHADERC_STATIC=ON -DSHADERC_LIBRARY=$PWD/deps/lib/libshaderc_combined.a
ninja -C build-libretro pcee2-libretro
# -> build-libretro/bin/pcee2_libretro.so
```

Windows (MSVC) and macOS use the matching static dependency scripts in
`pcee2-libretro/scripts/` — see `.github/workflows/libretro_builds.yml` for
the exact steps on all three platforms.

## Core options

| Option | Values | Notes |
|---|---|---|
| BIOS | auto / discovered images | restart required |
| Fast Boot | enabled / disabled | restart required |
| Memory Cards | Slot 1/2 enabled / disabled + discovered `.ps2` cards | applies immediately; scanned from `<system>/pcsx2/memcards` |
| Renderer | Vulkan / OpenGL / Software | applies on the fly |
| Internal Resolution | 1x–4x | applies on the fly, scales output too |
| Blending Accuracy | Minimum–Maximum | default Basic |
| Texture Filtering | Nearest / Bilinear (PS2/Forced/Forced-no-sprites) | default PS2 |
| Trilinear Filtering | Auto / Off / PS2 / Forced | |
| Anisotropic Filtering | Off–16x | |
| Dithering | Off / Scaled / Unscaled | default Unscaled |
| Hardware Mipmapping | enabled / disabled | |
| Deinterlacing | Automatic + 8 manual modes | default GameDB-driven |
| FXAA | enabled / disabled | |
| CAS (sharpening) | Disabled / Sharpen Only + sharpness 10–100 | |
| Aspect Ratio | Auto / 4:3 / 16:9 | Auto follows widescreen patches |
| Widescreen Patches | enabled / disabled | built-in 16:9 patches |
| No-Interlacing Patches | enabled / disabled | built-in progressive patches |
| EE Cycle Rate | 50%–300% | speed hack, may break games |
| EE Cycle Skip | Disabled–Maximum | speed hack, may break games |
| Multitap | Disabled / Port 1 / Port 2 / Both | up to 8 players, restart recommended |
| Lightgun (GunCon 2) | Disabled / USB 1 / USB 2 / Both | aim via frontend lightgun input, restart required |
| Rumble | enabled / disabled | DS2 vibration via frontend rumble |
| Analog Axis Scale | 100–150% | default 133% (DualShock 2 response) |
| Analog Deadzone | 0–30% | inside the emulated pad |

The Memory Cards category exposes the two standard PS2 slots and lists only
existing `.ps2` files found when the core options are registered. If the
directory is empty, the Card selectors are omitted for that run; PCSX2 keeps
its normal `Mcd001.ps2` / `Mcd002.ps2` defaults and creates missing cards
through its existing memory-card path. Changing Enabled or Card selection while
content is running is applied immediately through PCSX2's native memory-card
configuration path; no core or content restart is required. The candidate list
is still captured when the core options are registered.

All graphics options map directly onto the corresponding standalone PCSX2
settings; anything not exposed yet runs at the standalone default (including
automatic per-game fixes from the GameDB).

## Architecture notes

- The frontend (`pcee2-libretro/Libretro.cpp`) is modeled on `pcsx2-gsrunner`:
  a dedicated CPU thread runs the `VMManager::Execute()` loop, and
  `Host::PumpMessagesOnCPUThread()` paces it 1:1 against `retro_run()`.
- Frames leave the GS thread through `VKLibretro`
  (`pcsx2/GS/Renderers/Vulkan/VKLibretro.cpp`), which publishes the rendered
  `VkImage` to `retro_run()` for `set_image`; the fallback path uses
  `GSSetFramebufferReadback()`, a double-buffered readback on the GS thread
  added for this port (`GSRenderer.cpp`).
- Audio is pulled from a custom `AudioStream` registered through
  `SPU2::CustomOutputStreamFactory`.
- Savestates use `SaveState_ZipToBuffer`/`SaveState_UnzipFromBuffer`
  (in-memory variants of the existing zip paths).
- The CPU thread and PCSX2's process-level state (signal handlers, JIT
  memory) live for the whole core lifetime and are reused across
  RetroArch's deinit/init content cycles, mirroring the Qt frontend's
  lifetime model.
- Core modifications beyond these hooks are intentionally minimal; see
  `git log --oneline pcsx2up/master..libretro -- pcsx2/ common/` for the
  full delta (`pcsx2up` = the PCSX2 remote, see below).

## Upstream sync and versioning

`pcee2-libretro/upstream.version` records the upstream PCSX2 release the
emulation code is merged up to. The build reads it and reports that version as
the core version — in RetroArch's core info, in the Vulkan application version
and in the version stamped into savestates — instead of describing pcee2's own
git tags, which say nothing about which PCSX2 is inside. `bin/resources` from
the same PCSX2 version is the matching set.

Syncing to a newer PCSX2:

```sh
git remote add pcsx2up https://github.com/PCSX2/pcsx2.git   # once
git fetch pcsx2up master --tags
git merge pcsx2up/master
# then bump PCSX2_VERSION / PCSX2_COMMIT in pcee2-libretro/upstream.version
# to `git describe --tags pcsx2up/master` and its commit
```

CMake warns when the recorded commit is not an ancestor of the tree being
built, so a forgotten bump shows up in the build log.

## License

GPL-3.0+, same as PCSX2. All emulation code is the work of the
[PCSX2 team and contributors](https://github.com/PCSX2/pcsx2/graphs/contributors).
