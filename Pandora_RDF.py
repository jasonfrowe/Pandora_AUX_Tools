# %%
import numpy as np
import importlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandora_funcs

pandora = importlib.reload(pandora_funcs)

try:
	from IPython import get_ipython
	ip = get_ipython()
	if ip is not None:
		ip.run_line_magic('load_ext', 'autoreload')
		ip.run_line_magic('autoreload', '2')
except Exception:
	pass
# %%
datadir = '/opt/data2/rowe/pandora/2026/RDF1/data/'
science_file = 'Pandora_RDF_WASP-178b_all.fits'
fits_path = datadir + science_file

# Tunable bad-pixel/clump-repair settings.
badpix_params = {
	"min_intercept_offset_dn": 250.0,
	"intercept_sigma": 10.0,
	"slope_percentile": 0.5,
	"scatter_sigma": 10.0,
	"core_dilate_iterations": 0,
	# Tune this to control how far we repair residual bright wings around bad cores.
	"wing_iterations": 1,
	"repair_max_radius": 3,
	"repair_min_neighbors": 4,
}

# Tunable trace extraction priors.
trace_params = {
	"dispersion_min": 75,
	"dispersion_max": 282,
	"expected_spatial_center": 40,
	"max_half_width": 30,
	"smooth_window": 5,
	"threshold_sigma": 2.5,
	# Manual variable aperture bounds from visual tuning.
	"spatial_left_start": 37.0,
	"spatial_left_end": 33.0,
	"spatial_right_start": 43.0,
	"spatial_right_end": 41.0,
}
# %%
# RAW SCIENCE is stored as (integration, group, row, column).
ramp_cube, science_header = pandora.read_rdf_raw_science(fits_path)
cube = pandora.flatten_ramp_cube(ramp_cube)
nint, ngroup = ramp_cube.shape[0], ramp_cube.shape[1]

row_map, col_map, exposure_time_s = pandora.get_rdf_auxiliary_data(fits_path)

print(f"Loaded RAW SCIENCE ramp with shape: {ramp_cube.shape} (nint={nint}, ngroup={ngroup})")
print(f"Flattened science cube shape: {cube.shape}")
print(f"SCIENCE EXTNAME: {science_header.get('EXTNAME', 'N/A')}")

if exposure_time_s is not None:
	print(
		f"Loaded EXPOSURE_TIME array: {exposure_time_s.shape}, "
		f"min={np.nanmin(exposure_time_s):.6g}s, max={np.nanmax(exposure_time_s):.6g}s"
	)

# %%
# Step 1: inspect a raw frame.
pandora.display_science_image(cube, image_index=20, scale_style="zscale")

# %%
# Step 2: load VISDA reference products from the local pandora-ref submodule.
ref_products = pandora.load_visda_reference_products()
print(f"Reference products loaded from: {ref_products['paths']['visda_dir']}")

# %%
# Step 3: apply first-pass calibration corrections.
# We apply: bias_0d, bias_2d, stripes, dark (using EXPOSURE_TIME), flat, and bad pixel mask.
corrected_cube, correction_info = pandora.apply_visda_reference_corrections(
	cube,
	row_map=row_map,
	col_map=col_map,
	ref_products=ref_products,
	exposure_time_s=exposure_time_s,
	apply_bias_0d=True,
	apply_bias_2d=True,
	apply_stripes=True,
	apply_dark=True,
	apply_flat=True,
	apply_badpix_mask=True,
	convert_to_electrons=False,
)

print("Applied correction steps:")
for step in correction_info["steps"]:
	print(f" - {step}")

# %%
# Step 4: apply up-the-ramp zero-point correction in (integration, group) space.
corrected_ramp = pandora.unflatten_ramp_cube(corrected_cube, nint=nint, ngroup=ngroup)

fit_products_pre = pandora.compute_ramp_fit_products_r2s(
	corrected_ramp,
	sigcut=2.0,
)

raw_slope_cube = fit_products_pre["slope"]
print(f"Pre-CR slope cube shape (integration, row, column): {raw_slope_cube.shape}")

zp_corrected_ramp, zp_info = pandora.apply_scalar_zero_point_from_ramp_fit(
	corrected_ramp,
	intercept_cube=fit_products_pre["intercept"],
)

print(
	f"Ramp-fit scalar zero-point drift range=({zp_info['drift_min']:.3f}, "
	f"{zp_info['drift_max']:.3f}) DN"
)

hot_pixel_mask, hot_info = pandora.detect_hot_pixels_from_ramp_fit(
	fit_products_pre["intercept"],
	scatter_cube=fit_products_pre["scatter"],
	intercept_sigma=8.0,
	min_intercept_offset_dn=150.0,
	scatter_sigma=8.0,
)

print(
	f"Hot pixels from ramp zero-point map: {hot_info['n_hot']} "
	f"({100.0 * hot_info['hot_fraction']:.4f}%)"
)

bad_clump_mask, bad_clump_info = pandora.detect_bad_pixel_clumps_from_ramp_fit(
	fit_products_pre["intercept"],
	fit_products_pre["slope"],
	scatter_cube=fit_products_pre["scatter"],
	min_intercept_offset_dn=badpix_params["min_intercept_offset_dn"],
	intercept_sigma=badpix_params["intercept_sigma"],
	slope_percentile=badpix_params["slope_percentile"],
	scatter_sigma=badpix_params["scatter_sigma"],
	dilate_iterations=badpix_params["core_dilate_iterations"],
)

print(
	f"Bad clump pixels from ramp diagnostics: {bad_clump_info['n_clump_pixels']} "
	f"({100.0 * bad_clump_info['clump_fraction']:.4f}%)"
)

# Expand clump cores by one pixel to capture bright residual wings around defects.
repair_mask = pandora.dilate_binary_mask(
	bad_clump_mask,
	iterations=badpix_params["wing_iterations"],
)
wing_only_mask = repair_mask & (~bad_clump_mask)
print(
	f"Expanded repair mask pixels (core+wings): {int(np.sum(repair_mask))} "
	f"({100.0 * np.mean(repair_mask):.4f}%)"
)

# Build user-facing pixel masks/maps for QA and downstream use.
corrected_pixel_mask = repair_mask.copy()
pixel_status_map = np.zeros(repair_mask.shape, dtype=np.uint8)
hot_only_mask = hot_pixel_mask & (~corrected_pixel_mask)
pixel_status_map[hot_only_mask] = 1
pixel_status_map[bad_clump_mask] = 2
pixel_status_map[wing_only_mask] = 3

print(
	"Pixel status counts [0=good, 1=hot-only, 2=core, 3=wing]:",
	{int(k): int(v) for k, v in zip(*np.unique(pixel_status_map, return_counts=True))},
)
print(
	f"Corrected pixel mask size: {int(np.sum(corrected_pixel_mask))} "
	f"({100.0 * np.mean(corrected_pixel_mask):.4f}%)"
)

# Quick-look map so corrected pixels are easy to inspect.
fig, ax = plt.subplots(figsize=(8, 6))
cmap = ListedColormap(["black", "gold", "red", "cyan"])
im = ax.imshow(np.rot90(pixel_status_map), origin="lower", cmap=cmap, vmin=0, vmax=3, aspect="auto")
ax.set_title("Pixel Status Map (Core/Wing Classification)")
ax.set_xlabel("Dispersion pixel")
ax.set_ylabel("Spatial pixel")
cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
cb.ax.set_yticklabels(["good", "hot-only", "core", "wing"])
plt.tight_layout()
plt.show()

# %%
# Step 5: apply up-the-ramp cosmic ray correction in (integration, group) space.

cr_corrected_ramp, cr_info = pandora.apply_ramp_cosmic_ray_correction(
	zp_corrected_ramp,
	sigma_threshold=8.0,
	min_jump_dn=50.0,
	positive_jumps_only=True,
)

print(
	f"Cosmic ray jump events corrected: {cr_info['n_events']} "
	f"({100.0 * cr_info['fraction_diffs_flagged']:.4f}% of group diffs)"
)

# Refit the corrected ramps to produce integration-level slope images.
fit_products_post = pandora.compute_ramp_fit_products_r2s(
	cr_corrected_ramp,
	sigcut=2.0,
)

slope_cube = fit_products_post["slope"]
print(f"Post-CR slope cube shape (integration, row, column): {slope_cube.shape}")

slope_cube_repaired, repair_info = pandora.correct_bad_pixels_with_neighbors(
	slope_cube,
	bad_mask=repair_mask,
	max_radius=badpix_params["repair_max_radius"],
	min_neighbors=badpix_params["repair_min_neighbors"],
)

print(
	f"Neighbor-repaired bad pixels in slope cube: {repair_info['n_fixed']} "
	f"({100.0 * repair_info['fixed_fraction']:.3f}% of masked samples)"
)

# %%
# Step 6: apply row/column destriping to integration-level slope images.
# Build masks to protect the spectral trace while estimating background trends.
# For this dataset: dispersion is roughly rows 75-300, spatial trace is near column 40.
source_mask = np.zeros((slope_cube.shape[1], slope_cube.shape[2]), dtype=bool)
disp_lo, disp_hi = 75, 300
spat_center, spat_half_width = 40, 6

row_lo = max(0, disp_lo)
row_hi = min(source_mask.shape[0], disp_hi + 1)
col_lo = max(0, spat_center - spat_half_width)
col_hi = min(source_mask.shape[1], spat_center + spat_half_width + 1)
source_mask[row_lo:row_hi, col_lo:col_hi] = True

# Column offsets should be estimated from off-spectrum rows only to avoid
# over-subtracting the trace and creating side shadows.
column_mask = np.zeros_like(source_mask)
column_mask[row_lo:row_hi, :] = True

# Keep ramp-identified hot pixels out of row/column background estimation.
row_mask = source_mask | hot_pixel_mask
column_mask = column_mask | hot_pixel_mask
row_mask = row_mask | repair_mask
column_mask = column_mask | repair_mask

rc_corrected_cube, rc_info = pandora.apply_row_column_correction(
	slope_cube_repaired,
	apply_row=True,
	apply_column=True,
	row_mask=row_mask,
	column_mask=column_mask,
	preserve_frame_median=True,
)

print("Applied row/column correction steps:")
for step in rc_info["steps"]:
	print(f" - {step}")

# %%
# Step 7: compare pre- and post-correction slope images quantitatively.
image_index = min(22, nint - 1)
raw_med = float(np.nanmedian(raw_slope_cube[image_index]))
corr_med = float(np.nanmedian(rc_corrected_cube[image_index]))
delta_med = corr_med - raw_med

print(f"Raw slope image median={raw_med:.6g}")
print(f"Final corrected slope image median={corr_med:.6g}")
print(f"Median delta={delta_med:.6g} DN/group ({100.0 * delta_med / raw_med:.4f}%)")

# %%
# Step 8: visualize raw vs final corrected slope images with residuals.
pandora.display_correction_comparison(
	raw_slope_cube,
	rc_corrected_cube,
	image_index=image_index,
	scale_style="zscale",
    iraf_contrast=0.1
)

# %%
# Step 9: estimate a global spatial profile, then define a manual variable aperture.
trace_est = pandora.estimate_trace_aperture(
	rc_corrected_cube,
	dispersion_min=trace_params["dispersion_min"],
	dispersion_max=trace_params["dispersion_max"],
	expected_spatial_center=trace_params["expected_spatial_center"],
	max_half_width=trace_params["max_half_width"],
	smooth_window=trace_params["smooth_window"],
	threshold_sigma=trace_params["threshold_sigma"],
)

aperture_model = pandora.build_linear_trace_aperture(
	n_dispersion=rc_corrected_cube.shape[1],
	dispersion_min=trace_params["dispersion_min"],
	dispersion_max=trace_params["dispersion_max"],
	spatial_left_start=trace_params["spatial_left_start"],
	spatial_left_end=trace_params["spatial_left_end"],
	spatial_right_start=trace_params["spatial_right_start"],
	spatial_right_end=trace_params["spatial_right_end"],
	n_spatial=rc_corrected_cube.shape[2],
)

print(
	"Variable aperture bounds: "
	f"left {trace_params['spatial_left_start']:.1f}->{trace_params['spatial_left_end']:.1f}, "
	f"right {trace_params['spatial_right_start']:.1f}->{trace_params['spatial_right_end']:.1f}"
)
print(f"Detected peak positions (global double-hump diagnostic): {trace_est['peak_positions']}")

# Plot the spatial profile and estimated aperture bounds.
spat_x = np.arange(trace_est["profile_smooth"].size)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(spat_x, trace_est["profile"], alpha=0.5, label="Raw profile")
ax.plot(spat_x, trace_est["profile_smooth"], lw=2, label="Smoothed profile")
ax.axvline(trace_params["spatial_left_start"], color="r", ls="--", label="Start bounds")
ax.axvline(trace_params["spatial_right_start"], color="r", ls="--")
ax.axvline(trace_params["spatial_left_end"], color="orange", ls=":", label="End bounds")
ax.axvline(trace_params["spatial_right_end"], color="orange", ls=":")
ax.axhline(trace_est["threshold"], color="k", ls=":", label="Threshold")
for p in trace_est["peak_positions"]:
	ax.axvline(p, color="g", ls="-.", alpha=0.8)
ax.set_xlabel("Spatial pixel")
ax.set_ylabel("Collapsed slope signal")
ax.set_title("Global Spatial Profile and Variable Aperture Priors")
ax.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.show()

# Overlay aperture on the corrected slope image to validate trace placement.
disp_min = int(aperture_model["dispersion_pixels"][0])
disp_max = int(aperture_model["dispersion_pixels"][-1])

img = rc_corrected_cube[image_index]
finite = np.isfinite(img)
if np.any(finite):
	vmin = float(np.nanpercentile(img[finite], 5))
	vmax = float(np.nanpercentile(img[finite], 99.5))
else:
	vmin, vmax = 0.0, 1.0

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(
	img,
	origin="lower",
	aspect="auto",
	vmin=vmin,
	vmax=vmax,
	cmap="viridis",
)

# Variable aperture bounds in spatial direction.
ax.plot(
	aperture_model["spatial_left"],
	aperture_model["dispersion_pixels"],
	color="r",
	lw=1.8,
	ls="--",
	label="Variable aperture",
)
ax.plot(
	aperture_model["spatial_right"],
	aperture_model["dispersion_pixels"],
	color="r",
	lw=1.8,
	ls="--",
)

# Dispersion range bounds used for extraction.
ax.plot(
	[aperture_model["spatial_left"][0], aperture_model["spatial_right"][0]],
	[disp_min, disp_min],
	color="w",
	lw=1.0,
	ls=":",
	label="Extraction range",
)
ax.plot(
	[aperture_model["spatial_left"][-1], aperture_model["spatial_right"][-1]],
	[disp_max, disp_max],
	color="w",
	lw=1.0,
	ls=":",
)

# Double-hump peak markers from profile estimate.
for p in trace_est["peak_positions"]:
	ax.plot(p, 0.5 * (disp_min + disp_max), marker="x", color="cyan", ms=9, mew=2)

ax.set_xlim(0, img.shape[1] - 1)
ax.set_ylim(0, img.shape[0] - 1)
ax.set_xlabel("Spatial pixel")
ax.set_ylabel("Dispersion pixel")
ax.set_title(f"Trace/Aperture Overlay on Corrected Slope Image (integration {image_index})")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# %%
# Step 10: perform spectro-photometric extraction and diagnostic plots.
extracted_spectra, extracted_dispersion = pandora.extract_trace_spectra_variable_aperture(
	rc_corrected_cube,
	aperture_model=aperture_model,
)

if exposure_time_s is not None and exposure_time_s.size == nint * ngroup:
	trial_time_axis = exposure_time_s.reshape(nint, ngroup)[:, -1]
	if np.nanmax(trial_time_axis) > np.nanmin(trial_time_axis):
		integration_time_axis = trial_time_axis
		time_axis_label = "Ramp end time [s]"
	else:
		integration_time_axis = np.arange(nint, dtype=float)
		time_axis_label = "Integration index"
else:
	integration_time_axis = np.arange(nint, dtype=float)
	time_axis_label = "Integration index"

white_light = np.nansum(extracted_spectra, axis=1)
white_light_norm = white_light / np.nanmedian(white_light)
median_spectrum = np.nanmedian(extracted_spectra, axis=0)
normalized_spectra = extracted_spectra / median_spectrum[None, :]
spectral_scatter_ppm = 1.0e6 * np.nanstd(normalized_spectra, axis=0)

print(
	f"Extracted spectra shape: {extracted_spectra.shape} "
	"(integration, dispersion)"
)
print(
	f"White-light curve stats: median={np.nanmedian(white_light):.6g}, "
	f"std={np.nanstd(white_light):.6g}"
)
print(
	f"Median extracted spectrum range=({np.nanmin(median_spectrum):.6g}, "
	f"{np.nanmax(median_spectrum):.6g})"
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

ax = axes[0, 0]
ax.plot(extracted_dispersion, median_spectrum, color="tab:blue", lw=1.5)
ax.set_xlabel("Dispersion pixel")
ax.set_ylabel("Median aperture flux")
ax.set_title("Median Extracted Spectrum")

ax = axes[0, 1]
ax.plot(integration_time_axis, white_light_norm, marker=".", lw=1, color="tab:green")
ax.axhline(1.0, color="k", ls=":", lw=1)
ax.set_xlabel(time_axis_label)
ax.set_ylabel("Normalized white-light flux")
ax.set_title("White-light Curve")

ax = axes[1, 0]
finite_norm = np.isfinite(normalized_spectra)
if np.any(finite_norm):
	vmin = float(np.nanpercentile(normalized_spectra[finite_norm], 1.0))
	vmax = float(np.nanpercentile(normalized_spectra[finite_norm], 99.0))
else:
	vmin, vmax = 0.98, 1.02
time_lo = float(np.nanmin(integration_time_axis))
time_hi = float(np.nanmax(integration_time_axis))
if time_hi <= time_lo:
	time_hi = time_lo + 1.0
im = ax.imshow(
	normalized_spectra,
	origin="lower",
	aspect="auto",
	extent=[extracted_dispersion[0], extracted_dispersion[-1], time_lo, time_hi],
	vmin=vmin,
	vmax=vmax,
	cmap="magma",
)
ax.set_xlabel("Dispersion pixel")
ax.set_ylabel(time_axis_label)
ax.set_title("Normalized Spectral Time Series")
fig.colorbar(im, ax=ax, label="Relative flux")

ax = axes[1, 1]
ax.plot(extracted_dispersion, spectral_scatter_ppm, color="tab:red", lw=1.2)
ax.set_xlabel("Dispersion pixel")
ax.set_ylabel("Scatter [ppm]")
ax.set_title("Per-channel Temporal Scatter")

plt.show()
# %%
