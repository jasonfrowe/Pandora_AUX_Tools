import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from matplotlib import colors

def read_vissci_science_cube(fits_path):
	"""Read a science datacube from a Pandora FITS file.

	Parameters
	----------
	fits_path : str
		Path to the VisSci FITS file.

	Returns
	-------
	cube : numpy.ndarray
		Science datacube as a 3D array with shape (n_frames, ny, nx).
	header : astropy.io.fits.Header
		Header for the selected science extension.
	"""
	with fits.open(fits_path, memmap=False) as hdul:
		ext_candidates = ["SCIENCE", "RAW SCIENCE"]
		selected_hdu = None
		for ext in ext_candidates:
			if ext in hdul and hdul[ext].data is not None:
				selected_hdu = hdul[ext]
				break

		if selected_hdu is None:
			for hdu in hdul:
				if hdu.data is not None and np.asarray(hdu.data).ndim >= 2:
					selected_hdu = hdu
					break

		if selected_hdu is None:
			raise KeyError(
				"No image data extension found. Tried SCIENCE/RAW SCIENCE and image HDUs."
			)

		cube = np.asarray(selected_hdu.data)
		header = selected_hdu.header.copy()

	if cube.ndim < 2:
		raise ValueError(f"Science data must be at least 2D; got shape {cube.shape}.")

	if cube.ndim == 2:
		cube = cube[np.newaxis, :, :]
	elif cube.ndim == 3:
		pass
	else:
		# Collapse all leading axes into a frame axis, preserving the image plane.
		ny, nx = cube.shape[-2], cube.shape[-1]
		cube = cube.reshape(-1, ny, nx)

	return cube, header


def display_science_image(
	cube,
	image_index=0,
	scale_style="zscale",
	cmap="viridis",
	vmin=None,
	vmax=None,
	iraf_contrast=0.25,
):
	"""Display one image from a SCIENCE datacube.

	Parameters
	----------
	cube : numpy.ndarray
		3D SCIENCE datacube with shape (n_frames, ny, nx).
	image_index : int, optional
		Index of the image/frame to display. Default is 0.
	scale_style : str, optional
		Display scale style: "none", "zscale", "log", or "sqrt".
		Default is "zscale".
	cmap : str, optional
		Matplotlib colormap name. Default is "viridis".
	vmin, vmax : float, optional
		Display scaling limits. User-provided values are honored for all
		scale styles.
	iraf_contrast : float, optional
		Contrast parameter for IRAF-style zscale. Default is 0.25.
	"""
	if cube.ndim != 3:
		raise ValueError(f"Expected a 3D datacube; got array with shape {cube.shape}.")

	if not 0 <= image_index < cube.shape[0]:
		raise IndexError(
			f"image_index={image_index} is out of bounds for {cube.shape[0]} frames."
		)

	frame = cube[image_index]
	finite = np.isfinite(frame)
	if not np.any(finite):
		raise ValueError("Selected frame has no finite values for display scaling.")

	user_set_vmin = vmin is not None
	user_set_vmax = vmax is not None

	valid_styles = {"none", "zscale", "log", "sqrt"}
	style = str(scale_style).lower()
	if style not in valid_styles:
		raise ValueError(
			f"scale_style must be one of {sorted(valid_styles)}; got '{scale_style}'."
		)

	data_min = float(np.nanmin(frame[finite]))
	data_max = float(np.nanmax(frame[finite]))

	if style == "zscale":
		zscale = ZScaleInterval(contrast=iraf_contrast)
		auto_vmin, auto_vmax = zscale.get_limits(frame[finite])
	elif style == "log":
		positive = frame[finite & (frame > 0)]
		if positive.size == 0:
			raise ValueError(
				"Log scaling requires at least one positive pixel in the selected frame."
			)

		# Use robust defaults so log mode is useful without manual limits.
		auto_vmin = float(np.nanpercentile(positive, 1.0))
		auto_vmax = float(np.nanpercentile(positive, 99.5))
		if auto_vmin <= 0 or auto_vmin >= auto_vmax:
			auto_vmin = float(np.nanmin(positive))
			auto_vmax = float(np.nanmax(positive))
	else:
		auto_vmin, auto_vmax = data_min, data_max

	if vmin is None:
		vmin = auto_vmin
	if vmax is None:
		vmax = auto_vmax

	if style in {"none", "zscale"}:
		if vmin >= vmax:
			raise ValueError(f"vmin ({vmin}) must be less than vmax ({vmax}).")
		norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
	elif style == "sqrt":
		if vmin >= vmax:
			raise ValueError(f"vmin ({vmin}) must be less than vmax ({vmax}).")
		norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax, clip=True)
	else:
		if (vmin <= 0 and user_set_vmin) or (vmax <= 0 and user_set_vmax):
			raise ValueError(
				"Log scaling requires user-provided vmin and vmax to be > 0."
			)
		if vmin <= 0:
			vmin = float(np.nextafter(0, 1))
		if vmax <= 0:
			vmax = float(np.nanmax(frame[finite & (frame > 0)]))
		if vmin >= vmax:
			raise ValueError(f"vmin ({vmin}) must be less than vmax ({vmax}).")
		norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)

	fig, ax = plt.subplots(figsize=(7, 7))
	img = ax.imshow(
		frame,
		origin="lower",
		cmap=cmap,
		norm=norm,
		interpolation="nearest",
	)
	ax.set_title(f"SCIENCE image {image_index} ({style})")
	ax.set_xlabel("X pixel")
	ax.set_ylabel("Y pixel")
	fig.colorbar(img, ax=ax, label="Counts")
	plt.tight_layout()
	plt.show()


def _extract_index_base(index_map, axis_size):
	"""Return index map converted to 0-based indexing for numpy use."""
	idx = np.asarray(index_map, dtype=np.int64)
	if idx.size == 0:
		raise ValueError("index_map is empty.")

	if np.nanmin(idx) >= 0 and np.nanmax(idx) < axis_size:
		return idx

	if np.nanmin(idx) >= 1 and np.nanmax(idx) <= axis_size:
		return idx - 1

	raise ValueError(
		f"Index map values are out of bounds for axis size {axis_size}. "
		f"Range is [{np.nanmin(idx)}, {np.nanmax(idx)}]."
	)


def get_rdf_auxiliary_data(fits_path):
	"""Read ROW/COLUMN maps and per-frame exposure times from an RDF FITS file."""
	with fits.open(fits_path, memmap=False) as hdul:
		row_map = np.asarray(hdul["ROW"].data) if "ROW" in hdul else None
		col_map = np.asarray(hdul["COLUMN"].data) if "COLUMN" in hdul else None
		exp_time = np.asarray(hdul["EXPOSURE_TIME"].data) if "EXPOSURE_TIME" in hdul else None

	if exp_time is not None:
		exp_time = np.asarray(exp_time, dtype=float).reshape(-1)

	return row_map, col_map, exp_time


def read_rdf_raw_science(fits_path):
	"""Read the RAW SCIENCE array from an RDF FITS file in native dimensions."""
	with fits.open(fits_path, memmap=False) as hdul:
		if "RAW SCIENCE" not in hdul:
			raise KeyError("RAW SCIENCE extension not found in RDF FITS file.")
		raw = np.asarray(hdul["RAW SCIENCE"].data)
		header = hdul["RAW SCIENCE"].header.copy()

	if raw.ndim != 4:
		raise ValueError(
			f"RAW SCIENCE is expected to be 4D (integration, group, row, column); got {raw.shape}."
		)

	return raw, header


def flatten_ramp_cube(ramp_cube):
	"""Flatten a 4D ramp cube (integration, group, row, column) into 3D frames."""
	ramp = np.asarray(ramp_cube)
	if ramp.ndim != 4:
		raise ValueError(f"ramp_cube must be 4D, got shape {ramp.shape}.")
	nint, ngroup, ny, nx = ramp.shape
	return ramp.reshape(nint * ngroup, ny, nx)


def unflatten_ramp_cube(flat_cube, nint, ngroup):
	"""Restore a flattened 3D cube back to 4D ramp shape."""
	flat = np.asarray(flat_cube)
	if flat.ndim != 3:
		raise ValueError(f"flat_cube must be 3D, got shape {flat.shape}.")
	if flat.shape[0] != nint * ngroup:
		raise ValueError(
			f"Frame axis {flat.shape[0]} does not match nint*ngroup={nint * ngroup}."
		)
	return flat.reshape(nint, ngroup, flat.shape[1], flat.shape[2])


def apply_ramp_cosmic_ray_correction(
	ramp_cube,
	sigma_threshold=8.0,
	min_jump_dn=50.0,
	positive_jumps_only=True,
):
	"""Detect and correct jump-like cosmic ray events in up-the-ramp data.

	The method follows a JWST-style jump approach: evaluate group-to-group
	differences per integration/pixel, detect outlier jumps, then remove the
	step from all subsequent groups in that integration.
	"""
	ramp = np.asarray(ramp_cube, dtype=float)
	if ramp.ndim != 4:
		raise ValueError(f"ramp_cube must be 4D, got shape {ramp.shape}.")

	nint, ngroup, ny, nx = ramp.shape
	if ngroup < 2:
		return ramp.copy(), {
			"n_events": 0,
			"fraction_diffs_flagged": 0.0,
			"sigma_threshold": float(sigma_threshold),
			"min_jump_dn": float(min_jump_dn),
		}

	diffs = np.diff(ramp, axis=1)
	base = np.nanmedian(diffs, axis=1, keepdims=True)
	dev = diffs - base
	mad = np.nanmedian(np.abs(dev), axis=1, keepdims=True)
	robust_sigma = 1.4826 * mad
	noise_floor = np.maximum(robust_sigma, 1.0)

	threshold = sigma_threshold * noise_floor
	if positive_jumps_only:
		jump_mask = (dev > threshold) & (dev > min_jump_dn)
	else:
		jump_mask = (np.abs(dev) > threshold) & (np.abs(dev) > min_jump_dn)

	corrected = ramp.copy()
	events = np.argwhere(jump_mask)
	for integ_idx, diff_idx, row_idx, col_idx in events:
		step = dev[integ_idx, diff_idx, row_idx, col_idx]
		corrected[integ_idx, diff_idx + 1 :, row_idx, col_idx] -= step

	info = {
		"n_events": int(events.shape[0]),
		"fraction_diffs_flagged": float(events.shape[0] / jump_mask.size),
		"sigma_threshold": float(sigma_threshold),
		"min_jump_dn": float(min_jump_dn),
	}
	return corrected, info


def apply_ramp_zero_point_correction(
	ramp_cube,
	n_baseline_groups=2,
	preserve_integration_median=True,
):
	"""Correct per-pixel zero points using early reads in each integration.

	Parameters
	----------
	ramp_cube : numpy.ndarray
		4D array with shape (integration, group, row, column).
	n_baseline_groups : int, optional
		Number of initial groups used to estimate zero point per integration/pixel.
	preserve_integration_median : bool, optional
		If True, preserves each integration's global median level after correction.
	"""
	ramp = np.asarray(ramp_cube, dtype=float)
	if ramp.ndim != 4:
		raise ValueError(f"ramp_cube must be 4D, got shape {ramp.shape}.")

	nint, ngroup, ny, nx = ramp.shape
	n0 = int(max(1, min(n_baseline_groups, ngroup)))

	zero_point_map = np.nanmedian(ramp[:, :n0, :, :], axis=1)
	corrected = ramp - zero_point_map[:, None, :, :]

	if preserve_integration_median:
		integration_offsets = np.nanmedian(zero_point_map, axis=(1, 2))
		corrected += integration_offsets[:, None, None, None]
	else:
		integration_offsets = np.zeros(nint, dtype=float)

	info = {
		"n_baseline_groups": n0,
		"preserve_integration_median": bool(preserve_integration_median),
		"integration_offset_median": float(np.nanmedian(integration_offsets)),
		"integration_offset_min": float(np.nanmin(integration_offsets)),
		"integration_offset_max": float(np.nanmax(integration_offsets)),
		"zero_point_map": zero_point_map,
	}
	return corrected, info


def compute_ramp_fit_products_r2s(ramp_cube, sigcut=2.0):
	"""Compute robust ramp-fit products per integration using JWST-style r2s fitting.

	Returns slope, intercept, and fit scatter images with shape
	(integration, row, column).
	"""
	try:
		import jwst_soss_reduction as soss_red
	except Exception as exc:
		raise ImportError(
			"jwst_soss_reduction.py is required for r2s-style ramp fitting."
		) from exc

	ramp = np.asarray(ramp_cube, dtype=float)
	if ramp.ndim != 4:
		raise ValueError(f"ramp_cube must be 4D, got shape {ramp.shape}.")

	nint, ngroup, ny, nx = ramp.shape
	dq = np.zeros((ny, nx), dtype=np.int32)
	nrsatmap = np.full((ny, nx), ngroup, dtype=np.int32)

	intercept_cube = np.zeros((nint, ny, nx), dtype=float)
	slope_cube = np.zeros((nint, ny, nx), dtype=float)
	scatter_cube = np.zeros((nint, ny, nx), dtype=float)
	bpix_cube = np.zeros((nint, ny, nx), dtype=np.int16)

	for i in range(nint):
		zpt, stdimage, slope, bpixmap = soss_red.r2s_test(
			ramp[i],
			dq,
			nrsatmap,
			sigcut=float(sigcut),
		)
		intercept_cube[i] = np.asarray(zpt, dtype=float)
		slope_cube[i] = np.asarray(slope, dtype=float)
		scatter_cube[i] = np.asarray(stdimage, dtype=float)
		bpix_cube[i] = np.asarray(bpixmap, dtype=np.int16)

	info = {
		"sigcut": float(sigcut),
		"n_integrations": int(nint),
		"n_groups": int(ngroup),
	}

	return {
		"intercept": intercept_cube,
		"slope": slope_cube,
		"scatter": scatter_cube,
		"bpix": bpix_cube,
		"info": info,
	}


def apply_scalar_zero_point_from_ramp_fit(
	ramp_cube,
	intercept_cube,
	mask=None,
):
	"""Apply per-integration scalar zero-point correction from fitted intercept maps."""
	ramp = np.asarray(ramp_cube, dtype=float)
	intercept = np.asarray(intercept_cube, dtype=float)

	if ramp.ndim != 4:
		raise ValueError(f"ramp_cube must be 4D, got shape {ramp.shape}.")
	if intercept.ndim != 3:
		raise ValueError(
			f"intercept_cube must be 3D (integration,row,column), got {intercept.shape}."
		)
	if intercept.shape != ramp.shape[:1] + ramp.shape[2:]:
		raise ValueError(
			f"intercept_cube shape {intercept.shape} incompatible with ramp shape {ramp.shape}."
		)

	if mask is not None:
		mask_arr = np.asarray(mask, dtype=bool)
		if mask_arr.shape != intercept.shape[1:]:
			raise ValueError(
				f"mask shape {mask_arr.shape} must match image plane {intercept.shape[1:]}."
			)
		intercept_work = np.where(mask_arr[None, :, :], np.nan, intercept)
	else:
		intercept_work = intercept

	integration_zero = np.nanmedian(intercept_work, axis=(1, 2))
	zero_ref = float(np.nanmedian(integration_zero))
	integration_zero_drift = integration_zero - zero_ref
	corrected = ramp - integration_zero_drift[:, None, None, None]

	info = {
		"integration_zero": integration_zero,
		"zero_reference": zero_ref,
		"integration_zero_drift": integration_zero_drift,
		"zero_min": float(np.nanmin(integration_zero)),
		"zero_max": float(np.nanmax(integration_zero)),
		"zero_median": float(np.nanmedian(integration_zero)),
		"drift_min": float(np.nanmin(integration_zero_drift)),
		"drift_max": float(np.nanmax(integration_zero_drift)),
	}

	return corrected, info


def detect_hot_pixels_from_ramp_fit(
	intercept_cube,
	scatter_cube=None,
	intercept_sigma=8.0,
	min_intercept_offset_dn=150.0,
	scatter_sigma=8.0,
):
	"""Build persistent hot/unstable pixel mask from ramp-fit diagnostics."""
	intercept = np.asarray(intercept_cube, dtype=float)
	if intercept.ndim != 3:
		raise ValueError(
			f"intercept_cube must be 3D (integration,row,column), got {intercept.shape}."
		)

	pixel_intercept = np.nanmedian(intercept, axis=0)
	global_med = float(np.nanmedian(pixel_intercept))
	global_mad = float(np.nanmedian(np.abs(pixel_intercept - global_med)))
	global_sig = 1.4826 * global_mad

	ithr = max(global_med + intercept_sigma * global_sig, global_med + min_intercept_offset_dn)
	hot_mask = pixel_intercept > ithr

	if scatter_cube is not None:
		scatter = np.asarray(scatter_cube, dtype=float)
		if scatter.shape != intercept.shape:
			raise ValueError(
				f"scatter_cube shape {scatter.shape} must match intercept_cube {intercept.shape}."
			)
		pixel_scatter = np.nanmedian(scatter, axis=0)
		s_med = float(np.nanmedian(pixel_scatter))
		s_mad = float(np.nanmedian(np.abs(pixel_scatter - s_med)))
		s_sig = 1.4826 * s_mad
		s_thr = s_med + scatter_sigma * s_sig
		hot_mask |= pixel_scatter > s_thr
	else:
		s_thr = np.nan

	hot_mask |= ~np.isfinite(pixel_intercept)

	info = {
		"intercept_threshold_dn": float(ithr),
		"scatter_threshold": float(s_thr) if np.isfinite(s_thr) else None,
		"hot_fraction": float(np.mean(hot_mask)),
		"n_hot": int(np.sum(hot_mask)),
	}
	return hot_mask, info


def detect_bad_pixel_clumps_from_ramp_fit(
	intercept_cube,
	slope_cube,
	scatter_cube=None,
	min_intercept_offset_dn=150.0,
	intercept_sigma=8.0,
	slope_percentile=2.0,
	scatter_sigma=8.0,
	dilate_iterations=1,
):
	"""Detect persistent bad/hot clumps using ramp-fit intercept/slope diagnostics.

	Targets pixels with anomalously high intercepts but very low slopes, which are
	typical of near-saturated offsets that collapse to near-zero ramp slopes.
	"""
	intercept = np.asarray(intercept_cube, dtype=float)
	slope = np.asarray(slope_cube, dtype=float)
	if intercept.ndim != 3 or slope.ndim != 3:
		raise ValueError("intercept_cube and slope_cube must both be 3D arrays.")
	if intercept.shape != slope.shape:
		raise ValueError(
			f"intercept_cube shape {intercept.shape} must match slope_cube {slope.shape}."
		)

	pixel_intercept = np.nanmedian(intercept, axis=0)
	pixel_slope = np.nanmedian(slope, axis=0)

	i_med = float(np.nanmedian(pixel_intercept))
	i_mad = float(np.nanmedian(np.abs(pixel_intercept - i_med)))
	i_sig = 1.4826 * i_mad
	i_thr = max(i_med + intercept_sigma * i_sig, i_med + min_intercept_offset_dn)

	finite_slope = pixel_slope[np.isfinite(pixel_slope)]
	if finite_slope.size == 0:
		s_thr = 0.0
	else:
		s_thr = float(np.nanpercentile(finite_slope, slope_percentile))

	mask = (pixel_intercept > i_thr) & (pixel_slope < s_thr)

	if scatter_cube is not None:
		scatter = np.asarray(scatter_cube, dtype=float)
		if scatter.shape != intercept.shape:
			raise ValueError(
				f"scatter_cube shape {scatter.shape} must match intercept_cube {intercept.shape}."
			)
		pixel_scatter = np.nanmedian(scatter, axis=0)
		sc_med = float(np.nanmedian(pixel_scatter))
		sc_mad = float(np.nanmedian(np.abs(pixel_scatter - sc_med)))
		sc_sig = 1.4826 * sc_mad
		sc_thr = sc_med + scatter_sigma * sc_sig
		mask |= pixel_scatter > sc_thr
	else:
		sc_thr = np.nan

	mask |= ~np.isfinite(pixel_intercept) | ~np.isfinite(pixel_slope)

	# Grow mask slightly so small clumps are corrected as connected structures.
	for _ in range(max(0, int(dilate_iterations))):
		grown = mask.copy()
		for dr in (-1, 0, 1):
			for dc in (-1, 0, 1):
				if dr == 0 and dc == 0:
					continue
				shift = np.roll(np.roll(mask, dr, axis=0), dc, axis=1)
				# Zero wrapped edges from roll.
				if dr > 0:
					shift[:dr, :] = False
				elif dr < 0:
					shift[dr:, :] = False
				if dc > 0:
					shift[:, :dc] = False
				elif dc < 0:
					shift[:, dc:] = False
				grown |= shift
		mask = grown

	info = {
		"intercept_threshold_dn": float(i_thr),
		"slope_threshold": float(s_thr),
		"scatter_threshold": float(sc_thr) if np.isfinite(sc_thr) else None,
		"clump_fraction": float(np.mean(mask)),
		"n_clump_pixels": int(np.sum(mask)),
	}
	return mask, info


def dilate_binary_mask(mask, iterations=1):
	"""Dilate a 2D boolean mask with 8-connectivity using numpy only."""
	m = np.asarray(mask, dtype=bool)
	if m.ndim != 2:
		raise ValueError(f"mask must be 2D, got shape {m.shape}.")

	out = m.copy()
	for _ in range(max(0, int(iterations))):
		grown = out.copy()
		for dr in (-1, 0, 1):
			for dc in (-1, 0, 1):
				if dr == 0 and dc == 0:
					continue
				shift = np.roll(np.roll(out, dr, axis=0), dc, axis=1)
				if dr > 0:
					shift[:dr, :] = False
				elif dr < 0:
					shift[dr:, :] = False
				if dc > 0:
					shift[:, :dc] = False
				elif dc < 0:
					shift[:, dc:] = False
				grown |= shift
		out = grown

	return out


def correct_bad_pixels_with_neighbors(
	cube,
	bad_mask,
	max_radius=3,
	min_neighbors=4,
):
	"""Replace bad pixels/clumps with local robust neighbor medians per frame."""
	arr = np.asarray(cube, dtype=float)
	if arr.ndim != 3:
		raise ValueError(f"cube must be 3D, got shape {arr.shape}.")

	mask = np.asarray(bad_mask, dtype=bool)
	if mask.shape != arr.shape[1:]:
		raise ValueError(
			f"bad_mask shape {mask.shape} must match image plane {arr.shape[1:]}."
		)

	if not np.any(mask):
		return arr.copy(), {"n_fixed": 0, "fixed_fraction": 0.0}

	ny, nx = mask.shape
	idx = np.argwhere(mask)
	corrected = arr.copy()
	n_fixed = 0

	for f in range(corrected.shape[0]):
		frame = corrected[f]
		for r, c in idx:
			filled = False
			for rad in range(1, max(1, int(max_radius)) + 1):
				r0 = max(0, r - rad)
				r1 = min(ny, r + rad + 1)
				c0 = max(0, c - rad)
				c1 = min(nx, c + rad + 1)

				patch = frame[r0:r1, c0:c1]
				patch_mask = mask[r0:r1, c0:c1]
				vals = patch[~patch_mask]
				vals = vals[np.isfinite(vals)]

				if vals.size >= min_neighbors:
					frame[r, c] = float(np.median(vals))
					filled = True
					n_fixed += 1
					break

			if not filled:
				frame[r, c] = np.nan

		corrected[f] = frame

	info = {
		"n_fixed": int(n_fixed),
		"fixed_fraction": float(n_fixed / (mask.sum() * corrected.shape[0])),
	}
	return corrected, info


def detect_hot_pixels_from_zero_point(
	zero_point_map,
	sigma_threshold=8.0,
	min_offset_dn=150.0,
):
	"""Build a persistent hot-pixel mask from integration zero-point maps."""
	zp = np.asarray(zero_point_map, dtype=float)
	if zp.ndim != 3:
		raise ValueError(
			f"zero_point_map must be 3D (integration, row, column), got {zp.shape}."
		)

	pixel_median = np.nanmedian(zp, axis=0)
	global_med = float(np.nanmedian(pixel_median))
	mad = float(np.nanmedian(np.abs(pixel_median - global_med)))
	robust_sigma = 1.4826 * mad
	threshold = max(global_med + sigma_threshold * robust_sigma, global_med + min_offset_dn)

	hot_mask = pixel_median > threshold
	hot_mask |= ~np.isfinite(pixel_median)

	info = {
		"threshold_dn": float(threshold),
		"global_median_dn": float(global_med),
		"robust_sigma_dn": float(robust_sigma),
		"hot_fraction": float(np.mean(hot_mask)),
		"n_hot": int(np.sum(hot_mask)),
	}
	return hot_mask, info


def apply_row_column_correction(
	cube,
	apply_row=True,
	apply_column=True,
	mask=None,
	row_mask=None,
	column_mask=None,
	preserve_frame_median=True,
):
	"""Apply robust per-frame row/column background correction (destriping)."""
	arr = np.asarray(cube, dtype=float)
	if arr.ndim != 3:
		raise ValueError(f"cube must be 3D, got shape {arr.shape}.")

	def _validate_mask(in_mask, name):
		if in_mask is None:
			return None
		m = np.asarray(in_mask, dtype=bool)
		if m.shape != arr.shape[1:]:
			raise ValueError(
				f"{name} shape {m.shape} must match image plane {arr.shape[1:]}."
			)
		return m

	base_mask = _validate_mask(mask, "mask")
	row_mask_arr = _validate_mask(row_mask, "row_mask")
	col_mask_arr = _validate_mask(column_mask, "column_mask")

	if row_mask_arr is None:
		row_mask_arr = base_mask
	if col_mask_arr is None:
		col_mask_arr = base_mask

	corrected = arr.copy()
	for i in range(corrected.shape[0]):
		frame = corrected[i]
		baseline = np.nanmedian(frame)

		if row_mask_arr is not None:
			row_work = np.where(row_mask_arr, np.nan, frame)
		else:
			row_work = frame

		if apply_row:
			row_bg = np.zeros(frame.shape[0], dtype=float)
			for r in range(frame.shape[0]):
				vals = row_work[r]
				finite = np.isfinite(vals)
				if np.any(finite):
					row_bg[r] = np.median(vals[finite])
				else:
					row_bg[r] = 0.0
			row_bg = np.where(np.isfinite(row_bg), row_bg, 0.0)
			frame = frame - row_bg[:, None]

		if apply_column:
			if col_mask_arr is not None:
				col_work = np.where(col_mask_arr, np.nan, frame)
			else:
				col_work = frame
			col_bg = np.zeros(frame.shape[1], dtype=float)
			for c in range(frame.shape[1]):
				vals = col_work[:, c]
				finite = np.isfinite(vals)
				if np.any(finite):
					col_bg[c] = np.median(vals[finite])
				else:
					col_bg[c] = 0.0
			col_bg = np.where(np.isfinite(col_bg), col_bg, 0.0)
			frame = frame - col_bg[None, :]

		if preserve_frame_median:
			frame += baseline

		corrected[i] = frame

	steps = []
	if apply_row:
		steps.append("subtract per-row robust median")
	if apply_column:
		steps.append("subtract per-column robust median")
	if preserve_frame_median:
		steps.append("restore original frame median")

	info = {
		"steps": steps,
		"n_frames": int(arr.shape[0]),
	}
	return corrected, info


def estimate_trace_aperture(
	cube,
	dispersion_min=75,
	dispersion_max=290,
	expected_spatial_center=40,
	max_half_width=25,
	smooth_window=5,
	threshold_sigma=2.5,
):
	"""Estimate a spatial extraction aperture from a corrected cube.

	This is designed for spectra with potentially double-humped spatial profiles.
	It returns a robust spatial profile and recommended aperture bounds.
	"""
	arr = np.asarray(cube, dtype=float)
	if arr.ndim != 3:
		raise ValueError(f"cube must be 3D (integration, dispersion, spatial), got {arr.shape}.")

	nint, ndisp, nspat = arr.shape
	d0 = max(0, int(dispersion_min))
	d1 = min(ndisp - 1, int(dispersion_max))
	if d1 <= d0:
		raise ValueError(f"Invalid dispersion range: [{dispersion_min}, {dispersion_max}].")

	center = int(np.clip(expected_spatial_center, 0, nspat - 1))
	max_half_width = int(max(2, min(max_half_width, nspat // 2)))

	profile = np.nanmedian(arr[:, d0 : d1 + 1, :], axis=(0, 1))
	if not np.any(np.isfinite(profile)):
		raise ValueError("No finite values available to estimate trace aperture.")

	win = max(1, int(smooth_window))
	if win % 2 == 0:
		win += 1
	kernel = np.ones(win, dtype=float) / float(win)
	profile_smooth = np.convolve(np.nan_to_num(profile, nan=np.nanmedian(profile)), kernel, mode="same")

	x = np.arange(nspat)
	far = (x < center - max_half_width) | (x > center + max_half_width)
	if np.any(far & np.isfinite(profile_smooth)):
		bg_vals = profile_smooth[far & np.isfinite(profile_smooth)]
	else:
		bg_vals = profile_smooth[np.isfinite(profile_smooth)]

	bg_med = float(np.nanmedian(bg_vals))
	bg_mad = float(np.nanmedian(np.abs(bg_vals - bg_med)))
	bg_sig = max(1.4826 * bg_mad, 1e-6)

	search_lo = max(0, center - max_half_width)
	search_hi = min(nspat - 1, center + max_half_width)
	search = np.zeros(nspat, dtype=bool)
	search[search_lo : search_hi + 1] = True

	peak_val = float(np.nanmax(profile_smooth[search]))
	abs_thr = bg_med + threshold_sigma * bg_sig
	frac_thr = bg_med + 0.25 * max(peak_val - bg_med, 0.0)
	thr = max(abs_thr, frac_thr)

	above = (profile_smooth >= thr) & search
	if not np.any(above):
		# Fallback to a conservative aperture around expected center.
		ap_lo = max(0, center - 6)
		ap_hi = min(nspat - 1, center + 6)
	else:
		idx = np.where(above)[0]
		split = np.where(np.diff(idx) > 1)[0] + 1
		groups = np.split(idx, split)

		best_group = None
		for g in groups:
			if g[0] <= center <= g[-1]:
				best_group = g
				break
		if best_group is None:
			best_group = max(groups, key=lambda g: np.nanmax(profile_smooth[g]))

		ap_lo = int(best_group[0])
		ap_hi = int(best_group[-1])

	# Detect up to two local peaks inside the aperture for double-humped diagnostics.
	peak_candidates = []
	for i in range(max(ap_lo + 1, 1), min(ap_hi, nspat - 2) + 1):
		if profile_smooth[i] >= profile_smooth[i - 1] and profile_smooth[i] >= profile_smooth[i + 1]:
			peak_candidates.append(i)

	if len(peak_candidates) == 0:
		peak_candidates = [int(np.nanargmax(profile_smooth[ap_lo : ap_hi + 1]) + ap_lo)]

	peak_candidates = sorted(
		peak_candidates,
		key=lambda i: profile_smooth[i],
		reverse=True,
	)
	peak_positions = peak_candidates[:2]

	ap_mask = np.zeros(nspat, dtype=bool)
	ap_mask[ap_lo : ap_hi + 1] = True

	return {
		"dispersion_range": (d0, d1),
		"aperture_col_min": ap_lo,
		"aperture_col_max": ap_hi,
		"aperture_width": int(ap_hi - ap_lo + 1),
		"expected_spatial_center": center,
		"peak_positions": peak_positions,
		"threshold": float(thr),
		"background_median": float(bg_med),
		"background_sigma": float(bg_sig),
		"profile": profile,
		"profile_smooth": profile_smooth,
		"aperture_mask": ap_mask,
	}


def extract_trace_spectra(
	cube,
	aperture_col_min,
	aperture_col_max,
	dispersion_min=75,
	dispersion_max=290,
):
	"""Extract simple aperture spectra from a corrected cube.

	Returns
	-------
	spectra : numpy.ndarray
		Shape (integration, n_dispersion_pixels).
	dispersion_pixels : numpy.ndarray
		Dispersion pixel indices for extracted rows.
	"""
	arr = np.asarray(cube, dtype=float)
	if arr.ndim != 3:
		raise ValueError(f"cube must be 3D, got {arr.shape}.")

	nint, ndisp, nspat = arr.shape
	d0 = max(0, int(dispersion_min))
	d1 = min(ndisp - 1, int(dispersion_max))
	c0 = max(0, int(aperture_col_min))
	c1 = min(nspat - 1, int(aperture_col_max))

	if d1 <= d0 or c1 <= c0:
		raise ValueError("Invalid extraction bounds.")

	sub = arr[:, d0 : d1 + 1, c0 : c1 + 1]
	spectra = np.nansum(sub, axis=2)
	dispersion_pixels = np.arange(d0, d1 + 1, dtype=int)
	return spectra, dispersion_pixels


def build_linear_trace_aperture(
	n_dispersion,
	dispersion_min,
	dispersion_max,
	spatial_left_start,
	spatial_left_end,
	spatial_right_start,
	spatial_right_end,
	n_spatial=None,
):
	"""Build linearly varying spatial aperture bounds across dispersion."""
	d0 = int(dispersion_min)
	d1 = int(dispersion_max)
	if d1 < d0:
		raise ValueError("dispersion_max must be >= dispersion_min.")
	if d0 < 0 or d1 >= int(n_dispersion):
		raise ValueError(
			f"Dispersion bounds [{d0}, {d1}] are outside 0..{int(n_dispersion) - 1}."
		)

	dispersion_pixels = np.arange(d0, d1 + 1, dtype=int)
	if dispersion_pixels.size == 1:
		frac = np.array([0.0])
	else:
		frac = (dispersion_pixels - d0) / float(d1 - d0)

	left = spatial_left_start + frac * (spatial_left_end - spatial_left_start)
	right = spatial_right_start + frac * (spatial_right_end - spatial_right_start)

	spatial_min = np.minimum(left, right)
	spatial_max = np.maximum(left, right)

	if n_spatial is not None:
		spatial_min = np.clip(spatial_min, 0, int(n_spatial) - 1)
		spatial_max = np.clip(spatial_max, 0, int(n_spatial) - 1)

	return {
		"dispersion_pixels": dispersion_pixels,
		"spatial_left": spatial_min,
		"spatial_right": spatial_max,
		"spatial_left_int": np.floor(spatial_min).astype(int),
		"spatial_right_int": np.ceil(spatial_max).astype(int),
	}


def extract_trace_spectra_variable_aperture(cube, aperture_model):
	"""Extract spectra using per-dispersion-pixel variable spatial bounds."""
	arr = np.asarray(cube, dtype=float)
	if arr.ndim != 3:
		raise ValueError(f"cube must be 3D, got {arr.shape}.")

	disp = np.asarray(aperture_model["dispersion_pixels"], dtype=int)
	left = np.asarray(aperture_model["spatial_left_int"], dtype=int)
	right = np.asarray(aperture_model["spatial_right_int"], dtype=int)

	if not (disp.size == left.size == right.size):
		raise ValueError("Aperture model arrays must have matching lengths.")

	nint, ndisp, nspat = arr.shape
	spectra = np.full((nint, disp.size), np.nan, dtype=float)

	for k, d in enumerate(disp):
		if d < 0 or d >= ndisp:
			raise ValueError(f"Dispersion pixel {d} is out of bounds for cube shape {arr.shape}.")
		c0 = max(0, int(left[k]))
		c1 = min(nspat - 1, int(right[k]))
		if c1 < c0:
			continue
		spectra[:, k] = np.nansum(arr[:, d, c0 : c1 + 1], axis=1)

	return spectra, disp


def load_visda_reference_products(ref_root=None):
	"""Load VISDA reference products from the local pandora-ref repository."""
	if ref_root is None:
		ref_root = Path(__file__).resolve().parent / "pandora-ref"
	else:
		ref_root = Path(ref_root)

	visda_dir = ref_root / "src" / "pandoraref" / "data" / "visda"
	if not visda_dir.exists():
		raise FileNotFoundError(
			f"VISDA reference directory not found: {visda_dir}. "
			"Set ref_root to the pandora-ref checkout path."
		)

	def _read_image(path):
		with fits.open(path, memmap=False) as hdul:
			if len(hdul) > 1 and hdul[1].data is not None:
				return np.asarray(hdul[1].data), hdul[1].header.copy(), hdul[0].header.copy()
			if hdul[0].data is not None:
				return np.asarray(hdul[0].data), hdul[0].header.copy(), hdul[0].header.copy()
			raise ValueError(f"No image data found in {path}.")

	bias, _, bias_primary = _read_image(visda_dir / "bias.fits")
	flat, _, flat_primary = _read_image(visda_dir / "flat.fits")
	badpix, _, badpix_primary = _read_image(visda_dir / "badpix.fits")
	stripes_1d, _, stripes_primary = _read_image(visda_dir / "stripes.fits")

	with fits.open(visda_dir / "bias_0D.fits", memmap=False) as hdul:
		bias_0d = float(hdul[1].data[0][0])
		bias_0d_header = hdul[0].header.copy()

	with fits.open(visda_dir / "dark.fits", memmap=False) as hdul:
		dark_e_per_s = float(hdul[0].header["DARK"])
		dark_header = hdul[0].header.copy()

	with fits.open(visda_dir / "gain.fits", memmap=False) as hdul:
		gain_e_per_dn = float(hdul[0].header["GAIN"])
		gain_header = hdul[0].header.copy()

	return {
		"paths": {
			"visda_dir": str(visda_dir),
		},
		"bias": np.asarray(bias, dtype=float),
		"flat": np.asarray(flat, dtype=float),
		"badpix": np.asarray(badpix),
		"stripes_1d": np.asarray(stripes_1d, dtype=float).reshape(-1),
		"bias_0d": bias_0d,
		"dark_e_per_s": dark_e_per_s,
		"gain_e_per_dn": gain_e_per_dn,
		"headers": {
			"bias": bias_primary,
			"flat": flat_primary,
			"badpix": badpix_primary,
			"stripes": stripes_primary,
			"bias_0d": bias_0d_header,
			"dark": dark_header,
			"gain": gain_header,
		},
	}


def map_reference_to_roi(reference_2d, row_map, col_map):
	"""Project a full-frame 2D reference image onto the RDF ROI using ROW/COLUMN."""
	ref = np.asarray(reference_2d)
	if ref.ndim != 2:
		raise ValueError(f"reference_2d must be 2D, got shape {ref.shape}.")

	row = _extract_index_base(row_map, ref.shape[0])
	col = _extract_index_base(col_map, ref.shape[1])
	if row.shape != col.shape:
		raise ValueError(
			f"ROW/COLUMN map shape mismatch: {row.shape} vs {col.shape}."
		)

	return ref[row, col]


def map_stripes_to_roi(stripes_1d, col_map):
	"""Project a full-frame 1D column stripe model onto the RDF ROI."""
	stripes = np.asarray(stripes_1d).reshape(-1)
	col = _extract_index_base(col_map, stripes.size)
	return stripes[col]


def apply_visda_reference_corrections(
	cube,
	row_map,
	col_map,
	ref_products,
	exposure_time_s=None,
	apply_bias_0d=True,
	apply_bias_2d=True,
	apply_stripes=False,
	apply_dark=True,
	apply_flat=True,
	apply_badpix_mask=True,
	convert_to_electrons=False,
):
	"""Apply first-pass VISDA corrections to a science cube.

	Returns
	-------
	corrected_cube : numpy.ndarray
		Corrected cube.
	info : dict
		Metadata including a list of applied correction steps.
	"""
	arr = np.asarray(cube, dtype=float)
	if arr.ndim != 3:
		raise ValueError(f"cube must be 3D, got shape {arr.shape}.")

	if row_map is None or col_map is None:
		raise ValueError("row_map and col_map are required for ROI-based corrections.")

	if row_map.shape != arr.shape[1:] or col_map.shape != arr.shape[1:]:
		raise ValueError(
			"ROW/COLUMN maps must match image plane shape. "
			f"Got row={None if row_map is None else row_map.shape}, "
			f"col={None if col_map is None else col_map.shape}, image={arr.shape[1:]}."
		)

	corrected = arr.copy()
	steps = []

	bias_roi = map_reference_to_roi(ref_products["bias"], row_map, col_map)
	flat_roi = map_reference_to_roi(ref_products["flat"], row_map, col_map)
	badpix_roi = map_reference_to_roi(ref_products["badpix"], row_map, col_map)
	stripes_roi = map_stripes_to_roi(ref_products["stripes_1d"], col_map)

	if apply_bias_0d:
		corrected -= float(ref_products["bias_0d"])
		steps.append(f"subtract bias_0d ({ref_products['bias_0d']:.5g} DN)")

	if apply_bias_2d:
		corrected -= bias_roi[None, :, :]
		steps.append("subtract 2D bias map (ROI sampled)")

	if apply_stripes:
		corrected -= stripes_roi[None, :, :]
		steps.append("subtract stripe model (ROI sampled)")

	if apply_dark:
		dark_e_per_s = float(ref_products["dark_e_per_s"])
		gain_e_per_dn = float(ref_products["gain_e_per_dn"])
		dark_dn_per_s = dark_e_per_s / gain_e_per_dn if gain_e_per_dn != 0 else 0.0

		if exposure_time_s is None:
			exp = np.ones(arr.shape[0], dtype=float)
			steps.append(
				"subtract dark current using 1.0 s default exposure (EXPOSURE_TIME not provided)"
			)
		else:
			exp = np.asarray(exposure_time_s, dtype=float).reshape(-1)
			if exp.size != arr.shape[0]:
				raise ValueError(
					f"Exposure length {exp.size} does not match frame count {arr.shape[0]}."
				)
			steps.append("subtract dark current using per-frame EXPOSURE_TIME")

		corrected -= (exp[:, None, None] * dark_dn_per_s)

	if apply_flat:
		flat_safe = np.where(flat_roi == 0, np.nan, flat_roi)
		corrected /= flat_safe[None, :, :]
		steps.append("divide by flat field (ROI sampled)")

	if convert_to_electrons:
		corrected *= float(ref_products["gain_e_per_dn"])
		steps.append("convert DN to electrons using gain")

	if apply_badpix_mask:
		bad_mask = np.asarray(badpix_roi) > 0
		if np.any(bad_mask):
			corrected[:, bad_mask] = np.nan
		steps.append("mask bad pixels using badpix map")

	info = {
		"steps": steps,
		"gain_e_per_dn": float(ref_products["gain_e_per_dn"]),
		"dark_e_per_s": float(ref_products["dark_e_per_s"]),
		"bias_0d": float(ref_products["bias_0d"]),
		"ref_visda_dir": ref_products["paths"]["visda_dir"],
	}

	return corrected, info


def display_correction_comparison(
	raw_cube,
	corrected_cube,
	image_index=0,
	scale_style="zscale",
	iraf_contrast=0.25,
	cmap="viridis",
	diff_cmap="RdBu_r",
	rotate_k=1,
	xlabel=None,
	ylabel=None,
):
	"""Show raw, corrected, and residual images with consistent display scaling."""
	raw = np.asarray(raw_cube)
	corr = np.asarray(corrected_cube)
	if raw.ndim != 3 or corr.ndim != 3:
		raise ValueError("raw_cube and corrected_cube must both be 3D arrays.")
	if raw.shape != corr.shape:
		raise ValueError(f"Cube shape mismatch: {raw.shape} vs {corr.shape}.")
	if not 0 <= image_index < raw.shape[0]:
		raise IndexError(
			f"image_index={image_index} is out of bounds for {raw.shape[0]} frames."
		)

	raw_frame = raw[image_index]
	corr_frame = corr[image_index]
	resid = corr_frame - raw_frame

	raw_frame_plot = np.rot90(raw_frame, k=rotate_k)
	corr_frame_plot = np.rot90(corr_frame, k=rotate_k)
	resid_plot = np.rot90(resid, k=rotate_k)

	if xlabel is None or ylabel is None:
		# Original image uses (row, column) = (dispersion, spatial).
		# After odd 90-degree rotations, displayed x/y swap.
		if rotate_k % 2 != 0:
			xlabel_auto = "Dispersion pixel"
			ylabel_auto = "Spatial pixel"
		else:
			xlabel_auto = "Spatial pixel"
			ylabel_auto = "Dispersion pixel"
		xlabel = xlabel if xlabel is not None else xlabel_auto
		ylabel = ylabel if ylabel is not None else ylabel_auto

	combined = np.concatenate([raw_frame.ravel(), corr_frame.ravel()])
	combined = combined[np.isfinite(combined)]
	if combined.size == 0:
		raise ValueError("No finite values found for comparison display.")

	style = str(scale_style).lower()
	if style == "zscale":
		vmin, vmax = ZScaleInterval(contrast=iraf_contrast).get_limits(combined)
	else:
		vmin, vmax = float(np.nanpercentile(combined, 1)), float(np.nanpercentile(combined, 99))

	resid_abs = np.nanpercentile(np.abs(resid[np.isfinite(resid)]), 99)
	if not np.isfinite(resid_abs) or resid_abs == 0:
		resid_abs = 1.0

	fig, axes = plt.subplots(
		3,
		1,
		figsize=(10, 10),
		gridspec_kw={"hspace": 0.08},
		constrained_layout=True,
	)

	im0 = axes[0].imshow(
		raw_frame_plot,
		origin="lower",
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		aspect="auto",
	)
	axes[0].set_title("Raw")
	axes[0].set_xlabel(xlabel)
	axes[0].set_ylabel(ylabel)
	fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

	im1 = axes[1].imshow(
		corr_frame_plot,
		origin="lower",
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		aspect="auto",
	)
	axes[1].set_title("Corrected")
	axes[1].set_xlabel(xlabel)
	axes[1].set_ylabel(ylabel)
	fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

	im2 = axes[2].imshow(
		resid_plot,
		origin="lower",
		cmap=diff_cmap,
		vmin=-resid_abs,
		vmax=resid_abs,
		aspect="auto",
	)
	axes[2].set_title("Residual (Corrected - Raw)")
	axes[2].set_xlabel(xlabel)
	axes[2].set_ylabel(ylabel)
	fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

	plt.show()