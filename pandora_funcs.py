import numpy as np
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