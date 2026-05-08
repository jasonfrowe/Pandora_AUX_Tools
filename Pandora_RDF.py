# %%
import numpy as np
import importlib
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
InfImg = 'Pandora_RDF_WASP-178b_0.fits'
# %%
cube, science_header = pandora.read_vissci_science_cube(datadir + InfImg)
print(f"Loaded SCIENCE cube with shape: {cube.shape}")
print(f"SCIENCE EXTNAME: {science_header.get('EXTNAME', 'N/A')}")
pandora.display_science_image(cube, image_index=20, scale_style="zscale")
# %%
