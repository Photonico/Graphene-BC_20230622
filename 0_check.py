### Check vasp work
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

#%%

from Store.output import vasprun_directory, plot_color_families

Dir = "1_Self_Consistent"

vasprun_directory(Dir)

# %%

plot_color_families()

# %%
