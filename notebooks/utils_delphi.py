from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
import smplotlib
from hmf import MassFunction
from astropy.cosmology import FlatLambdaCDM, Planck18
import pandas as pd
import gay_color_maps as gcm
#############
### Paths ###
#############
root = Path("../").resolve()
tree_dir = Path("/Users/dexbhadra/Documents/delphi/input_z4p5Tree_files/binary_tree_n_progenitors_mh8to14_2_not_mine")
results = root / "Results_600"
zgrid = Path("/Users/dexbhadra/Documents/delphi/input_z4p5Tree_files/redshift_grid_mh8to14.bin")
models = {"fiducial", "eSFE", "eIMF"}

path_merger_tree = Path("/Users/dexbhadra/Documents/delphi/input_z4p5Tree_files/binary_tree_n_progenitors_mh8to14_2_not_mine") / "merger_tree"
path_redshifts = Path("/Users/dexbhadra/Documents/delphi/input_z4p5Tree_files/redshift_grid_mh8to14.bin")

Om, Ob, Ode, h = 0.30966, 0.04897, 0.68885, 0.6766
G, Msun, Mpc = 6.67e-8, 2.0e33, 3.086e24
a_grain, rho_grain, spin = 5.0e-6, 2.25, 0.04
lam_A, c_A = 1500.0, 2.9979e18
H0_cgs = h * 1e7 / Mpc

gotberg19_path = Path("/Users/dexbhadra/Documents/delphi/delphi25_valentin/600_trees/goetberg19")
sb99_path = Path("/Users/dexbhadra/Documents/delphi/published_runs_SB99")
pysb_path = Path("/Users/dexbhadra/Documents/delphi/delphi25_valentin/strippers/data_pysb99/pysb99")
metallicities = ["0.014", "0.006", "0.002", "0.0002"]                         # the four G19 metallicities (0.014 = solar)
suffix = {"0.014": "014", "0.006": "006", "0.002": "002", "0.0002": "0002"}   # file name endings of nbrs*.dat and q*.dat
c_A = 2.99792458e18            # speed of light [Angstrom / s]
eV_A = 12398.42                # photon energy E [eV] = 12398.42 / wavelength [A]
lam_H = 911.75                 # hydrogen ionization edge [A] (13.6 eV)

Z_color = {"0.014": "#D60270", "0.006": "#9B4F96", "0.002": "#0038A8", "0.0002": "#000000"}   
single_color = {2: "#870047", 5: "#9B4F96", 10: "#0038A8"}                                    
age_color = {11: "#FF218C", 20: "#FFD800", 50: "#21B1FF", 100: "#000000"}
metal_color = {"0.014": "#870047", "0.006": "#9B4F96", "0.002": "#0038A8", "0.0002": "black"}
age_color_stripped = {30: "#FF218C", 150: "#FFD800", 300: "#21B1FF", 450: "#000000",
                      600: "#870047", 750: "#9B4F96", 900: "#0038A8"}

sb99_file = {"0.014": "run_Z014/SED_SB99_Z0.014_constant.txt",
             "0.006": "run_Z006/SED_SB99_Z0.008_constant.txt",
             "0.002": "run_Z002/SED_SB99_Z0.002_constant.txt",
             "0.0002": "run_Z0002/SED_SB99_Z0.001_constant.txt"}
pysb_Z = {"0.014": "0.014", "0.006": "0.006", "0.002": "0.002", "0.0002": "0.0004"}


    
def define_mass_function():
    cosmo = FlatLambdaCDM(H0=100 * h, Om0=Om, Ob0=Ob, Tcmb0=2.7255)
    mass_function = MassFunction(z=4.5, cosmo_model=cosmo, sigma_8=0.811, n=0.965,
                                 hmf_model="ST", transfer_model="EH", Mmin=7,
                                 Mmax=15, dlog10m=0.01)
    logM_st = np.log10(mass_function.m / h)
    phi_st = mass_function.dndlnm * h**3

    return mass_function, logM_st, phi_st

def give_me_my_cosmo():
    cosmo = FlatLambdaCDM(H0=100 * h, Om0=Om, Ob0=Ob, Tcmb0=2.7255)
    return cosmo
    
def redshift_grid_reader(path_redshifts):
    with FortranFile(path_redshifts, "r") as f:
        n_redshifts = f.read_ints(np.int32)[0]
        redshifts = f.read_reals(np.float64)
    return n_redshifts, redshifts


def read_single_merger_tree_input(p):
    with FortranFile(p, "r") as f:
        n_halo_tree = int(f.read_ints(np.int32)[0])
        halo_redshift_int = f.read_ints(np.int32)
        halo_mass = f.read_reals(np.float64)
        halo_parent = f.read_ints(np.int32)
        halo_n_progenitors = f.read_ints(np.int32)
    return n_halo_tree, halo_redshift_int, halo_mass, halo_parent, halo_n_progenitors

def read_merger_tree_output(p, index):
    with FortranFile(p / f"merger_tree_out_{index:04d}", "r") as f:
        sfr = f.read_record(np.float64)
        stellar_mass_final = f.read_record(np.float64)
        lum_0 = f.read_record(np.float64)   # ionising photon rate
        lum_1 = f.read_record(np.float64)   
        lum_2 = f.read_record(np.float64)   # 1500 A  
        lum_3 = f.read_record(np.float64)
        lum_4 = f.read_record(np.float64)
        lum_5 = f.read_record(np.float64)
        lum_6 = f.read_record(np.float64)
        gas_mass_pre_feedback = f.read_record(np.float64)
        gas_mass_final = f.read_record(np.float64)
        dust_mass_final = f.read_record(np.float64)
        init_metallicity = f.read_record(np.float64)
        gas_metallicity = f.read_record(np.float64)
    return sfr, stellar_mass_final, lum_0, lum_1, lum_2, lum_3, lum_4, lum_5, lum_6, gas_mass_pre_feedback, gas_mass_final, dust_mass_final, init_metallicity, gas_metallicity

def Lnu_from_Llambda(L_l):
    return L_l * (lam_A ** 2) / c_A                    # erg/s/A -> erg/s/Hz

def Muv_from_Lnu(L_nu):
    d_10pc = 10.0 * 3.0857e18
    f_nu = L_nu / (4.0 * np.pi * d_10pc ** 2)
    with np.errstate(divide="ignore"):
        return -2.5 * np.log10(f_nu / 3631e-23)

def Muv_from_Llambda(L_l):
    return Muv_from_Lnu(Lnu_from_Llambda(L_l))

def tau_uv(mass_halos, z, dust):
    Hz = H0_cgs * np.sqrt(Om * (1 + z) ** 3 + Ode)
    vc = (10.0 * G * mass_halos * Msun * Hz) ** (1.0 / 3.0)
    rvir = G * mass_halos * Msun / vc ** 2
    rgas = 4.5 * spin * rvir * (1 + z) / 6.0
    tau = 3.0 * dust * Msun / (4.0 * np.pi * rgas ** 2 * a_grain * rho_grain)
    return tau

def transmit(tau):
    f = np.ones_like(tau, dtype=float)
    f[tau > 1e-6] = -np.expm1(-tau[tau > 1e-6]) / tau[tau > 1e-6]
    return f

def uv_lf_single_run(results, model, z_use, bins_magnitude, logM_st, phi_st, dlogM_tree):
    hist_int = {z_i: np.zeros(len(bins_magnitude) - 1) for z_i in z_use}
    hist_obs = {z_i: np.zeros(len(bins_magnitude) - 1) for z_i in z_use}
    for outp in sorted(results.glob("merger_tree_out_*")):
        index_tree = int(outp.name.split("_")[-1])
        inp = tree_dir / f"merger_tree_{index_tree:05d}"
        n_halo_tree, halo_redshift_int, halo_mass, halo_parent, halo_n_progenitors = read_single_merger_tree_input(f"{path_merger_tree}_{index_tree:05d}")
        
        sfr, stellar_mass_final, lum_0, lum_1, luminosity_uv, lum_3, lum_4, lum_5, lum_6, gass_mass_pre_feedback, gas_mass_final, dust_mass_final, init_metallicity, gas_metallicity = read_merger_tree_output(results, index_tree)
        
        w = np.interp(np.log10(halo_mass[0]), logM_st, phi_st) * dlogM_tree
       # import pdb; pdb.set_trace()
        for z_t, z_i in z_use.items():         
            sel = halo_redshift_int == z_i
            z = redshifts[z_i - 1]
            ww = np.full(sel.sum(), w)
            M_int = Muv_from_Llambda(luminosity_uv[sel])
            M_obs = Muv_from_Llambda(luminosity_uv[sel] * transmit(tau_uv(halo_mass[sel], z, dust_mass_final[sel])))
            hist_int[z_t] += np.histogram(M_int, bins=bins_magnitude, weights=ww)[0]
            hist_obs[z_t] += np.histogram(M_obs, bins=bins_magnitude, weights=ww)[0]
    return hist_int, hist_obs


def rho_uv_one_run(results, M_lim, snaps, logM_st, phi_st, dlogM_tree):
    rho_int = np.zeros(len(snaps))
    rho_obs = np.zeros(len(snaps))
    for outp in sorted(results.glob("merger_tree_out_[0-9][0-9][0-9][0-9]")):
        index_tree = int(outp.name.split("_")[-1])
        n_halo_tree, halo_redshift_int, halo_mass, halo_parent, halo_n_progenitors = read_single_merger_tree_input(f"{path_merger_tree}_{index_tree:05d}")
        
        sfr, stellar_mass_final, lum_0, lum_1, luminosity_uv, lum_3, lum_4, lum_5, lum_6, gass_mass_pre_feedback, gas_mass_final, dust_mass_final, init_metallicity, gas_metallicity = read_merger_tree_output(results, index_tree)
        w = np.interp(np.log10(halo_mass[0]), logM_st, phi_st) * dlogM_tree
        for k, z_i in enumerate(snaps):
            sel = halo_redshift_int == z_i
            z = redshifts[z_i - 1]
            L_int = Lnu_from_Llambda(luminosity_uv[sel])
            L_obs = L_int * transmit(tau_uv(halo_mass[sel], z, dust_mass_final[sel]))
            rho_int[k] += w * L_int.sum()
            rho_obs[k] += w * L_obs[Muv_from_Lnu(L_obs) < M_lim].sum()
    return rho_int, rho_obs

def Lnu_of_M(M):
    # inverse of Muv_from_Lnu
    d_10pc = 10.0 * 3.0857e18                          # 10 pc in cm
    f_nu = 3631e-23 * 10 ** (-0.4 * M)                 # erg/s/cm^2/Hz at 10 pc
    return 4.0 * np.pi * d_10pc ** 2 * f_nu            # erg/s/Hz

def rho_dpl(phi_star, M_star,M_grid, alpha, beta):
    # double power law, Donnan+23 Eq. 2
    phi = phi_star / (10 ** (0.4 * (alpha + 1) * (M_grid - M_star)) + 10 ** (0.4 * (beta + 1) * (M_grid - M_star)))
    return np.trapezoid(phi * Lnu_of_M(M_grid), M_grid)

def smf_one_run(results, bins_mstar, z_use, logM_st, phi_st, dlogM_tree):
    hist_mstar = {z_i: np.zeros(len(bins_mstar) - 1) for z_i in z_use}
    for outp in sorted(results.glob("merger_tree_out_*")):
        index_tree = int(outp.name.split("_")[-1])
        inp = tree_dir / f"merger_tree_{index_tree:05d}"
        n_halo_tree, halo_redshift_int, halo_mass, halo_parent, halo_n_progenitors = read_single_merger_tree_input(f"{path_merger_tree}_{index_tree:05d}")
        
        sfr, stellar_mass_final, lum_0, lum_1, luminosity_uv, lum_3, lum_4, lum_5, lum_6, gass_mass_pre_feedback, gas_mass_final, dust_mass_final, init_metallicity, gas_metallicity = read_merger_tree_output(results, index_tree)
        mstar = stellar_mass_final
        weight = np.interp(np.log10(halo_mass[0]), logM_st, phi_st) * dlogM_tree
        for z_target, z_index in z_use.items():
            #import pdb; pdb.set_trace()
            index_range = (halo_redshift_int == z_index)
            hist_mstar[z_target] += np.histogram(np.log10(mstar[index_range]), bins=bins_mstar, weights=np.full(index_range.sum(), weight))[0]
    return hist_mstar


def read_counts(Z):
    names = ["time", "total", "caseA", "caseB", "BCEE", "above10", "above5", "above3"]
    return pd.read_csv(gotberg19_path / f"nbrs{suffix[Z]}.dat", sep=r"\s+", comment="#", names=names)

def read_q(Z):
    # the photon rates Q0 (H), Q1 (He I), Q2 (He II) [1/s] after a 1e6 Msun burst.
    # log10 Q = -30 when there are no stripped stars, so 10**(-30) is effectively zero.
    q = pd.read_csv(gotberg19_path / f"q{suffix[Z]}.dat", sep=r"\s+", comment="#", names=["time", "Q0", "Q1", "Q2"])
    q[["Q0", "Q1", "Q2"]] = 10.0 ** q[["Q0", "Q1", "Q2"]]
    return q

def read_spectra(path):
    # first row: 0, then the wavelengths [A]; 
    #every other row: time [yr], then L_lambda [erg/s/A] at each wavelength
    table = pd.read_csv(path, sep=r"\s+", comment="#", header=None).to_numpy()
    return table[1:, 0], table[0, 1:], table[1:, 1:]          # time, wavelength, luminosity (time x wavelength)

# The G19 stripped-star spectra: burst and constant star formation
burst, constant = {}, {}
for Z in metallicities:
    burst[Z] = read_spectra(gotberg19_path / "sed" / f"SED_Z{Z}_starburst.txt")
    constant[Z] = read_spectra(gotberg19_path / "sed" / f"SED_Z{Z}_constant.txt")

# Both return the single stars of a 1e6 Msun burst at age_yr: wavelength [A] and L_lambda [erg/s/A]
def sb99_single(Z, age_yr):
    # Starburst99 gives a constant SFR of 1 Msun/yr. The light of a 1e6 Msun burst at age t is
    table = pd.read_csv(sb99_path / sb99_file[Z], sep=r"\s+", comment="#", header=None).to_numpy()
    time, wave, log_L = table[1:, 0], table[0, 1:], table[1:, 1:]
    L_const = 10**log_L
    L_const[log_L <= 0] = 0                
    L_after = np.array([np.interp(age_yr + 0.05e6, time, L) for L in L_const.T])
    L_before = np.array([np.interp(age_yr - 0.05e6, time, L) for L in L_const.T])
    return wave, 1e6 * (L_after - L_before) / 0.1e6

def pysb_single(Z, age_yr):
    # pySB99 is run for a 1e6 Msun burst directly.
    folder = pysb_path / f"Z{pysb_Z[Z]}"
    time = np.loadtxt(folder / "SED_times.txt")
    wave = np.loadtxt(folder / "SED_wavelength.txt")
    log_L = np.load(folder / "pySB_SED_stellar.npy")          # log10 L_lambda (time x wavelength)
    return wave, 10**log_L[np.argmin(np.abs(time - age_yr))]


def to_g19_grid(wave, wave_s, L_s):
    # put single-star L_s (on wave_s) on the G19 wavelengths 
    ion, ion_s = wave <= lam_H, wave_s <= lam_H
    return np.where(ion, np.interp(wave, wave_s[ion_s], L_s[ion_s]), np.interp(wave, wave_s[~ion_s], L_s[~ion_s]))


def plot_single(ax, wave, L_lambda, color, label, ls="-"):
    # draw log10(L_nu / L_nu(13.6 eV)) against photon energy
    energy = eV_A / wave
    L_nu = L_lambda * wave**2 / c_A
    ion = energy >= 13.6                                              # L_nu(13.6 eV) from the ionizing side of the edge
    L_at_13_6 = np.interp(13.6, energy[ion][::-1], L_nu[ion][::-1])   # [::-1]: np.interp needs increasing energy
    ax.plot(energy, np.log10(np.maximum(L_nu / L_at_13_6, 1e-10)), color=color, ls=ls, lw=1.2, label=label)

def plot_population_shape(ax, Z, single_stars, age_myr=None):
    time, wave, L_strip = burst[Z]          # stripped stars
    # AGN: L_nu proportional to nu^-1.2 ... nu^-2 (grey band)
    E_agn = np.linspace(13.6, 90, 100)
    ax.fill_between(E_agn, -2.0 * np.log10(E_agn / 13.6), -1.2 * np.log10(E_agn / 13.6), color="#D9D9D9", lw=0, label="AGN")

    # young ages: single stars only
    if age_myr is None:
        for age_myr, color in single_color.items():
            #The star unpacks the tuple into two distinct argments of wavelengths and luminosities
            plot_single(ax, wave, to_g19_grid(wave, *single_stars(Z, age_myr * 1e6)), color, f"{age_myr} Myr, single")

    # later ages: single + stripped stars
        for age_myr, color in age_color.items():
            row = np.argmin(np.abs(time - age_myr * 1e6))              # the stripped-star time closest to this age
            L_total = to_g19_grid(wave, *single_stars(Z, age_myr * 1e6)) + L_strip[row]
            plot_single(ax, wave, L_total, color, f"{age_myr} Myr")
    else:
        # one line per age, coloured by age (colour bar drawn in plot_population)
        cmap = gcm.pan()
        norm = plt.Normalize(min(age_myr), max(age_myr))
        for age in age_myr:
            row = np.argmin(np.abs(time - age * 1e6))
            L_total = to_g19_grid(wave, *single_stars(Z, age * 1e6)) + L_strip[row]
            plot_single(ax, wave, L_total, cmap(norm(age)), label=None)




            
            
        
    # ionization edges of H, He I and He II
    for edge_eV in [13.6, 24.6, 54.4]:
        ax.axvline(edge_eV, color="gray", ls=":", lw=0.8)

    ax.set_xlim(10, 90)
    ax.set_ylim(-3, 0.5)
    ax.text(0.95, 0.92, f"Z = {Z}", transform=ax.transAxes, ha="right", fontsize=14)


def plot_population(single_stars, title, file_name, age_myr=None):
    # 2 rows, 2 columns: one panel per metallicity
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    for ax, Z in zip(axes.ravel(), metallicities):
        plot_population_shape(ax, Z, single_stars, age_myr)

    for ax in axes[1, :]:
        ax.set_xlabel("energy [eV]")
    axes[1, 0].set_xticks(np.arange(10, 81, 10))     
    for ax in axes[:, 0]:
        ax.set_ylabel(r"log$_{10}$(L$_\nu$ / L$_\nu$(13.6 eV))")
    fig.suptitle(title)
    if age_myr is None:
        axes[1, 1].legend(frameon=False, title="years after burst", ncol=2, fontsize=10, loc="lower right")    # the empty corner of the Z = 0.0002 panel
        fig.tight_layout()
    else:
        # ages on a colour bar instead of a legend
        axes[1, 1].legend(handles=axes[1, 1].get_legend_handles_labels()[0], frameon=False, fontsize=10, loc="lower right")   # AGN band only
        fig.tight_layout(rect=(0, 0, 0.92, 1))
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(min(age_myr), max(age_myr)), cmap=gcm.pan())
        fig.colorbar(sm, cax=fig.add_axes([0.93, 0.1, 0.015, 0.8]), label="age after burst [Myr]")
    fig.savefig(file_name, dpi=200)
    plt.show()

def plot_single_comparison(ax, age_myr):
    for Z, color in metal_color.items():
        wave = burst[Z][1]                  
        plot_single(ax, wave, to_g19_grid(wave, *pysb_single(Z, age_myr * 1e6)), color, label = rf'Z={Z}', ls="--")
        plot_single(ax, wave, to_g19_grid(wave, *sb99_single(Z, age_myr * 1e6)), color, label = None)

    for edge_eV in [13.6, 24.6, 54.4]:
        ax.axvline(edge_eV, color="gray", ls=":", lw=0.8)
    ax.set_xlim(10, 90)
    ax.set_ylim(-3, 0.5)
    ax.set_xlabel("energy [eV]")
    ax.text(0.95, 0.92, f"{age_myr} Myr", transform=ax.transAxes, ha="right", fontsize=14)


def plot_population_shape_stripped_only(ax, Z):
    time, wave, L_strip = burst[Z]          # G19 stripped stars of a 1e6 Msun burst
    for age_myr, color in age_color_stripped.items():
        row = np.argmin(np.abs(time - age_myr * 1e6))              # the stripped-star time closest to this age
        plot_single(ax, wave, L_strip[row], color, f"{age_myr} Myr")
    # ionization edges of H, He I and He II
    for edge_eV in [13.6, 24.6, 54.4]:
        ax.axvline(edge_eV, color="gray", ls=":", lw=0.8)
    ax.set_xlim(10, 90)
    ax.set_ylim(-3, 0.5)
    ax.text(0.95, 0.92, f"Z = {Z}", transform=ax.transAxes, ha="right", fontsize=14)


def plot_population_stripped_only(title, file_name):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    for ax, Z in zip(axes.ravel(), metallicities):
        plot_population_shape_stripped_only(ax, Z)
    for ax in axes[1, :]:
        ax.set_xlabel("energy [eV]")
    axes[1, 0].set_xticks(np.arange(10, 81, 10))      
    for ax in axes[:, 0]:
        ax.set_ylabel(r"log$_{10}$(L$_\nu$ / L$_\nu$(13.6 eV))")
    axes[1, 1].legend(frameon=False, title="years after burst", ncol=2, fontsize=10, loc="lower right")   
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(file_name, dpi=200)
    plt.show()
