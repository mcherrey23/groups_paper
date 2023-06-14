import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
import astropy.units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord, Distance
from astropy.cosmology import Planck15
from astropy.cosmology import WMAP9 as cosmo
import scipy.special as special
import matplotlib.backends.backend_pdf
from scipy.optimize import minimize
from scipy import integrate
import math


# GLOBAL VARIABLEs
C = const.c.to(u.km/u.s) # speed of light in km/s
h = cosmo.H(0).value/100 # the adimensional Hubble constant / 100


def FoF(df, D0 = 630, V0 = 400):
	
	df["dist_ang"] = Distance(unit=u.kpc, z = df["Z"]).value/((1+df["Z"])**2)
	df["dist_lum"] = Distance(unit=u.kpc, z = df["Z"]).value
	df["group_id"] = -df["ID"]

	i = 1
	ungrouped = df[df["group_id"]<=0]

	while len(ungrouped) != 0:
		print("iter: ", i, "      nb of ungrouped = ", len(ungrouped))
		df = df.sort_values(by=["group_id"]) #we order to have negative ID first
		df = search_companions(df, df.iloc[0], i, D0, V0)
		ungrouped = df[df["group_id"]<=0]
		i += 1

	return df


def search_companions(df, center, i, D0, V0):
	#print("search companions, center ID = ", float(center["ID"].values))

	ra = center["RA"]
	dec = center["DEC"]
	z = center["Z"]
	dist = center["dist_ang"]

	c1 = SkyCoord(ra*u.degree, dec*u.degree)
	c2 = SkyCoord(df["RA"]*u.degree, df["DEC"]*u.degree)
	sep = c1.separation(c2)
	df["DMEAN"] = (df["dist_ang"]+dist)/2
	df["D12"] = sep.radian*df["DMEAN"]
	df["V12"] = const.c.value*(z - df["Z"])/(1+z)/1000
	
	Dl = D0
	Vl = V0

	#Dl = 300 # kpc
	#Vl = 1e6 # 1000 km/s

	#print(df[["ID", "D12", "V12"]])

	f1 = df["D12"] <= Dl
	f2 = np.abs(df["V12"]) <= Vl


	# we find the new companions
	df["is_grouped"] = f1 & f2
	grouped = df[df["is_grouped"] == True] 
	print("Number of companions found: ", len(grouped))
	print(grouped[["ID", "group_id"]])

	# we assign the group id to the new companions
	idx = df.index[df["is_grouped"] == True].tolist() 
	df.loc[idx, "group_id"] = i 

	# Then we do a recursion on the new companions to find companions of companions
	if len(grouped) > 1: # if there is not only the center itself:
		for j, g in grouped.iterrows():
			if g["ID"] != center["ID"] and g["group_id"] != i:
				df = search_companions(df, g, i, D0, V0)
	return df
		
def build_stellar_masses(df1, df2):
	stellar_masses = pd.concat([df1, df2])
	for i in stellar_masses.columns:
		stellar_masses = stellar_masses.rename(columns = {i: i[:-1]})
	stellar_masses = stellar_masses.rename(columns={"id": "WHITE_ID"})
	stellar_masses.to_csv("stellar_masses.csv", index = False)
	print("N of stellar_masses = ", len(stellar_masses))
	return stellar_masses

def get_groups(df, N_min = 5, get_ungrouped = False):
	"""
	take as an input a dataframe containing all the galaxies and for each of them a group id indicating
	to which group it belong. It returns a dataframe describing the groups and their properties.
	If their is an column "outlier", indicating that some galaxies have been excluded thanks to the caustic method (for example), they will
	automatically be excluded from the groups.
	
	get_ungrouped: if True, all the single galaxies will also be included in the result Dataframe with a unique ID for each.
	"""
	if "outlier" in df.columns:
		df = df[df["outlier"] == False]

	if "Mvir_sigma2" not in df.columns:
		df["Mvir_sigma2"] = np.nan
	if "Mvir_sigma3" not in df.columns:
		df["Mvir_sigma3"] = np.nan
	if "Rvir_sigma2" not in df.columns:
		df["Rvir_sigma2"] = np.nan
	if "Rvir_sigma3" not in df.columns:
		df["Rvir_sigma3"] = np.nan
	
	df["mass10"] = 10**df["mass"]
	df["ra_mass"] = df["RA"]*df["mass10"]
	df["dec_mass"] = df["DEC"]*df["mass10"]
	df["z_mass"] = df["Z"]*df["mass10"] 

	gpk = df.groupby(by=["group_id"])
	gp_FIELD = np.array(gpk.field_id.max())
	gp_GRP = np.array(gpk.group_id.max())
	gp_N = np.array(gpk.Z.count())
	gp_Z_mean = np.array(gpk.Z.mean())
	gp_RA_mean = np.array(gpk.RA.mean())
	gp_DEC_mean = np.array(gpk.DEC.mean())
	gp_Mvir_sigma2 = np.array(gpk.Mvir_sigma2.mean())
	gp_Mvir_sigma3 = np.array(gpk.Mvir_sigma3.mean())
	gp_Rvir_sigma2 = np.array(gpk.Rvir_sigma2.mean())
	gp_Rvir_sigma3 = np.array(gpk.Rvir_sigma3.mean())
	#gp_RADIUS = np.array(gpk.Radius_kpc.mean())
	gp_b_min = np.array(gpk.B_KPC.min())
	gp_Mstar = np.array(gpk.mass10.sum())

	data = np.array([gp_FIELD, gp_GRP, gp_N, gp_Z_mean, gp_RA_mean, gp_DEC_mean, gp_b_min, \
					gp_Mvir_sigma2,  gp_Mvir_sigma3, gp_Rvir_sigma2, gp_Rvir_sigma3, gp_Mstar])
	G = pd.DataFrame(data = data.T, columns = ["field_id", "group_id", "N_gal", "mean_z", "mean_ra", "mean_dec", "b_min_kpc",\
											   "Mvir_sigma2", "Mvir_sigma3", "Rvir_sigma2", "Rvir_sigma3","Mstar"])
	G = G.astype({"N_gal":"int"})
	G = G.astype({"Mvir_sigma2":"float"})
	G = G.astype({"Mvir_sigma3": "float"})
	G = G.astype({"b_min_kpc":"float"})
	G = G.astype({"Mstar":"float"})
	
	G = G.sort_values(by=["N_gal"], ascending = False)
	f1 = G["N_gal"] >= N_min
	if get_ungrouped == False:
		f2 = G["group_id"] != -1
		G = G[f1 & f2]
	else:
		G = G[f1]

	G.reset_index(drop= True, inplace = True)

	return G


def calc_mass_ratio(G, R):
	G["m1_m2_ratio"] = -1
	for i, g in G.iterrows():
		r = R[R["group_id"] == g["group_id"]]
		r = r.sort_values(by="mass10", ascending = False)
		r = r[r["mass"].isna() == False]
        #print(r["mass"])
		if len(r) >= 2:
			r0 = r[:1]
			r1 = r[1:2]
			r0 = r0.squeeze()
			r1 = r1.squeeze()
			ratio = (10**r0["mass"])/(10**r1["mass"])
			#print("----------------------- RATIO = ", ratio)
		G.loc[i, "m1_m2_ratio"] = ratio
	return G


def recompute_centers(G, R):


	G["center_ra"] = np.nan
	G["center_dec"] = np.nan
	G["center_z"] = np.nan

	for i, g in G.iterrows():
		r = R[R["group_id"] == g["group_id"]]
		r = r.sort_values(by = "mass10", ascending = False)

        # we define the heaviest galaxy as the center of the group i
        # if the mass ratio between the 2 heaviest galaxies is above 1.5:
		center = r[:1].squeeze()
		#print(len(r))
		if g["m1_m2_ratio"] >= 1.5:
			G.loc[i, "center_ra"] = center["RA"]
			G.loc[i, "center_dec"] = center["DEC"]
			G.loc[i, "center_z"] = center["Z"]
		elif np.isnan(g["center_z"]):
			G.loc[i, "center_ra"] = g["mean_ra"]
			G.loc[i, "center_dec"] = g["mean_dec"]
			G.loc[i, "center_z"] = g["mean_z"]
	return G


def recompute_centers_new(G, R):


	G["center_ra"] = np.nan
	G["center_dec"] = np.nan
	G["center_z"] = np.nan

	for i, g in G.iterrows():
		r = R[R["group_id"] == g["group_id"]]
		r = r.sort_values(by = "mass10", ascending = False)

        # we define the heaviest galaxy as the center of the group i
        # if the mass ratio between the 2 heaviest galaxies is above 1.5:
		center = r[:1].squeeze()
		#print(len(r))
		if g["m1_m2_ratio"] >= 1.5:
			G.loc[i, "center_ra"] = center["RA"]
			G.loc[i, "center_dec"] = center["DEC"]
			G.loc[i, "center_z"] = center["Z"]
			G.loc[i, "center_ra_err"] = 0
			G.loc[i, "center_dec_err"] = 0
			G.loc[i, "center_z_err"] = 0
		elif np.isnan(g["center_z"]):
			r = r[~np.isnan(r["mass10"])]

			center_ra = (r["mass10"] * r["RA"]).sum() / (r["mass10"].sum())
			center_dec = (r["mass10"] * r["DEC"]).sum() / (r["mass10"].sum())
			center_z = (r["mass10"] * r["Z"]).sum() / (r["mass10"].sum())

			sum_mi = r["mass10"].sum()
			sum_ra = r["RA"].sum()
			sum_dec = r["DEC"].sum()
			sum_mirai = (r["RA"] * r["mass10"]).sum()
			sum_mideci = (r["DEC"] * r["mass10"]).sum()
			sum_mizi = (r["Z"] * r["mass10"]).sum()

			ra_err = (r["RA"] * sum_mi - sum_mirai) * r["mass10_err"]
			ra_err = ra_err / (sum_mi ** 2)
			ra_err_sum = ra_err.sum()
			dec_err = (r["DEC"] * sum_mi - sum_mideci) * r["mass10_err"]
			dec_err = dec_err / (sum_mi ** 2)
			dec_err_sum = dec_err.sum()
			z_err = (r["Z"] * sum_mi - sum_mizi) * r["mass10_err"]
			z_err = z_err / (sum_mi ** 2)
			z_err_sum = z_err.sum()

			G.loc[i, "center_ra"] = center_ra
			G.loc[i, "center_dec"] = center_dec
			G.loc[i, "center_z"] = center_z

			G.loc[i, "center_ra_err"] = np.abs(ra_err_sum)
			G.loc[i, "center_dec_err"] = np.abs(dec_err_sum)
			G.loc[i, "center_z_err"] = np.abs(z_err_sum)
	return G

def recompute_centers_2(G, R):


	G["center_ra"] = np.nan
	G["center_dec"] = np.nan
	G["center_z"] = np.nan

	G["center_ra_err"] = np.nan
	G["center_dec_err"] = np.nan
	G["center_z_err"] = np.nan

	for i, g in G.iterrows():
		r = R[R["group_id"] == g["group_id"]]
		r = r.sort_values(by = "mass10", ascending = False)
		r = r[~np.isnan(r["mass10"])]

		center_ra = (r["mass10"] * r["RA"]).sum() / (r["mass10"].sum())
		center_dec = (r["mass10"] * r["DEC"]).sum() / (r["mass10"].sum())
		center_z = (r["mass10"] * r["Z"]).sum() / (r["mass10"].sum())

		sum_mi = r["mass10"].sum()
		sum_ra = r["RA"].sum()
		sum_dec = r["DEC"].sum()
		sum_mirai = (r["RA"] * r["mass10"]).sum()
		sum_mideci = (r["DEC"] * r["mass10"]).sum()
		sum_mizi = (r["Z"] * r["mass10"]).sum()

		ra_err = (r["RA"] * sum_mi - sum_mirai) * r["mass10_err"]
		ra_err = ra_err / (sum_mi ** 2)
		ra_err_sum = ra_err.sum()
		dec_err = (r["DEC"] * sum_mi - sum_mideci) * r["mass10_err"]
		dec_err = dec_err / (sum_mi ** 2)
		dec_err_sum = dec_err.sum()
		z_err = (r["Z"] * sum_mi - sum_mizi) * r["mass10_err"]
		z_err = z_err / (sum_mi ** 2)
		z_err_sum = z_err.sum()

		G.loc[i, "center_ra"] = center_ra
		G.loc[i, "center_dec"] = center_dec
		G.loc[i, "center_z"] = center_z

		G.loc[i, "center_ra_err"] = np.abs(ra_err_sum)
		G.loc[i, "center_dec_err"] = np.abs(dec_err_sum)
		G.loc[i, "center_z_err"] = np.abs(z_err_sum)

	return G


def recompute_barycenters(G, R):


	G["barycenter_ra"] = np.nan
	G["barycenter_dec"] = np.nan
	G["barycenter_z"] = np.nan

	G["barycenter_ra_err"] = np.nan
	G["barycenter_dec_err"] = np.nan
	G["barycenter_z_err"] = np.nan

	for i, g in G.iterrows():
		r = R[R["group_id"] == g["group_id"]]
		r = r.sort_values(by = "mass10", ascending = False)
		r = r[~np.isnan(r["mass10"])]

		center_ra = (r["mass10"] * r["RA"]).sum() / (r["mass10"].sum())
		center_dec = (r["mass10"] * r["DEC"]).sum() / (r["mass10"].sum())
		center_z = (r["mass10"] * r["Z"]).sum() / (r["mass10"].sum())

		sum_mi = r["mass10"].sum()
		sum_ra = r["RA"].sum()
		sum_dec = r["DEC"].sum()
		sum_mirai = (r["RA"] * r["mass10"]).sum()
		sum_mideci = (r["DEC"] * r["mass10"]).sum()
		sum_mizi = (r["Z"] * r["mass10"]).sum()

		ra_err = (r["RA"] * sum_mi - sum_mirai) * r["mass10_err"]
		ra_err = ra_err / (sum_mi ** 2)
		ra_err_sum = ra_err.sum()
		dec_err = (r["DEC"] * sum_mi - sum_mideci) * r["mass10_err"]
		dec_err = dec_err / (sum_mi ** 2)
		dec_err_sum = dec_err.sum()
		z_err = (r["Z"] * sum_mi - sum_mizi) * r["mass10_err"]
		z_err = z_err / (sum_mi ** 2)
		z_err_sum = z_err.sum()

		G.loc[i, "barycenter_ra"] = center_ra
		G.loc[i, "barycenter_dec"] = center_dec
		G.loc[i, "barycenter_z"] = center_z

		G.loc[i, "barycenter_ra_err"] = ra_err_sum
		G.loc[i, "barycenter_dec_err"] = dec_err_sum
		G.loc[i, "barycenter_z_err"] = z_err_sum

	return G


def calc_mass(R, G, N_lim = 5):
	"""
	This function estimates the Mass of galaxy groups by using the speed distribution method.
	The Mass is given by the formula: M = 5*R*Vlos²/G
	It takes as input a DataFrame with all the galaxies and their corresponding group
	N_lim is the minimum number of galaxies per group to perform the mass computation.
	It returns a copy of the input DataFrame with a new column "M_vel_disp" and a new column "Radius_kpc"
	a sigma clipping can be applied in order to obtain a robust Mas estimation
	"""
	from scipy.stats import sigmaclip
	pd.options.mode.chained_assignment = None  # default='warn'
	
	# First all the different groups of the input Dataframe are identified:
	#groups = df["group"].unique()

	R["Mvir_group_sigma2"] = 0
	R["Rvir_group_sigma2"] = 0
	R["Mvir_group_sigma2_error"] = 0
	R["Rvir_group_sigma2_error"] = 0

	R["Mvir_group_sigma3"] = 0
	R["Rvir_group_sigma3"] = 0
	R["Mvir_group_sigma3_error"] = 0
	R["Rvir_group_sigma3_error"] = 0

	G["Mvir_sigma2"] = 0
	G["Rvir_sigma2"] = 0
	G["Mvir_sigma2_error"] = 0
	G["Rvir_sigma2_error"] = 0

	G["Mvir_sigma3"] = 0
	G["Rvir_sigma3"] = 0
	G["Mvir_sigma3_error"] = 0
	G["Rvir_sigma3_error"] = 0

	G["Rmax"] = 0
	G["sigma_v"] = 0
	G["sigma_v_error"] = 0

	# Then for each group:
	for i, g in G.iterrows():
		if g["group_id"] >= 0:
			F = R[R["group_id"] == g['group_id']] #we extract each group
			if "outlier" in F.columns:
				F = F[F["outlier"] == False]
			Z_mean = F["Z"].mean() # compute the average redshift
			RA_mean = F["RA"].mean()  # compute the average redshift
			DEC_mean = F["DEC"].mean()  # compute the average redshift
			F["delta Z"] = abs(F["Z"] - Z_mean) # the redshift difference for each galaxy

			RA = F["RA"] * u.degree
			DEC = F["DEC"] * u.degree
			c_gal = SkyCoord(RA, DEC)
			c_mean = SkyCoord(RA_mean*u.degree, DEC_mean*u.degree)
			sep_mean = c_gal.separation(c_mean)
			d_mean = sep_mean.radian * Distance(unit=u.m, z=Z_mean).value / ((1 + Z_mean) ** 2)
			F["d_mean"] = d_mean

			# Then for groups large enough, we can apply the speed distribution method
			if len(F) >= N_lim:

				dmax = 0
				for _, i in F.iterrows():
					for _, j in F.iterrows():
						Z = i["Z"]
						
						ra1 = i["RA"]*u.degree
						ra2 = j["RA"]*u.degree
						dec1 = i["DEC"]*u.degree
						dec2 = j["DEC"]*u.degree
						c1 = SkyCoord(ra1, dec1)
						c2 = SkyCoord(ra2, dec2)
						sep = c1.separation(c2)
						d = sep.radian*Distance(unit=u.m, z = Z).value/((1+Z)**2)						
						if d > dmax:
							dmax = d
				# We then assume that the radius is half the maximum distance
				r = dmax/2
				r = r*u.m
				r_kpc = r.to(u.kpc).value
				rrms = ((F["d_mean"]**2).sum()/(len(F["d_mean"])-1))**0.5
				rrms = rrms*u.m

				# The mass of the group is then estimated:
				V = (F["delta Z"])*const.c.value/(1+Z_mean) # we deduce the relative speed to the center of the group
				N = len(V)
				sigma = ((V**2).sum()/(N-1))**0.5
				sigma = sigma*(u.m/u.s) # in m/s

				Rvir_sigma3 = 2*sigma/(deltavir(Z_mean)**(1/2)*cosmo.H(Z_mean))
				Mvir_sigma3 = 4*sigma**3/(cosmo.H(Z_mean)*const.G*(deltavir(Z_mean))**(1/2))
				Mvir_sigma3 = Mvir_sigma3.to(u.kg)
				Rvir_sigma3 = Rvir_sigma3.to(u.kpc)
				Rvir_sigma3 = Rvir_sigma3.value
				Mvir_sigma3 = (Mvir_sigma3/const.M_sun).value

				Mvir_sigma2 = 5 * sigma ** 2 * rrms / const.G
				#Mvir_sigma2 = 6 * sigma ** 2 * rrms / const.G
				#Mvir_sigma2 = 6 * sigma ** 2 * r / const.G
				#Mvir_sigma2 = 3*sigma**2*r/const.G
				Mvir_sigma2 = Mvir_sigma2.to(u.kg)
				Mvir_sigma2 = (Mvir_sigma2/const.M_sun)
				Rvir_sigma2 = get_Rvir(Mvir_sigma2, Z_mean)


				# the errors:
				sigma_err = sigma_error(sigma, N) # in m/s
				#Mvir_sigma2_error = 12*sigma_v*r*sigma_err/(const.G * const.M_sun)
				Mvir_sigma2_error = 2 * Mvir_sigma2 * sigma_err / sigma # This is all 1-sigma error
				Mvir_sigma3_error = 3 * Mvir_sigma3 * sigma_err / sigma # This is all 1-sigma error
				Rvir_sigma2_error = Rvir_sigma2 * Mvir_sigma2_error / Mvir_sigma2 /3
				Rvir_sigma3_error = Rvir_sigma3 * Mvir_sigma3_error / Mvir_sigma3 /3


				# Finally the Resulting Dataframe is filled with the computed mass:
				R.loc[R["group_id"] == g["group_id"], "Mvir_group_sigma2"] = Mvir_sigma2
				R.loc[R["group_id"] == g["group_id"], "Rvir_group_sigma2"] = Rvir_sigma2
				R.loc[R["group_id"] == g["group_id"], "Mvir_group_sigma2_error"] = Mvir_sigma2_error
				R.loc[R["group_id"] == g["group_id"], "Rvir_group_sigma2_error"] = Rvir_sigma2_error

				R.loc[R["group_id"] == g["group_id"], "Mvir_group_sigma3"] = Mvir_sigma3
				R.loc[R["group_id"] == g["group_id"], "Rvir_group_sigma3"] = Rvir_sigma3
				R.loc[R["group_id"] == g["group_id"], "Mvir_group_sigma3_error"] = Mvir_sigma3_error
				R.loc[R["group_id"] == g["group_id"], "Rvir_group_sigma3_error"] = Rvir_sigma3_error

				G.loc[G["group_id"] == g["group_id"], "Mvir_sigma2"] = Mvir_sigma2
				G.loc[G["group_id"] == g["group_id"], "Rvir_sigma2"] = Rvir_sigma2
				G.loc[G["group_id"] == g["group_id"], "Mvir_sigma2_error"] = Mvir_sigma2_error
				G.loc[G["group_id"] == g["group_id"], "Rvir_sigma2_error"] = Rvir_sigma2_error

				G.loc[G["group_id"] == g["group_id"], "Mvir_sigma3"] = Mvir_sigma3
				G.loc[G["group_id"] == g["group_id"], "Rvir_sigma3"] = Rvir_sigma3
				G.loc[G["group_id"] == g["group_id"], "Mvir_sigma3_error"] = Mvir_sigma3_error
				G.loc[G["group_id"] == g["group_id"], "Rvir_sigma3_error"] = Rvir_sigma3_error

				G.loc[G["group_id"] == g["group_id"], "sigma_v"] = sigma/1000 # in km/s
				G.loc[G["group_id"] == g["group_id"], "sigma_v_error"] = sigma_err/1000  # in km/s
				G.loc[G["group_id"] == g["group_id"], "Rmax"] = r_kpc # in kpc

				#print("Mvir = ", np.log10(Mvir), " -------M_vel_disp = ", np.log10(M_vel_disp) ,"--------- Mold = ", np.log10(M_old))
	return R, G


def deltavir(z):
    x = cosmo.Om(z)-1
    return 18*np.pi**2 + 82*x - 39*x**2


# implement sigma clipping avec 3 ou 4 sigma.
# utiliser la relation Rs - rhoO

def nfw_rho(r, rho0, Rs):
	"""
	return the NFW density at a given radius. r and Rs must be in the same unit
	"""
	K = r/Rs
	D = K*(1+K)**2
	rho = rho0/D
	return rho

def nfw_cumsum(Rmax, rho0, Rs):
	"""
	return the cumulated mass of a NFW profile within a given radius.
	Rmax and Rs must be given in kpc
	the resulting mass is given in solar masses
	"""
	kpc_to_m = 1*u.kpc.to(u.m)
	c = Rmax/Rs
	M = 4*np.pi*rho0*(Rs)**3*(np.log(1+c)-c/(1+c))
	return M

def get_esc_v(M_inner, R):
	"""
	return the escape velocity for a given mass distribution (assumed spherically distributed) at a radius R
	The velocity is given in km/s
	"""
	kpc_to_m = 1*u.kpc.to(u.m)
	M_inner = (M_inner*u.Msun).to(u.kg).value
	R = R*kpc_to_m
	v = np.sqrt(2*const.G.value*M_inner/R)
	v = v*u.m/u.s
	return v.to(u.km/u.s)


def get_vir(df_group,rho0, Rs):
	"""
	return the Virial radius and virial mass for a NFW profile
	"""
	kpc_to_m = 1*u.kpc.to(u.m)
	R = np.linspace(1,3000, num = 10000 )
	z_mean = df_group["Z"].mean()
	Hz = cosmo.H(z_mean)
	rho200 = 200*3*Hz**2/(8*np.pi*const.G)
	rho200 = rho200.to(u.kg/u.m**3).value
	rho = nfw_rho(R*kpc_to_m, rho0, Rs*kpc_to_m)
	crit = rho < rho200
	Rvir = R[crit].min()
	Rvir = Rvir*u.kpc
	#print("R virial: ", np.round(Rvir))
	Mvir = nfw_cumsum(Rvir.value, rho0, Rs)
	#print("M virial: ", "{:e}".format(Mvir))
	return Rvir, Mvir
	
def get_Rvir(Mvir, z):
	"""
	return the virial radius for a NFW located at redshift z and having a given virial mass.
	ref: https://arxiv.org/pdf/astro-ph/9710107.pdf
	"""
	x = cosmo.Om(z)-1
	deltac = 18*np.pi**2 + 82*x - 39*x**2

	Hz = cosmo.H(z) # The hubble parameter at z
	rhoc = cosmo.critical_density(z)
	rhoc = rhoc.to(u.Msun/u.kpc**3).value
	#Rvir_old = (3*Mvir/(4*np.pi*178*rhoc))**(1/3) # the virial radius
	Rvir = (3*Mvir/(4*np.pi*deltac*rhoc))**(1/3) # the virial radius
	#print("Rvir old= ", Rvir_old, "  Rvir new = ", Rvir)
	return Rvir


def get_sigma_Rvir(Rvir, Mvir, sigma_Mvir):
	"""
	return the virial radius for a NFW located at redshift z and having a given virial mass.
	ref: https://arxiv.org/pdf/astro-ph/9710107.pdf
	"""
	#x = cosmo.Om(z)-1
	#deltac = 18*np.pi**2 + 82*x - 39*x**2

	#Hz = cosmo.H(z) # The hubble parameter at z
	#rhoc = cosmo.critical_density(z)
	#rhoc = rhoc.to(u.Msun/u.kpc**3).value
	#Rvir_old = (3*Mvir/(4*np.pi*178*rhoc))**(1/3) # the virial radius
	#Rvir = (3*Mvir/(4*np.pi*deltac*rhoc))**(1/3) # the virial radius

	#sigma_Rvir = Mvir**(-2/3)*(3/(4*np.pi*deltac*rhoc))**(1/3)/3
	sigma_Rvir = sigma_Mvir*Rvir/Mvir/3
	#print("Rvir old= ", Rvir_old, "  Rvir new = ", Rvir)
	return sigma_Rvir

def get_nfw_param(Mvir, z):
	"""
	From the Virial Mass and the redshift, this function get the NFW profile parameters rhoO and Rs.
	For that it uses the c - M relation described in Correa et al. 2018
	"""
	
	Rvir = get_Rvir(Mvir, z)
	c = Correa(Mvir, z)
	Rs = Rvir/c
	rho0 = Mvir/(4*np.pi*(Rs**3)*(np.log(1 + c) - c/(1 + c)))
	

	#print("rho0: ", rho0)
	#print("Rs: ", Rs)
	#print("c: ", c)
	#print("Rvir: ", Rvir)
	return rho0, Rs
	

def Correa(Mvir, z):
	"""
	compute the concentration of a NFW profile according to Correa et al. 2015
	"""
	a = 1.62774 - 0.2458*(1 + z) + 0.01716*(1 + z)**2
	b = 1.66079 + 0.00359*(1 + z) - 1.6901*(1 + z)**0.00417
	g = -0.02049 + 0.0253*(1 + z)**(-0.1044)
	log10_c = a + b*np.log10(Mvir)*(1 + g*(np.log10(Mvir))**2)
	
	return 10**log10_c


def match_qso(df, fields_info):
	"""
	match a dataframe with the dataframe containing the position of the quasar objects
	"""


	fields_info = fields_info.rename({'PSF': 'PSF_qso', 'Comments': 'Comments_qso', 'depth': "depth_qso", \
						   'ebv_sandf': 'ebv_sandf_qso', 'ebv_planck': 'ebv_planck_qso', \
							'ebv_sfd': 'ebv_sfd_qso', "HST": "HST_qso", 'rmag': 'rmag_qso' }, axis='columns')
	qso_sub = fields_info[["field_id", 'EXPTIME(s)','PSF_qso',\
					'Comments_qso', 'zqso_sdss', 'depth_qso',\
	   'ebv_sfd_qso', 'ebv_sandf_qso', 'ebv_planck_qso', 'HST_qso', 'rmag_qso']]


	df = df.merge(qso_sub, on = "field_id", how = "left")
	
	ra = fields_info["ra"]
	dec = fields_info["dec"]

	c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
	fields_info["ra_qso"] = c.ra.value
	fields_info["dec_qso"] = c.dec.value
	
	ra = []
	dec = []
	for i, g in df.iterrows():
		f = g["field_id"]
		qso = fields_info[fields_info["field_id"] == f]
		ra.append(qso["ra_qso"])
		dec.append(qso["dec_qso"])

	ra_qso = np.array(ra)
	dec_qso = np.array(dec)
	#print(ra_qso)
	df["ra_qso"] = ra_qso
	df["dec_qso"] = dec_qso
	deg_to_rad = (1*u.degree).to(u.radian).value
	
	#ra1 = df["mean_ra"]*u.degree
	#ra2 = df["ra_qso"]*u.degree
	#dec1 = df["mean_dec"]*u.degree
	#dec2 = df["dec_qso"]*u.degree
	#c1 = SkyCoord(ra1, dec1)
	#c2 = SkyCoord(ra2, dec2)
	#sep = c1.separation(c2)
	#df["dist"] = Distance(unit=u.kpc, z = df["mean_z"]).value/((1+df["mean_z"])**2)
	#df["b_center_kpc"] = sep.radian*df["dist"]
	#df = df.astype({"b_center_kpc":"float"})
	
	return df
	
def match_absorptions(G1, Abs, dv = 1e6):
	"""
	match the Abs dataframe describing the absorptions with the G dataframe describing the groups.
	
	dv: maximum velocity difference used to appariate an absorption with a group
	"""

	pd.options.mode.chained_assignment = None
	G = G1.copy()
	#G_energy["field"] = G_energy["field_id"].str.slice(2,12)
	G["REW_2796"] = 0
	G["sig_REW_2796"] = 0
	G["z_absorption"] = 0
	G["z_absorption_dist"] = 0
	for j,i in G.iterrows():
		T = Abs[Abs["field_name"] == i["field_id"]]
		T["v_dist"] = abs(T["z_abs"] - i["center_z"])*const.c.value/(1+i["center_z"])
		min_v_dist = T["v_dist"].min()
		abs_min = T[T["v_dist"] == min_v_dist]
		idx = G.index[G["group_id"] == i["group_id"]].to_list()[0]
		#print(abs_min, "    ", idx)
		G.loc[idx, "REW_2796" ] = abs_min["REW_2796"].mean()
		try:
			G.loc[idx, "sig_REW_2796"] = abs_min["sig_REW_2796"].mean()
		except:
			print("error")
		G.loc[idx, "z_absorption"] = abs_min["z_abs"].mean()
		G.loc[idx, "vel_absorption_dist"] = min_v_dist
		G.loc[idx, "N100_abs"] = abs_min["N100_abs"].max()
		#print("================")
		#print(j, abs_min["REW_2796"],  abs_min["sig_REW_2796"], abs_min["z_abs"], min_z_dist)
		#print(idx)
		#G.at[idx, "REW_2796"] = abs_min["REW_2796"]

	G["bool_absorption"] = 0
	idx = G.index[G["vel_absorption_dist"] < dv].to_list()
	G.loc[idx, "bool_absorption" ] = 1

	# Finally we use this boolean to set to 0 the groups with an absorption too far:
	G["REW_2796"] = G["REW_2796"]*G["bool_absorption"]
	G["sig_REW_2796"] = G["sig_REW_2796"]*G["bool_absorption"]
	G["z_absorption"] = G["z_absorption"]*G["bool_absorption"]
	G["vel_absorption_dist"] = G["vel_absorption_dist"]*G["bool_absorption"]

	return G

def match_absorptions_isolated_galaxies(R, Abs, dv = 1e6):
	"""
	match the Abs dataframe describing the absorptions with the R dataframe containing the galaxies.
	
	dv: maximum velocity difference used to appariate an absorption with a galaxy
	"""

	pd.options.mode.chained_assignment = None
	
	R["REW_2796"] = 0
	R["sig_REW_2796"] = 0
	R["z_absorption"] = 0
	R["z_absorption_dist"] = 0
	for j,i in R.iterrows():
		T = Abs[Abs["field_name"] == i["field_id"]]
		T["v_dist"] = abs(T["z_abs"] - i["Z"])*const.c.value/(1+i["Z"])
		min_v_dist = T["v_dist"].min()
		abs_min = T[T["v_dist"] == min_v_dist]
		idx = R.index[R["ID"]== i["ID"]].to_list()[0]
		R.loc[idx, "REW_2796" ] = abs_min["REW_2796"].mean()
		try:
			R.loc[idx, "sig_REW_2796"] = abs_min["sig_REW_2796"].mean()
		except:
			print("error")
		R.loc[idx, "z_absorption"] = abs_min["z_abs"].mean()
		R.loc[idx, "vel_absorption_dist"] = min_v_dist

	R["bool_absorption"] = 0
	idx = R.index[R["vel_absorption_dist"] < dv].to_list()
	R.loc[idx, "bool_absorption" ] = 1

	R["REW_2796"] = R["bool_absorption"]*R["REW_2796"]
	R["sig_REW_2796"] = R["sig_REW_2796"]*R["bool_absorption"]
	R["z_absorption"] = R["z_absorption"]*R["bool_absorption"]
	R["vel_absorption_dist"] = R["vel_absorption_dist"]*R["bool_absorption"]

	return R


def get_N100_abs(Abs, R, dv = 1e6):
	"""
	Get the number of galaxies in a 100kpc radius around the QSO LOS for each absorber.
	
	dv: maximum velocity difference between the absorption and the galaxies taken into account.
	"""
	N100 = []
	for i, absorption in Abs.iterrows():
		f1 = np.abs(R["Z"] - absorption["z_abs"])*const.c.value/(1+absorption["z_abs"])<dv
		f2 = R["field_id"] == absorption["field_name"]
		f3 = R["B_KPC"]<100
		F = R[f1 & f2 & f3]
		N100.append(len(F))
	Abs["N100_abs"] = np.array(N100)
	return Abs
	
def get_Nxxx_abs(Abs, R, bmax = 100, dv = 0.5e6):
	"""
	Get the number of galaxies in a 100kpc radius around the QSO LOS for each absorber.
	
	dv: maximum velocity difference between the absorption and the galaxies taken into account.
	"""
	Nxxx = []
	bmin = []
	Abs2 = Abs.copy()
	#print("tototototo ", dv)
	for i, absorption in Abs2.iterrows():
		f1 = np.abs(R["Z"] - absorption["z_abs"])*const.c.value/(1+absorption["z_abs"])<dv
		f2 = R["field_id"] == absorption["field_name"]
		f3 = R["B_KPC"]< bmax
		F = R[f1 & f2 & f3]
		F["dv_abs"] = np.abs(F["Z"] - absorption["z_abs"])*const.c.value/(1+absorption["z_abs"])
		#print(absorption["field_name"], absorption["z_abs"], F[["ID","Z", "dv_abs"]])
		Nxxx.append(len(F))
		bmin.append(np.min(F["B_KPC"]))
		#print(Nxxx)
	colname_N= "N"+str(bmax)+"_abs"
	colname_b = "bmin"+str(bmax)+"_abs"
	Abs2[colname_N] = np.array(Nxxx)
	Abs2[colname_b] = np.array(bmin)
	return Abs2

def get_Nxxx_abs_test(Abs, R, bmax = 100, dv = 1e6):
	"""
	Get the number of galaxies in a xxx kpc radius and at an offset of v_offset around the QSO LOS for each absorber.
	Meaning that we look at random points in space (cf 1 point correlation function)
	
	dv: maximum velocity difference between the absorption and the galaxies taken into account.
	"""
	v_offset = 15e6
	
	Nxxx = []
	for i, absorption in Abs.iterrows():
		z_offset = np.abs(v_offset)*(1+absorption["z_abs"])/const.c.value
		f1 = np.abs(R["Z"] - absorption["z_abs"]+z_offset)*const.c.value/(1+absorption["z_abs"])<dv
		f2 = R["field_id"] == absorption["field_name"]
		f3 = R["B_KPC"]< bmax
		F = R[f1 & f2 & f3]
		Nxxx.append(len(F))
	colname = "N"+str(bmax)+"_abs_random"
	Abs[colname] = np.array(Nxxx)
	return Abs
	
def get_N100_LOS(G, R, dv = 1e6):
	"""
	Get the number of galaxies in a 100kpc radius around the QSO LOS for each group of galaxies.
	"""
	N100 = []
	for i, g in G.iterrows():
		f1 = np.abs(R["Z"] - g["mean_z"])*const.c.value/(1+g["mean_z"])<dv
		f2 = R["field_id"] == g["field_id"]
		f3 = R["B_KPC"]<100
		F = R[f1 & f2 & f3]
		N100.append(len(F))
	G["N100_los"] = np.array(N100)
	return G 
	
def get_Nxxx_neighb(R, radius = 100, dv = 1e6):
	"""
	Get the number of neighbour galaxies within a given radius for each galaxy.
	
	dv: maximum velocity difference taken for taking into account a galaxy
	"""
	label = "N"+str(radius)+"_neighb"
	N_neighb = []
	for i, gal in R.iterrows():
		f1 = np.abs(R["Z"] - gal["Z"])*const.c.value/(1+gal["Z"])<dv
		f2 = R["field_id"] == gal["field_id"]
		#f3 = R["ID"] != gal["ID"]
		#F = R[f1 & f2 & f3]
		F = R[f1 & f2]
		#print(F)
		c1 = SkyCoord(gal["RA"]*u.degree, gal["DEC"]*u.degree)
		c2 = SkyCoord(F["RA"]*u.degree, F["DEC"]*u.degree)
		sep = c1.separation(c2)
		F["dist"] = Distance(unit=u.kpc, z = gal["Z"]).value/((1+gal["Z"])**2)
		F["neighb_dist"] = sep.radian*F["dist"]
		F_filt = F[F["neighb_dist"]<radius]
		N_neighb.append(len(F_filt))
	R[label] = np.array(N_neighb)-1
	return R

def get_Nxxx_all_groups(R,G, radius = 100):
	"""
	Get the number of galaxies within a given radius around the center of each group.
	
	Dv: maximum velocity difference for taking into account a galaxy
	"""
	label = "N"+str(radius)
	G[label] = np.nan
	N = []
	for i, g in G.iterrows():
		group_id = g["group_id"]
		field_id = g["field_id"]
		#f1 = R["field_id"] == field_id
		#f2 = np.abs(R["Z"] - g["mean_z"]) <= Dz
		f1 = R["group_id"] == group_id
		f2 = R["outlier"] == False
		df = R[f1 & f2]
		n =  get_Nxxx_group(df, g["center_ra"], g["center_dec"], g["center_z"], radius = radius)
		N.append(n)
	G[label] = np.array(N)
	return G


def get_Nxxx_Rvir_all_groups(R,G, Rvir_factor = 1.0):
	"""
	Get the number of galaxies within a given radius (equal to Rvir_factor times the virial raidus) around the center of each group.
	
	Dz: maximum redshift difference for taking into account a galaxy
	"""	
	label = "N"+str(Rvir_factor)+"_Rvir"
	G[label] = np.nan
	N = []
	for i, g in G.iterrows():
		group_id = g["group_id"]
		field_id = g["field_id"]
		#f1 = R["field_id"] == field_id
		#f2 = np.abs(R["Z"] - g["mean_z"]) <= Dz
		f1 = R["group_id"] == group_id
		f2 = R["outlier"] == False
		df = R[f1 & f2]
		radius = Rvir_factor*g["Rvir"]
		n =  get_Nxxx_group(df, g["center_ra"], g["center_dec"], g["center_z"], radius = radius)
		N.append(n)
	G[label] = np.array(N)
	return G
	
def get_Nxxx_group(df, ra_center, dec_center, z_center, radius = 100):
	"""
	Get the number of galaxies in a given radius from a central point.
	"""
	
	c1 = SkyCoord(ra_center*u.degree, dec_center*u.degree)
	
	# then we compute the distance to the center for each galaxy:
	count = 0
	for i, gal in df.iterrows():
		c2 = SkyCoord(gal["RA"]*u.degree, gal["DEC"]*u.degree)
		sep = c1.separation(c2)
		d = Distance(unit=u.kpc, z = z_center).value/((1+z_center)**2)
		r = sep.radian*d
		if r <= radius :
			count +=1

	return count 


def get_all_expected_group_density(G, Radius = 100, dv = 1e6, rhoU = 0.1e-9):
	"""
	Compute the number of galaxies that would be expected within 100 kpc of each group, given its estimated mass.
	the uncertainty is also computed
	"""
	N100_r0 = []
	N100_exp = []
	N100_exp_sigma = []
	N100_exp_sigma_poiss = []
	for i, g in G.iterrows():
		r0 = get_r0(g["M_vel_disp"])
		print("R0 = ", r0)
		N, N_sigma, N_sigma_poiss = get_expected_group_density(g["mean_z"], r0, Radius = Radius, dv = dv, rhoU = rhoU)
		N100_exp.append(N)
		#N100_exp_sigma.append(N**0.5) # Here we just consider a poissonian variance. Must be refined with cluster variance (cf Bouche et al. 2005)
		N100_exp_sigma.append(N_sigma)
		N100_exp_sigma_poiss.append(N_sigma_poiss)
		N100_r0.append(r0)
	label = "N"+str(Radius)
	G[label+"_r0"] = np.array(N100_r0)
	G[label+"_exp"] = np.array(N100_exp)
	G[label+"_exp_sigma"] = np.array(N100_exp_sigma)
	G[label+"_exp_sigma_poiss"] = np.array(N100_exp_sigma_poiss)
	return G

def get_expected_group_density(Z, r0, Radius = 100, dv = 1e6, rhoU = 0.1e-9):
	"""
	Get the number of galaxies that we can expect in a group/cluster of parameter r0 within a given radius, in a given redshift slice
	(Outside of a group, r0 = 0) and located at redshift Z.
	In other words the 2 points correlation function is integrated over a cylinder along the LOS
	(the circular section is orthogonal to the LOS, and the depth of the cylinder is along the redshifts)
	"""
	from scipy.integrate import quad, dblquad
	
	H = cosmo.H(Z).to(u.m/u.kpc/u.s)
	#dz = dv*Z/const.c.value
	#rz = C.value*dz/H.value
	rz = dv/H.value
	print("Z = ", Z, "      rz = ", rz)
	#return dblquad(f, 0, Radius, lambda r: 0, lambda r: dz, args = (r0, 1.77, rhoU, H))
	N = quad(Adelberger, 0, Radius, args = (rz, r0, 1.77, rhoU))
	Nu = quad(Adelberger, 0, Radius, args = (rz, 0, 1.77, rhoU))
	N = N[0]
	Nu = Nu[0]
	
	# uncertainties
	gamma = 1.77
	J2 = 72/((3-gamma)*(4-gamma)*(6-gamma)*2**gamma)
	#J2 = 4.87
	Nexp = Nu
	Nobs = N
	V2pt = Nexp**2*((r0/Radius)**gamma)*J2
	Vsn = Nobs
	N_sigma = (Vsn + V2pt)**0.5
	N_sigma_poiss = Vsn**0.5
	print(N, Nu, V2pt, rz)
	
	return N, N_sigma, N_sigma_poiss 



def f(z, r, r0, gamma, rho, H):
	"""
	The function to integrate in order to obtain the number of galaxies in a group/cluster.
	It is assumed that the correlation function is of the form DN = rho(1+(r/r0)**(-gamma))
	with r0 depending on the mass/size of the cluster.
	"""
	return rho*(1 + ((r**2 + (C.value*z/H)**2)**0.5/r0)**(-gamma))*2*np.pi*r*C.value/H
	
def f0(z, r, r0, gamma, rho, Z):
	"""
	The function to integrate in order to obtain the number of galaxies "in the field".
	"""
	C = const.c.to(u.km/u.s)
	H = cosmo.H(Z).to(u.km/u.kpc/u.s)
	# Then the Number of galaxies is simply the mean density rho times 2.pi.r.dr_orthog.dr_parall
	# with dr_parall = C.dz/H 
	return rho*2*np.pi*r*C.value/H.value
	
	
	
def get_r0(M):
	"""
	Use few data points from #ref_to_add and interpolate to obtain an approximative value of the parameter r0 that is used in the 2 point correlation function
	NOTE:: M must be given in solar Masses
	"""
	from scipy.interpolate import interp1d
	
	M_lst = [9, 11, 12, 13, 14, 15, 16]
	r0_lst = [1, 3, 4.5, 7, 14.5, 25, 40]
	f = interp1d(M_lst, r0_lst)
	
	logM = np.log10(M)
	#print(logM)

	# we multiply by h because r0 is in Mpc/h
	try: 
		return f(logM)*1e3*h
	except:
		#print("Mass value out of interpolation bound: ", logM)
		return 0


def Adelberger(r, rz, r0, gamma, rho):
	"""
	compute the expected number of galaxies at a radius r from an overdensity (1+ (r/r0)^-gamma) and in a redshift range [-dz; dz].
	Taken from Adelberger et al. 2003
	"""
	x = rz**2/(rz**2 + r**2)
	w = r0**gamma*r**(1-gamma)*special.beta(0.5, (gamma - 1)/2)*special.betainc(0.5, (gamma - 1)/2, x)/(2*rz)
	n_exp = rho*2*np.pi*r*rz*2 # the last factor 2 is to take into account that the cylinder span on [-rz, +rz]
	return n_exp*(w+1)



def plot_group_absorption(i, R, G):
	"""
	plot the phase space for the given group and the closest absorption.
	"""
	
	grp_lst = G["group_id"].unique()
	g = grp_lst[i]
	group_df = G[G["group_id"] == g]
	df = R[R["group"] == g]
	df_filt = df[df["outlier"] == False]

	C = const.c.to('km/s').value
	deg_to_rad = 1*u.degree.to(u.radian)


	v_abs = float((group_df["mean_z"] - group_df["z_absorption"])*const.c.value/(1+group_df["z_absorption"])/1000)
	print(group_df["Rvir"])
	
	plt.figure(figsize = (15,4))
	#plt.subplot(121)
	plt.scatter(df["r_to_gcenter"], df["vlos_to_gcenter"], label = "all")
	plt.scatter(df_filt["r_to_gcenter"], df_filt["vlos_to_gcenter"], c = "red", label = "filt")
	plt.axvline(float(group_df["Rvir"]), ls = "--", c = "red", label = "R virial")
	plt.axhline(0, c = "black")
	if int(group_df["bool_absorption"]) == 1:
		plt.axhline(v_abs, c = "purple", label = "absorption")
	plt.xlim((0,(np.max(df["r_to_gcenter"]))*2))
	#plt.ylim((-1500,1500))
	plt.xlabel("r to group center (kpc)", size = 12)
	plt.ylabel("v (km/s)", size = 12)
	plt.legend()
	
	return

def plot_group_absorption2(i, R, G, Abs, dv = 1e6):
	"""
	plot the phase space for the given group and the closest absorption. Multiple absorptions are tolerated.
	"""
	#dz = 3e6/const.c.value
	grp_lst = G["group_id"].unique()
	g = grp_lst[i]
	
	
	group_df = G[G["group_id"] == g]
	ID = group_df["ID"].values[0]
	field_id = group_df["field_id"].values[0]
	df = R[R["group"] == g]
	df_filt = df[df["outlier"] == False]
	Abs_filt = Abs[Abs["field_name"] == field_id]
	Abs_filt["v_diff"] = np.abs(Abs_filt["z_abs"] -group_df["center_z"].mean())*const.c.value/(1+group_df["center_z"].mean())
	f1 = np.abs(Abs_filt["z_abs"] -group_df["center_z"])*const.c.value/(1+Abs_filt["z_abs"]) <= dv
	Abs_filt = Abs_filt[np.abs(Abs_filt["z_abs"] -group_df["center_z"].mean())*const.c.value/(1+Abs_filt["z_abs"]) <= dv]
	Abs_filt["v_abs"] = (Abs_filt["z_abs"] - group_df["center_z"].mean())*const.c.to(u.km/u.s).value/(1+Abs_filt["z_abs"])
	#print(Abs_filt)
	center_z = group_df["center_z"].values[0]
	
	print("group id: ", g, " field :", field_id)
	
	file_name = "uves_data/j"+field_id[1:]+".dat"
	uves = pd.read_csv(file_name, delim_whitespace = True, names = ["lambda", "flux_norm", "flux", "C", "continuum"], index_col = False)
	uves["z_abs"] = uves["lambda"]/2796.352 - 1
	uves["delta_v"] = (uves["z_abs"] - center_z)*const.c.to(u.km/u.s).value/(1+center_z)


	plt.figure(figsize = (15,4))
	plt.subplot(121)
	plt.scatter(df["B_KPC"], df["vlos_to_gcenter"], label = ID)
	plt.scatter(df_filt["B_KPC"], df_filt["vlos_to_gcenter"], c = "red")
	#plt.axvline(float(group_df["Rvir_recomputed"]), ls = "--", c = "red", label = "R virial")
	plt.axhline(0, c = "black")
	for i, a in Abs_filt.iterrows():
		plt.axhline(a["v_abs"], c = "lime")
	plt.xlim((0,(np.max(df["B_KPC"]))*1.2))
	#plt.ylim((-1500,1500))
	plt.xlabel("b kpc)", size = 12)
	plt.ylabel("v (km/s)", size = 12)
	plt.legend()


	plt.subplot(122)
	for l in df["vlos_to_gcenter"]:
		plt.axvline(l, c = "red", linewidth = 0.8)
	for i, a in Abs_filt.iterrows():
		plt.axvline(a["v_abs"], c = "lime", label = "absorption")
	plt.plot(uves["delta_v"], uves["flux_norm"])
	#plt.plot(uves["lambda"], uves["flux_norm"])
	plt.ylim((-0.2,1.5))
	plt.xlim((-1000,1000))

	#print(uves.head(50))
	
	return


def get_nth_neighb_dist(df, ra_center, dec_center, z_center, n = 1):
	"""
	Get the distance of the nth neighbour galaxy relatively to a given position
	"""
	c1 = SkyCoord(df["RA"]*u.degree, df["DEC"]*u.degree)
	c2 = SkyCoord(ra_center*u.degree, dec_center*u.degree)
	sep = c1.separation(c2)
	d = Distance(unit=u.kpc, z = z_center).value/((1+ z_center)**2)
	r = sep.radian*d
	r.sort()
	
	return r[n-1]


def get_all_nth_neighb_dist_from_qso(G, R, n=1):
	"""
	Get the distance of the nth neighbour galaxy relatively to the QSO line for all groups
	
	WARNING: outliers (removed with the caustics method) are taken into account !!!
	"""
	
	label = "d"+str(n)+"_qso"
	G[label] = np.nan
	D = []
	for i, g in G.iterrows():
		grpid = g["group_id"]
		df = R[R["group"] == grpid]
		ra_qso = g["ra_qso"]
		dec_qso = g["dec_qso"]
		z_center = g["mean_z"]
		#print(df)
		d = np.nan
		try:
			d = get_nth_neighb_dist(df, ra_qso, dec_qso, z_center, n = n)
			#print(d)
		except: 
			print("Neigb dist calculation failed !")
		D.append(d)
	G[label] = np.array(D)
	return G

def plot_group(grpid, R, G, col = "mass"):
	r_select = R[R["group"] == grpid]
	g_select = G[G["group_id"] == grpid]
	plt.figure(figsize = (14,5))
	plt.subplot(121)
	plt.scatter(r_select["RA"], r_select["DEC"], c = "silver")
	plt.scatter(r_select["RA"], r_select["DEC"], c = r_select[col])
	plt.colorbar(label = col)
	plt.scatter(g_select["ra_qso"], g_select["dec_qso"], marker = "*", color = "red", s = 50)
	plt.scatter(g_select["mean_ra"], g_select["mean_dec"], color = "orange", s = 50)
	plt.annotate("g center", (g_select["mean_ra"], g_select["mean_dec"]))
	plt.xlabel("ra", size = 12)
	plt.ylabel("dec", size = 12)

def create_id(df, sort_col):
	r = df.copy()
	r["ID"] = 0
	r.sort_values(by = [sort_col], ascending = False, inplace = True)
	r.reset_index(inplace = True)
	#print(r)
	for i in range(len(r)):
		r.at[i, "ID"] = i+1
	return r


def fc_cum(rew, b, rew_lim, nbins = 10, blim = (0,250), plot = True, xlog = False, ylog = False, xset = False, xlim = (0,0), yset = False, ylim = (0,0)):
	"""
	compute the differential and cumulative covering fraction. Also offers the possibility to plot the covering fraction.
	"""


	bmin = blim[0]
	bmax = blim[1]
	binsedge = np.arange(bmin, bmax, (bmax-bmin)/(nbins))
	binscenter = (binsedge[1:] + binsedge[:-1])/2
	binswidth = binsedge[2] - binsedge[1]
	b_abs = b[rew >= rew_lim]
	b_noabs = b[rew < rew_lim]
	
	#print(binswidth)
	fcdiff = []
	fccum = []
	for i in binscenter:
		f1 = b >= i - binswidth/2
		f2 = b < i + binswidth/2
		f3 = rew >= rew_lim
		rew_filt1 = rew[f1 & f2 & f3]
		rew_filt2 = rew[f2 & f3]
		rew_filt3 = rew[f1 & f2]
		rew_filt4 = rew[f2]
		try:
			fcdiff.append(len(rew_filt1)/len(rew_filt3))
		except:
			fcdiff.append(np.nan)
		fccum.append(len(rew_filt2)/len(rew_filt4))
		
	print(len(binscenter), len(fccum),len(fcdiff) )
	
	if plot == True:
		plt.figure(figsize = (10,5))
		plt.scatter(b_abs, np.ones(len(b_abs)), marker = "+", c = "black")
		plt.scatter(b_noabs, np.zeros(len(b_noabs)), marker = "+", c = "black")
		plt.plot(binscenter, fcdiff, label = 'fc diff', marker = "s")
		plt.plot(binscenter, fccum, label = 'fc cum', marker = "s")
		plt.legend()
		plt.grid()
		if xlog:
			plt.xscale("log")
		if ylog:
			plt.yscale("log")
	return fcdiff, fccum


def plot_groups(R, G, Nmin = 5, save = False,  filename = "none"):
	if save == True:
		if filename == "none":
			from datetime import datetime
			d = datetime.now()
			name = str(d.year)+str(d.month)+str(d.day)+"_groups.pdf"
		else:
			name = filename
		pdf = matplotlib.backends.backend_pdf.PdfPages(name)

	G_filt = G[G["N_gal"]>= Nmin]

	for i, g in G_filt.iterrows():
		#try:
		df = R[R["group"] == g["group_id"]]	
		z_mean = g["mean_z"]
		ra_mean = g["mean_ra"]*u.degree
		dec_mean = g["mean_dec"]*u.degree

		z_bary = g["mean_z"]
		ra_bary = g["mean_ra"]*u.degree
		dec_bary = g["mean_dec"]*u.degree

		if ~np.isnan(g["center_z"]):
			z_center = g["center_z"]
			ra_center =  g["center_ra"]*u.degree
			dec_center = g["center_dec"]*u.degree
		else: 
			z_center = g["mean_z"]
			ra_center =  g["mean_ra"]*u.degree
			dec_center = g["mean_dec"]*u.degree

		ra_qso = (df["ra_qso"].mean())*u.degree
		dec_qso = (df["dec_qso"].mean())*u.degree
		rho0, Rs = get_nfw_param(g["Mvir"], z_mean)
		r = np.linspace(1,3000, num = 10000 )
		M = nfw_cumsum(r, rho0, Rs)
		V = get_esc_v(M, r)

		C = const.c.to('km/s').value
		deg_to_rad = 1*u.degree.to(u.radian)
		
		ra2 = df["RA"]*u.degree
		dec2 = df["DEC"]*u.degree
		c1 = SkyCoord(ra_center, dec_center)
		c2 = SkyCoord(ra2, dec2)
		sep = c1.separation(c2)
		df["r_to_gcenter"] = sep.radian*Distance(unit=u.kpc, z = z_center).value/((1+z_center)**2)
		df["vlos_to_gcenter"] = (df["Z"]-z_center)*C/(1+z_center)
		
		Rvir = g["Rvir_recomputed"]

		ra = df["RA"]
		dec = df["DEC"]
		vlos = df["vlos_to_gcenter"]
		rcenter = df["r_to_gcenter"]

		df_filt = df[df["outlier"] == False]
		df_out = df[df["outlier"] == True]


		ra = np.array(ra)
		dec = np.array(dec)
		vlos = np.array(vlos)
		rcenter = np.array(rcenter)
		
		fig = plt.figure(figsize = (15,4))
		title = "Grp"+str(g["ID"])\
				+":   log($Mvir/M_{\odot}$) = "\
				+str(round(np.log10(g["Mvir"]),1))\
				+",     Rvir = "+str(round(g["Rvir_recomputed"]))\
				+" kpc" \
				+", field: "+str(g["field_id"])\
				+", z = " +str(np.round(g["center_z"], 2))
		plt.suptitle(title, fontweight = "bold")
		#---SUBPLOT 1 ----------------------------------------------------------------
		plt.subplot(121)
		plt.scatter(df["r_to_gcenter"], df["vlos_to_gcenter"], \
					c = df["Psat"], edgecolor = 'grey', marker = "s", s = 50, vmin = 0, vmax = 1)
		if "Psat" in df.columns:
			plt.scatter(df["r_to_gcenter"], df["vlos_to_gcenter"], \
					c = df["Psat"], edgecolor = 'grey', vmin = 0, vmax = 1, s = df['mass10']/1e8 + 80)
			cbar = plt.colorbar(label =	 "Psat")
		else:
			plt.scatter(df["r_to_gcenter"], df["vlos_to_gcenter"], \
					c = -df["m_abs_r"], edgecolor = 'grey', s = df['mass10']/1e8 + 80)
			cbar = plt.colorbar(label =	 "$- m_r$ absolute")
		plt.scatter(df_out["r_to_gcenter"], df_out["vlos_to_gcenter"], s = 250, facecolors='none', edgecolors='r')
		plt.axvline(Rvir, ls = "--", c = "green", label = "R virial")
		plt.plot(r, V, c = "black", label = "caustic")
		plt.plot(r, -V, c = "black")
		try:
			plt.xlim((0,(np.max(df["r_to_gcenter"]))*2))
		except: 
			print("no center!")
		#plt.ylim((-1500,1500))
		plt.xlabel("r (kpc)")
		plt.ylabel("$\Delta v (km/s)$")
		plt.axhline(c = "grey")
		plt.legend()
		for i in range(len(df)):
			plt.annotate(str(i), (rcenter[i], vlos[i]))


		#---SUBPLOT 2-------------------------------------------------------------
		kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(z_mean)
		arcmin = 1/60
		plt.subplot(122)
		r100 = 100/kpc_per_arcmin.value/60
		plt.scatter(ra_qso.value, dec_qso.value, marker = "*", s = 100, c = "red")
		rectangle = plt.Rectangle((ra_qso.value - arcmin/2,dec_qso.value - arcmin/2), arcmin, arcmin, fc=None ,ec=None, lw = 0, fill = False)
		plt.gca().add_patch(rectangle)
		circle = plt.Circle((ra_qso.value,dec_qso.value),r100, fill = False,ec="green")
		plt.gca().add_patch(circle)
		plt.scatter(df["RA"],df["DEC"], \
					c = df["Psat"], edgecolor = 'grey', marker = "s", s = 50, vmin = 0, vmax = 1)
		if "Psat" in df.columns:
			plt.scatter(df["RA"],df["DEC"], \
						c = df["Psat"], edgecolor = 'grey', s = df['mass10']/1e8 + 80, vmin = 0, vmax = 1)
			plt.colorbar(label = "Psat")
		else:
			plt.scatter(df["RA"],df["DEC"], \
						c = -df["m_abs_r"], edgecolor = 'grey', s = df['mass10']/1e8 + 80)
			plt.colorbar(label = "$- m_r$ absolute")
		plt.xticks([])
		plt.yticks([])
		plt.scatter(df_out["RA"], df_out["DEC"], s=250, facecolors='none', edgecolors='r')
		plt.scatter(ra_qso.value, dec_qso.value, marker = "*", s = 100, c = "red")
		plt.scatter(ra_mean.value, dec_mean.value, marker = "x", s = 40, c = "magenta")
		plt.scatter(ra_center.value, dec_center.value, marker = "*", s = 60, c = "orange")
		plt.gca().set_aspect('equal', adjustable='box')
		for i in range(len(df)):
			plt.annotate(str(i), (ra[i], dec[i]))
		fig.tight_layout()

		if save == True:
			pdf.savefig(fig)

		#except Exception as e:  
		#	print(e)

	if save == True:
		pdf.close()

	return



def plot_groups_2(R, G, Nmin = 5, save = False,  filename = "none", dv = 1e6):
	if save == True:
		if filename == "none":
			from datetime import datetime
			d = datetime.now()
			name = str(d.year)+str(d.month)+str(d.day)+"_groups.pdf"
		else:
			name = filename
		pdf = matplotlib.backends.backend_pdf.PdfPages(name)

	G_filt = G[G["N_gal"]>= Nmin]
	N = len(G_filt)
	p_size = len(G_filt)*4

	fig = plt.figure(figsize = (17,p_size))
	k = 0
	for i, g in G_filt.iterrows():
		k +=1
		print(i,k)
		#try:
		df = R[R["group_id"] == g["group_id"]]	
		z_mean = g["mean_z"]
		ra_mean = g["mean_ra"]*u.degree
		dec_mean = g["mean_dec"]*u.degree

		#z_bary = g["barycenter_z"]
		#ra_bary = g["barycenter_ra"]*u.degree
		#dec_bary = g["barycenter_dec"]*u.degree


		if ~np.isnan(g["center_z"]):
			z_center = g["center_z"]
			ra_center =  g["center_ra"]*u.degree
			dec_center = g["center_dec"]*u.degree
		else: 
			z_center = g["mean_z"]
			ra_center =  g["mean_ra"]*u.degree
			dec_center = g["mean_dec"]*u.degree

		ra_qso = (df["ra_qso"].mean())*u.degree
		dec_qso = (df["dec_qso"].mean())*u.degree
		rho0, Rs = get_nfw_param(g["Mvir_sigma2"], z_mean)
		r = np.linspace(1,3000, num = 10000 )
		M = nfw_cumsum(r, rho0, Rs)
		#V = get_esc_v(M, r)
		#print(g["Mvir_sigma2"]*u.solMass, g["center_z"])
		V = NFW_escape_vel_from_Mvir(r*u.kpc, g["Mvir_sigma2"]*u.solMass, z = g["center_z"])

		C = const.c.to('km/s').value
		deg_to_rad = 1*u.degree.to(u.radian)
		
		ra2 = df["RA"]*u.degree
		dec2 = df["DEC"]*u.degree
		c1 = SkyCoord(ra_center, dec_center)
		c2 = SkyCoord(ra2, dec2)
		sep = c1.separation(c2)
		df["r_to_gcenter"] = sep.radian*Distance(unit=u.kpc, z = z_center).value/((1+z_center)**2)
		df["vlos_to_gcenter"] = (df["Z"]-z_center)*C/(1+z_center)
		
		Rvir = g["Rvir_sigma2"]

		ra = df["RA"]
		dec = df["DEC"]
		vlos = df["vlos_to_gcenter"]
		rcenter = df["r_to_gcenter"]

		df_filt = df[df["outlier"] == False]
		df_out = df[df["outlier"] == True]


		ra = np.array(ra)
		dec = np.array(dec)
		vlos = np.array(vlos)
		rcenter = np.array(rcenter)
		
		
		title = "Grp"+str(g["ID"])\
				+":   $log(M_{\mathrm{vir}}/\mathrm{M_{\odot}}$) = "\
				+str(round(np.log10(g["Mvir_sigma2"]),1))\
				+",     $R_{\mathrm{vir}}$ = "+str(round(g["Rvir_sigma2"]))\
				+" kpc" \
				+", field: "+str(g["field_id"])\
				+", z = " +str(np.round(g["center_z"], 2))
		
		#---SUBPLOT 1 ----------------------------------------------------------------
		plt.subplot(N,3,3*k-2)
		kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(z_mean)
		arcmin = 60
		r100 = 100/kpc_per_arcmin.value*60
		plt.axhline(0, color = "gray", linestyle = "--")
		plt.axvline(0, color = "gray", linestyle = "--")
		rectangle = plt.Rectangle((-arcmin/2, -arcmin/2), arcmin, arcmin, fc=None ,ec=None, lw = 0, fill = False)
		plt.gca().add_patch(rectangle)
		circle = plt.Circle((0,0),r100, fill = False, ec="green")
		plt.gca().add_patch(circle)
		plt.scatter((df["RA"] - ra_qso.value)*3600, (df["DEC"] - dec_qso.value)*3600, \
					c = "blue", marker = "s", s = 50, vmin = 0, vmax = 1)
		if "Psat" in df.columns:
			plt.scatter((df["RA"] - ra_qso.value)*3600, (df["DEC"] - dec_qso.value)*3600, \
						c = "blue", s = np.minimum(df['mass10']/1e8 + 80, 500), vmin = 0, vmax = 1)
			#plt.colorbar(label = "Psat")	
		else:
			plt.scatter((df["RA"] - ra_qso.value)*3600, (df["DEC"] - dec_qso.value)*3600, \
						c = -df["m_abs_r"], edgecolor = 'grey', s = df['mass10']/1e8 + 80)
			plt.colorbar(label = "$- m_r$ absolute")
		#plt.xticks([])
		#plt.yticks([])
		plt.scatter((df_out["RA"] - ra_qso.value)*3600, (df_out["DEC"] - dec_qso.value)*3600, s=250, facecolors='none', edgecolors='r')
		plt.scatter(0,0, marker = "*", s = 150, c = "red")
		#plt.scatter((ra_mean.value - ra_qso.value)*3600, (dec_mean.value-dec_qso.value)*3600, marker = "*", s = 80, c = "green")
		#plt.scatter((ra_bary.value - ra_qso.value) * 3600, (dec_bary.value - dec_qso.value) * 3600, marker="*", s=80, c="orange")
		plt.scatter((ra_center.value - ra_qso.value)*3600, (dec_center.value - dec_qso.value)*3600, marker = "*", s = 120, c = "orange")
		plt.errorbar((ra_center.value - ra_qso.value) * 3600, (dec_center.value - dec_qso.value) * 3600,
					xerr = g["center_ra_err"]*3600, yerr = g["center_dec_err"]*3600,  c="orange")
		plt.gca().set_aspect('equal', adjustable='box')
		plt.xlabel("$\delta(\")$")
		plt.ylabel("$\delta(\")$")
		for j in range(len(df)):
			plt.annotate(str(j), ((ra[j] - ra_qso.value)*3600 +2, (dec[j] - dec_qso.value)*3600+2))
		fig.tight_layout()


		#---SUBPLOT 2-------------------------------------------------------------
		plt.subplot(N,3,3*k-1)
		plt.title(title, pad = 10)
		plt.scatter(df["r_to_gcenter"], df["vlos_to_gcenter"], \
					c = "blue", marker = "s", s = 50, vmin = 0, vmax = 1)
		if "Psat" in df.columns:
			plt.scatter(df["r_to_gcenter"], df["vlos_to_gcenter"], \
					c = "blue", vmin = 0, vmax = 1, s = np.minimum(df['mass10']/1e8 + 80,500))
			#cbar = plt.colorbar(label =	 "Psat")
		else:
			plt.scatter(df["r_to_gcenter"], df["vlos_to_gcenter"], \
					c = -df["m_abs_r"], s = df['mass10']/1e8 + 80)
			cbar = plt.colorbar(label =	 "$- m_r$ absolute")
		plt.scatter(df_out["r_to_gcenter"], df_out["vlos_to_gcenter"], s = 250, facecolors='none', edgecolors='r')
		plt.axvline(Rvir, ls = "--", c = "green")
		plt.plot(r, V, c = "black", label = "caustic")
		plt.plot(r, -V, c = "black")
		vmax = np.max(np.abs(df["vlos_to_gcenter"]))
		Vmax = (np.max(V)).value
		#print(vmax, Vmax)
		ymax = max(Vmax, vmax)+50 # for the ylim we take the maximum between the caustic and the galaxies
		plt.ylim((-ymax, ymax))
		try:
			#plt.xlim((0,(np.max(df["r_to_gcenter"]))*1.2))
			plt.xlim((0,600))
		except: 
			print("no center!")
		#plt.ylim((-1500,1500))
		plt.xlabel("r (kpc)")
		plt.ylabel("$\Delta v$ [km/s]")
		plt.axhline(c = "grey")
		#plt.legend()
		for j in range(len(df)):
			plt.annotate(str(j), (rcenter[j] + 12, vlos[j]))
		
	#---- SUBPLOT 3 ---------------------------------------------------------------------------------
		Abs = pd.read_csv("MgII_all_absorption")
		Abs_filt = Abs[Abs["field_name"] == g["field_id"]]
		Abs_filt["v_diff"] = np.abs(Abs_filt["z_abs"] -g["center_z"])*const.c.value/(1+g["center_z"])
		Abs_filt = Abs_filt[np.abs(Abs_filt["z_abs"] -g["center_z"])*const.c.value/(1+Abs_filt["z_abs"]) <= dv]
		Abs_filt["v_abs"] = (Abs_filt["z_abs"] - g["center_z"])*const.c.to(u.km/u.s).value/(1+Abs_filt["z_abs"])
		#print(Abs_filt)
		field_id = g["field_id"]

		#print("group id: ", g, " field :", field_id)
		
		file_name = "uves_data/j"+field_id[1:]+".dat"
		uves = pd.read_csv(file_name, delim_whitespace = True, names = ["lambda", "flux_norm", "flux", "C", "continuum"], index_col = False)
		uves["z_abs"] = uves["lambda"]/2796.352 - 1
		uves["delta_v"] = (uves["z_abs"] - z_center)*const.c.to(u.km/u.s).value/(1+z_center)

		plt.subplot(N,3,3*k)
		for l in df["vlos_to_gcenter"]:
			plt.axvline(l, c = "blue", linewidth = 0.5)
		for i, a in Abs_filt.iterrows():
			plt.axvline(a["v_abs"], c = "limegreen", label = "absorption", linewidth = 1.5)
		plt.plot(uves["delta_v"], uves["flux_norm"], color = "red", linewidth = 0.5)
		#plt.plot(uves["lambda"], uves["flux_norm"])
		plt.xlabel("$\Delta v$ [km/s]")
		plt.ylabel("Norm flux")
		plt.ylim((-0.2,1.5))
		plt.xlim((-1000,1000))
		mm = 0
		for j in range(len(df)):
			plt.annotate(str(j), (vlos[j]+2, -0.1 + mm*0.2))
			mm = (mm + 1)%3 

	if save == True:
		pdf.savefig(fig)

		#except Exception as e:  
		#	print(e)

	if save == True:
		pdf.close()

	return


def Behroozi(log10Mstar, z):
	a = 1/(1+z)
	M00 = 11.09
	M0a = 0.56
	M10 = 12.27
	M1a = -0.84
	beta0 = 0.65
	betaa = 0.31
	delta0 = 0.56
	deltaa = -0.12
	gamma0 = 1.12
	gammaa = -0.53
	
	log10M1 = M10 + M1a*(a-1)
	log10M0 = M00 + M0a*(a-1)
	beta = beta0 + betaa*(a-1)
	delta = delta0 + deltaa*(a-1)
	gamma = gamma0 + gammaa*(a-1)
	
	log10Mh = log10M1 + beta*(log10Mstar - log10M0) \
				+ ((10**log10Mstar/(10**log10M0))**delta)/(1 + (10**log10Mstar/(10**log10M0))**-gamma)\
				- 0.5
	return log10Mh


def Behroozi_2019_inv(log10Mhalo, z):
    
    epsilon0 = -1.435
    alphalna = -1.732
    epsilona = 1.831
    alphaz = 0.178
    epsilonlna = 1.368
    beta0 = 0.482
    epsilonz = -0.217
    betaa = -0.841
    M0 = 12.035
    betaz = -0.471
    Ma = 4.556
    delta0 = 0.411
    Mlna = 4.417
    gamma0 = -1.034
    Mz = -0.731
    gammaa = -3.100
    alpha0 = 1.963
    gammaz = -1.055
    alphaa = -2.316
    chi2 = 157
    
    a = 1/(1+z)
    epsilon = epsilon0 + epsilona*(a-1) - epsilonlna*np.log(a) + epsilonz*z
    alpha = alpha0 + alphaa*(a-1) - alphalna*np.log(a) + alphaz*z
    beta = beta0 + betaa*(a-1) + betaz*z
    delta = delta0
    gamma = 10**(gamma0 + gammaa*(a-1) + gammaz*z)
    
    M1 = 10**(M0 + Ma*(a-1) - Mlna*np.log(a) + Mz*z)
    
    x = log10Mhalo - np.log10(M1)
    
    log10Mstar = epsilon - np.log10(10**(-alpha*x) + 10**(-beta*x))\
                + gamma*np.exp(-0.5*((x/delta)**2)) + np.log10(M1)
    
    return log10Mstar


def Behroozi_2019(log10Mstar, z):
    """
    the relation from Behroozi 2019 is inverted to obtain the Mhalo from M*
    for that we do an interpolation of M* from Mhalo
    """
    
    print(" M* ==== ", log10Mstar)
    log10Mh = np.linspace(6,18, 100000)
    log10Ms = log10Ms = Behroozi_2019_inv(log10Mh, z)
    
    if np.isnan(log10Mstar):
        return np.nan
    
    
    diff = np.abs(log10Ms-log10Mstar)
    argmin = np.argmin(diff)
    
    print("closest match = ", log10Ms[argmin])
    # interpolation:
    if log10Ms[argmin] - log10Mstar >= 0:
        print(log10Ms[argmin - 1], log10Mstar, log10Ms[argmin])
        slope = (log10Mh[argmin] - log10Mh[argmin - 1])/(log10Ms[argmin] - log10Ms[argmin - 1])
        print("slope = ", slope)
        log10Mh_res = log10Mh[argmin - 1] + slope*(log10Mstar-log10Ms[argmin - 1])
    if log10Ms[argmin] - log10Mstar < 0:
        print(log10Ms[argmin], log10Mstar, log10Ms[argmin+1])
        slope = (log10Mh[argmin+1] - log10Mh[argmin])/(log10Ms[argmin+1] - log10Ms[argmin])
        print("slope = ", slope)
        log10Mh_res = log10Mh[argmin] + slope*(log10Mstar-log10Ms[argmin])
        
    return log10Mh_res


def Girelli_2020_inv(log10Mhalo, z):
    
    Mhalo = 10**log10Mhalo
    
    B = 11.79
    mu = 0.2
    C = 0.046
    nu = -0.38
    D = 0.709
    eta = -0.18
    F = 0.043
    E = 0.96
    
    Ma = 10**(B+z*mu)
    A = C*(1+z)**nu
    gamma = D*(1+z)**eta
    beta = F*z+E
    
    T1 = (Mhalo/Ma)**(-beta)
    T2 = (Mhalo/Ma)**(gamma)
    K = 2*A/(T1 + T2)
    
    log10Mstar = np.log10(K)+log10Mhalo
    
    return log10Mstar

def Girelli_2020(log10Mstar, z):
    """
    the relation from Behroozi 2019 is inverted to obtain the Mhalo from M*
    for that we do an interpolation of M* from Mhalo
    """
    
    print(" M* ==== ", log10Mstar)
    log10Mh = np.linspace(6,18, 100000)
    log10Ms = log10Ms = Girelli_2020_inv(log10Mh, z)
    
    if np.isnan(log10Mstar):
        return np.nan
    
    
    diff = np.abs(log10Ms-log10Mstar)
    argmin = np.argmin(diff)
    
    print("closest match = ", log10Ms[argmin])
    # interpolation:
    if log10Ms[argmin] - log10Mstar >= 0:
        print(log10Ms[argmin - 1], log10Mstar, log10Ms[argmin])
        slope = (log10Mh[argmin] - log10Mh[argmin - 1])/(log10Ms[argmin] - log10Ms[argmin - 1])
        print("slope = ", slope)
        log10Mh_res = log10Mh[argmin - 1] + slope*(log10Mstar-log10Ms[argmin - 1])
    if log10Ms[argmin] - log10Mstar < 0:
        print(log10Ms[argmin], log10Mstar, log10Ms[argmin+1])
        slope = (log10Mh[argmin+1] - log10Mh[argmin])/(log10Ms[argmin+1] - log10Ms[argmin])
        print("slope = ", slope)
        log10Mh_res = log10Mh[argmin] + slope*(log10Mstar-log10Ms[argmin])
        
    return log10Mh_res


def f(x):
	A = 1/(x**2 - 1)
	B = (1-x**2)**0.5
	C = np.log((1+B)/x)
	D = (x**2-1)**0.5

	if x < 1:
		return A*(1-C/B)
	if x == 1:
		return 1/3
	if x > 1:
		return A*(1-np.arctan(D)/D)


def fE(z):
	return 1/((cosmo.Om0*(1+z)**3 + cosmo.Ok0*(1+z)**2 + cosmo.Ode0)**0.5)

def Dcomov(Z):
	from scipy.integrate import quad, dblquad
	A = (const.c*quad(fE, 0, Z)/cosmo.H0).to(u.Mpc)
	return A[0].value # in cMpc

def Vmegaflow(z1, z2):
	"""
	return the comoving volume of 1 megalfow field between z0 and z1.
	It is assumed that the Muse field is 1x1 arcmin2
	"""
	from scipy.integrate import quad, dblquad
	A = quad(Dcomov, z1, z2)
	B = A[0]
	C = (B)**2*np.pi/60/360
	return C*(u.Mpc)**3 #cMpc3



def calc_Psat(r, Mvir, z_center):

	rho0, Rs = get_nfw_param(Mvir, z_center)
	R = np.linspace(1,3000, num = 10000)
	M = nfw_cumsum(R, rho0, Rs)
	V = get_esc_v(M, R)
	Rvir = get_Rvir(Mvir, z_center)
	kpc_to_m = (1*u.kpc).to(u.m)
	Vvir = (const.G*Mvir*const.M_sun/(Rvir*u.kpc))**0.5*(1+z_center) #in km/s
	Vvir = Vvir.to(u.km/u.s)
    #R["Vvir"] = (const.G*R["Mhalo"]*const.M_sun/(R["Rvir"]*kpc_to_m))**0.5/1000 #in km/s
	sigmav = Vvir/(2**0.5)
	Bsat = 10

	rhoU = cosmo.Om(z_center)*cosmo.critical_density(z_center)
	rhoU = rhoU.to(u.Msun/u.kpc**3).value

	r["Dz"] = z_center - r["Z"]
    
	F = (cosmo.H(0)/const.c).to(u.kpc**-1)
	F = F.value

	for j, k in r.iterrows():
		x = k["r_to_gcenter"]/Rs
		r.loc[j, "f(x)"] = f(x)

	r["Pproj"] = 2*F*Rs*rho0/rhoU*r["f(x)"]
	r["Pz"] = const.c*np.exp(-(const.c.to(u.km/u.s)*r["Dz"])**2/(2*sigmav**2))/((2*np.pi)**0.5*sigmav)
	#print(sigmav)
	#print("ddd : ", -(const.c.to(u.km/u.s)*0.0001)**2/(2*sigmav**2))
    
	r["Psat"] = 1-1/(1+r["Pproj"]*r["Pz"]/Bsat)

	return r

def plot_Psat(r, r_core, g, Rvir, Mvir, Mstar, Mhalo_from_Mstar, ra_center, dec_center, ra_qso, dec_qso):
	#print(" PLOT PSAT values:", r["Psat"])

	fig = plt.figure(figsize = (15,4))
	title = "ID: "+ str(g["group_id"])+ " Mvir = " + str(np.log10(Mvir))+ "--- Mass M* = "+ str(np.log10(Mstar))\
				+ "--- Mhalo from M* = "+str(np.log10(Mhalo_from_Mstar))
	plt.suptitle(title)
	plt.subplot(121)
	plt.scatter(r["r_to_gcenter"], r["vlos_to_gcenter"], c = np.array(r["Psat"], dtype =float), marker = "s", s = 50, vmin = 0, vmax = 1)
	plt.scatter(r_core["r_to_gcenter"], r_core["vlos_to_gcenter"], marker = "s", facecolors = "none", edgecolors = "r",\
		s = 60)
	plt.scatter(r["r_to_gcenter"], r["vlos_to_gcenter"], c = np.array(r["Psat"], dtype =float), s = r['mass10']/1e8 +80, vmin = 0, vmax = 1)
	plt.colorbar()
	plt.scatter(r_core["r_to_gcenter"], r_core["vlos_to_gcenter"], facecolors = "none", edgecolors = "r",\
			s = r_core['mass10']/1e8 + 150)
    #plt.scatter(r_core["r_to_gcenter"], r_core["vlos_to_gcenter"], facecolors = "none", edgecolors = "r", s = 100)
	plt.axhline(0, color = "gray", linestyle = "--")
	plt.axvline(Rvir, ls = "--", c = "green", label = "R virial")
	#plt.plot(R, V, c = "black", label = "caustic")
	#plt.plot(R, -V, c = "black")
	plt.xlim((0,(np.max(r["r_to_gcenter"]))*2))
	r_center = np.array(r["r_to_gcenter"])
	vlos = np.array(r["vlos_to_gcenter"])
	psat = np.array(r["Psat"])
	#for j in range(len(r)):
	#		plt.annotate(str(psat[j]), (r_center[j] + 12, vlos[j]))

	plt.subplot(122)
	plt.scatter(r["RA"], r["DEC"], c = np.array(r["Psat"], dtype =float), marker = "s", s = 50, vmin = 0, vmax = 1)
	plt.scatter(r_core["RA"], r_core["DEC"], marker = "s", facecolors = "none", edgecolors = "r", \
				s = 60)
	plt.scatter(r["RA"], r["DEC"], c = np.array(r["Psat"], dtype =float), s = r['mass10']/1e8 +80, vmin = 0, vmax = 1)
    #plt.scatter(group_caust["center_ra"], group_caust["center_dec"], marker = "s", color = "pink")
	plt.scatter(r_core["RA"], r_core["DEC"], facecolors = "none", edgecolors = "r", \
				s = r_core['mass10']/1e8 + 150)
	plt.scatter(ra_center.value, dec_center.value, marker = "X", color = "pink")
	plt.scatter(ra_qso, dec_qso, marker = "*", s = 100, c = "red", label = "QSO")
	kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(g["center_z"])
	arcmin = 1/60
	r100 = 100/kpc_per_arcmin.value/60
	rvir = Rvir/kpc_per_arcmin.value/60
	rectangle = plt.Rectangle((ra_qso - arcmin/2,dec_qso - arcmin/2), arcmin, arcmin, fc=None ,ec=None, lw = 0, fill = False)
	circle = plt.Circle((ra_qso,dec_qso),r100, fill = False,ec="green")
	plt.gca().set_aspect('equal', adjustable='box')
	plt.gca().add_patch(circle)
	plt.gca().add_patch(rectangle)
	fig.tight_layout()


def group_iterative_2(g, r):
	r = r.sort_values(by = "mass10", ascending = False)

	# We add a new column to indicate which galaxy is a candidate to the group.
	r["candidate"] = 0

	# we identify the heaviest galaxy of the group:
	heaviest = r[:1].squeeze()
	heaviest5 = r[:5].squeeze()
	ra_qso = g["ra_qso"]
	dec_qso = g["dec_qso"]

	# if we have a center we use it, otherwise we use the mean coordinates.
	if ~np.isnan(g["center_z"]):
		z_center = g["center_z"]
		ra_center =  g["center_ra"]*u.degree
		dec_center = g["center_dec"]*u.degree
	else:
		print("NO CENTER !")
		z_center = g["mean_z"]
		ra_center =  g["mean_ra"]*u.degree
		dec_center = g["mean_dec"]*u.degree

    # We compute the distance and velocity relatively to the group center:
	c1 = SkyCoord(r["RA"]*u.degree, r["DEC"]*u.degree)
	c2 = SkyCoord(ra_center, dec_center)
	sep = c1.separation(c2)
	r["r_to_gcenter"] = sep.radian*Distance(unit=u.kpc, z = z_center).value/((1+z_center)**2)
	r["vlos_to_gcenter"] = (r["Z"]-z_center)*const.c.value/(1+z_center)/1000

	# We compute the halo Mvir and Rvir from the stellar mass of the heaviest galaxies 
	#(not the 5 heaviest because they may have important distances in phase space) :
	Mvir = 10**Girelli_2020(heaviest["mass"], z_center)
	#Mvir = 10**Behroozi_2019(np.log10(heaviest5["mass10"].sum()), z_center)
	Rvir = get_Rvir(Mvir, z_center)
	Mstar = heaviest["mass10"].sum()
	Mhalo_from_Mstar = Mvir
	Psat_max = 1
	i = 5

	# We compute Psat a first time using the Mhalo derived from the stellar mass:
	r = calc_Psat(r, Mvir, z_center)

	# Then we consider as candidate to the group the 5 galaxies with the highest Psat values:
	r = r.sort_values(by = "Psat", ascending = False)
	r.loc[r[:5].index,"candidate"] = 1
	r_core = r[r["candidate"]==1]
	r_rest = r[r["candidate"]==0]

	# if some galaxies have a Psat value below 0.5 we stop here:
	r_core = r_core[r_core["Psat"] >= 0.5]
	if len(r_core) < 5:
		print("BREAK: ", len(r_core), "candidates kept")
		return g, r
	# We plot the group a first time with the Psat values:
	plot_Psat(r, r_core, g, Rvir, Mvir, Mstar, Mhalo_from_Mstar, ra_center, dec_center, ra_qso, dec_qso)

	# Then we recompute the mass with the velocity dispersion method and we recompute Psat:
	G = pd.DataFrame(g).transpose()
	r_core, _ = calc_mass(r_core, G, N_lim = 3)
	Mvir = r_core["Mvir_group_sigma2"].mean()
	Rvir = r_core["Rvir_group_sigma2"].mean()
	Mstar = r_core["mass10"].sum()
	Mhalo_from_Mstar = 10**Girelli_2020(np.log10(Mstar), z_center)

	r = calc_Psat(r, Mvir, z_center)
	# we update r_core and r_rest:
	r_core = r[r["candidate"]==1]
	r_rest = r[r["candidate"]==0]

	# Among the 5 candidates we keep the candidates if their Psat value is above 0.5:
	r_core = r_core[r_core["Psat"] >= 0.5]

	# We plot the group a second time with the new Psat values:
	plot_Psat(r, r_core, g, Rvir, Mvir, Mstar, Mhalo_from_Mstar, ra_center, dec_center, ra_qso, dec_qso)

	# if some of our 5 candidates have a Psat value below 0.5 we stop here and indicate the number of galaxy retained:
	if len(r_core) < 5:
		print("BREAK: ", len(r_core), "candidates kept")
		return g, r 

	# else, we loop to add galaxies one by one:
	while i <= len(r):
		
		#print("iteration ", i, ": ")

		# if there is no remaining galaxy with a Psat value above 0.5, we stop here.
		Psat_max = r_rest["Psat"].max()
		if Psat_max < 0.5:
			print("END: Psat max = ", Psat_max)
			return g, r

		# Else we add the galaxy with the highest Psat to the group.
		new = r[r["candidate"] == 0]
		new = new[new["Psat"] == new["Psat"].max()]
		r.loc[new.index,"candidate"] = 1

		#r = r.sort_values(by = "Psat", ascending = False)
		r_core = r[r["candidate"]==1]
		r_rest = r[r["candidate"]==0]

		# and we recompute the group properties:
		r_core, _ = calc_mass(r_core, G, N_lim = 3)
		Mvir = r_core["Mvir_group_sigma2"].mean()
		Rvir = r_core["Rvir_group_sigma2"].mean()
		Mstar = r_core["mass10"].sum()
		Mhalo_from_Mstar = 10**Girelli_2020(np.log10(Mstar), z_center)

		
		# and the Psat values:
		r = calc_Psat(r, Mvir, z_center)

		# Then we plot:
		print("PLOT PLOT PLOT")
		plot_Psat(r, r_core, g, Rvir, Mvir, Mstar, Mhalo_from_Mstar, ra_center, dec_center, ra_qso, dec_qso)

		i +=1

	return g, r

def hod_refinement_2(G, R):
	for i, g in G.iterrows():
		print(i)
		r = R[R["group_id"] == g["group_id"]]
		if len(r) >= 5:
			print("------------- GROUP ID: ", g["group_id"])
			
			g, r = group_iterative_2(g, r)

			R = R[R["group_id"] != g["group_id"]] # we drop the previous group from R
			R = R.append(r, ignore_index = True)
			G = G[G["group_id"] != g["group_id"]]
			G = G.append(g, ignore_index = True)

	return G, R


def Fukugita(REW_2796, sig_REW_2796, z):
	"""
	convert a Mg+ equivalent width in NHI column density.
	"""
	A = 10**18.96
	a = 1.69
	b = 1.88

	sig_a = 0.13
	sig_b = 0.29

	sigma = sig_REW_2796*A*a*(REW_2796)**(a-1)*(1+z)**b + \
			A*np.log(REW_2796)*(REW_2796)**a*(1+z)**b*sig_a + \
			A*np.log(1+z)*(REW_2796)**a*(1+z)**b*sig_b

	return A*(REW_2796)**a*(1+z)**b, sigma

    

def c4(n):
	return 1-1/(4*n) - 7/(32*n**2) - 19/(128*n**3)

def sigma_error(sigma_v, n):
	"""
	Compute the 1-sigma error on the velocity dispersion estimation. Depends on the number of galaxies that we have
	for the sampling.
	For more details see: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
	"""
	return sigma_v*np.sqrt(1-(c4(n))**2)

def radius_crop_error(R, z):
	"""
	Estimate the error coming from the fact that the MUSE FOV crop the field of view. The visually estimated radius is
	then often underestimated.
	"""
	# the size of the MUSE field (the distance between the center and the side of the frame actually)
	L = 2*np.pi*30/3600/60*Distance(unit=u.kpc, z = z).value/((1+z)**2)

	#the average error is then approximately (see personal notes):
	case_when_error = (2*L)**2 - (2*(L-R))**2
	total_cases = (2*L)**2
	proba_to_make_error = case_when_error/total_cases

	error_magnitude = (R/L)**2*(2*L-R)

	avg_error = proba_to_make_error*error_magnitude

	return avg_error

def b_center_error(G):
	G["b_center_err"] = 0
	for i, g in G.iterrows():
		ra_c = g["center_ra"] * u.degree
		dec_c = g["center_dec"] * u.degree
		ra_c_err = (g["center_ra"] + g["center_ra_err"]) * u.degree
		dec_c_err = (g["center_dec"] + g["center_dec_err"]) * u.degree

		ra_qso = g["ra_qso"] * u.degree
		dec_qso = g["dec_qso"] * u.degree
		c1 = SkyCoord(ra_c, dec_c)
		c2 = SkyCoord(ra_qso, dec_c)
		c3 = SkyCoord(ra_c, dec_qso)
		c4 = SkyCoord(ra_c_err, dec_c)
		c5 = SkyCoord(ra_c, dec_c_err)

		ra_sep = c1.separation(c2)
		dec_sep = c1.separation(c3)
		ra_err_sep = c1.separation(c4)
		dec_err_sep = c1.separation(c5)
		ra_dist = ra_sep.radian * Distance(unit=u.kpc, z=g["center_z"]).value / ((1 + g["center_z"]) ** 2)
		dec_dist = dec_sep.radian * Distance(unit=u.kpc, z=g["center_z"]).value / ((1 + g["center_z"]) ** 2)
		ra_err_dist = ra_err_sep.radian * Distance(unit=u.kpc, z=g["center_z"]).value / ((1 + g["center_z"]) ** 2)
		dec_err_dist = dec_err_sep.radian * Distance(unit=u.kpc, z=g["center_z"]).value / ((1 + g["center_z"]) ** 2)
		# df["r_to_gcenter"] = sep.radian * Distance(unit=u.kpc, z=z_center).value / ((1 + z_center) ** 2)
		# print("dist =", ra_dist, dec_dist)
		b_kpc = (ra_dist ** 2 + dec_dist ** 2) ** 0.5

		# print(b_kpc, g["b_center_kpc"])
		# print("errors = ", ra_err_dist, dec_err_dist)
		b_ra_err = ra_dist * ra_err_dist / b_kpc
		b_dec_err = dec_dist * dec_err_dist / b_kpc

		b_err = b_ra_err + b_dec_err
		G.loc[i, "b_center_err"] = b_err

	return G

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#-- set of function to compute escape velocities for a NFW profile --
#--- from the notebook community:
#--- https://notebook.community/saga-survey/erik/ipython_notebooks/NFW%20Escape%20Velocity

#from astropy import cosmology
#cosmo = cosmology.default_cosmology.get()
#FlatLambdaCDM(name="WMAP9", H0=69.3 km / (Mpc s), Om0=0.286, Tcmb0=2.725 K, Neff=3.04, m_nu=[ 0.  0.  0.] eV, Ob0=0.0463)

def NFW_escape_vel(r, Mvir, Rvir, CvirorRs, truncated=False):
	"""
    NFW profile escape velocity

    Parameters
    ----------
    r : Quantity w/ length units
        Radial distance at which to compute the escape velocity
    Mvir : Quantity w/ mass units
        Virial Mass
    CvirorRs : Quantity w/ dimensionless or distance units
        (Virial) Concentration parameter (if dimensionless),
        or halo scale radius (if length units)
    Rvir : Quantity w/ length units
        Virial radius
    truncated : bool or float
        False for infinite-size NFW or a number to cut off the
        halo  at this many times Rvir
    """
	CvirorRs = u.Quantity(CvirorRs)
	if CvirorRs.unit.is_equivalent(u.m):
		Cvir = Rvir / CvirorRs
	elif CvirorRs.unit.is_equivalent(u.one):
		Cvir = CvirorRs
	else:
		raise TypeError('CvirorRs must be length or dimensionless')

	a = Rvir / Cvir

	# "f-function" from the NFW literature (e.g. klypin 02) evaluated at Cvir
	fofC = np.log(1 + Cvir) - Cvir / (1 + Cvir)

	# value of the NFW potential at that point
	potential = (-const.G * Mvir / fofC) * np.log(1 + (r / a)) / r

	if truncated:
		rtrunc = Rvir * float(truncated)
		Ctrunc = rtrunc / a

		mtrunc = Mvir * (np.log(1 + Ctrunc) - Ctrunc / (1 + Ctrunc)) / fofC

		outer = r >= rtrunc
		potential[outer] = - Gkpc * mtrunc / r[outer]
		potential[~outer] = potential[~outer] + (Gkpc * Mvir / a) / (Ctrunc + 1) / fofC

	vesc = (2 * np.abs(potential)) ** 0.5
	return vesc.to(u.km / u.s)


def Deltavir(cosmo, z=0):
    """
    Standard Delta-vir from Bryan&Norman 98 (*not* Delta-c)
    """
    x = cosmo.Om(z) - 1
    return (18*np.pi**2 + 82*x - 39*x**2)/(x+1)


def rvirmvir(rvirormvir, cosmo, z=0):
	"""
    Convert between Rvir and Mvir

    Parameters
    ----------
    rvirormvir : Quantity w/ mass or length units
        Either Rvir or Mvir, depending on the input units
    cosmo : astropy cosmology
        The cosmology to assume
    z : float
        The redshift to assume for the conversion

    Returns
    -------
    mvirorrvir : Quantity w/ mass or length units
         Whichever ``rvirormvir`` is *not*
    """
	rhs = Deltavir(cosmo=cosmo, z=z) * cosmo.Om(z) * cosmo.H(z) ** 2 / (2 * const.G)

	if rvirormvir.unit.is_equivalent(u.solMass):
		mvir = rvirormvir
		return ((mvir / rhs) ** (1 / 3)).to(u.kpc)
	elif rvirormvir.unit.is_equivalent(u.kpc):
		rvir = rvirormvir
		return (rhs * rvir ** 3).to(u.solMass)
	else:
		raise ValueError('invalid input unit {}'.format(rvirormvir))

def mvir_to_cvir(mvir, z=0):
    """ Power-law fit to the c_vir-M_vir relation from
    Equations 12 & 13 of Dutton & Maccio 2014, arXiv:1402.7073.
    """
    a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * z**1.08)
    b = -0.097 + 0.024 * z
    m0 = 1e12 * u.solMass

    logc = a + b * np.log10(mvir / m0)
    return 10**logc


def NFW_escape_vel_from_Mvir(r, Mvir, z=0,
                             cosmo=cosmo,
                             truncated=False):
    cvir = mvir_to_cvir(Mvir, z)
    rvir = rvirmvir(Mvir, cosmo, z)
    return NFW_escape_vel(r, Mvir=Mvir,
                          CvirorRs=cvir,
                          Rvir=rvir,
                          truncated=truncated)


def calc_sigma_intrisic(yi, modeled_yi, sigma_mi):
    """
    compute the intrisic scatter from the comparison between data and model.
    It must be used iteratively until convergence to an intrisic scatter value.
    """

    
    mean_residual = np.mean(yi-modeled_yi)
    
    intrisic_i = (yi - modeled_yi - mean_residual)**2 - sigma_mi**2
    sigma_intri = np.median(intrisic_i)
    
    #print("yi = ", yi)
    #print("modeled_yi = ", modeled_yi)
    #print("sigma_mi = ", sigma_mi)
    #print("mean_residual = ", mean_residual)
    #print("sigma_intrisic = ", sigma_intri**0.5)
    
    return sigma_intri**0.5


def model(param, x):
    # y = ax +b
    return param[0] + param[1]*x


def logL_Hogg_total(param, x1, y1, sig_x1, sig_y1, x2, y2, sig_x2, sig_y2):
    theta = np.arctan(param[1])
    #print(sig_y1[0])
    N1 = len(x1)
    LL = 0
    
    for i in range(N1):
        xi = x1[i]
        yi = y1[i]
        sig_xi = sig_x1[i]
        sig_yi = sig_y1[i]
        Deltai_2 = (-np.sin(theta)*xi + np.cos(theta)*yi - np.cos(theta)*param[0])**2
        Sigmai_2 = (np.sin(theta))**2*sig_xi**2 + (np.cos(theta))**2*sig_yi**2
        LL += -Deltai_2/Sigmai_2/2
        
    N2 = len(x2)
    for i in range(N2):
        xi = x2[i]
        yi = y2[i]
        sig_xi = sig_x2[i]
        sig_yi = sig_y2[i]
        Deltai_2 = (-np.sin(theta)*xi + np.cos(theta)*yi - np.cos(theta)*param[0])**2
        Sigmai_2 = (np.sin(theta))**2*sig_xi**2 + (np.cos(theta))**2*sig_yi**2
        X = -(Deltai_2/Sigmai_2/2)**0.5
        I = 0.5*(1 + math.erf(X))
        LL += np.log(I)    
    return -LL



def fit_REW(G5, x1, y1, sig_x1, sig_y1, x2, y2, sig_x2, sig_y2, sig_y_noabs = 0.3, Niter = 5, init_param = np.array([0,-0.015])):
    """
    
    """
    
    G5_abs = G5[G5["bool_absorption"] == 1]
    G5_noabs = G5[G5["bool_absorption"] == 0]

    N = 5
    sigma_intrinsic_start = 0
    sigma_intrinsic = sigma_intrinsic_start
    sigma_intrisic_list = [sigma_intrinsic]
    for i in range(Niter):
        print("N = ", i)
        sig_y1_mi = np.array(G5_abs["sig_REW_2796"]/G5_abs["REW_2796"])
        sig_y1 = (sig_y1_mi**2 + sigma_intrinsic**2)**0.5
        LL_model_Hogg_total = minimize(logL_Hogg_total, init_param, \
                                       args = (x1, y1, sig_x1, sig_y1, x2, y2, sig_x2, sig_y2), method='BFGS')
        sigma_intrinsic = calc_sigma_intrisic(y1, model(LL_model_Hogg_total['x'], x1), sig_y1_mi)
        print(sigma_intrinsic)
        sigma_intrisic_list.append(sigma_intrinsic)
        
    plt.plot(sigma_intrisic_list)
    
    return LL_model_Hogg_total, sigma_intrisic_list


#--------------------------------------
def Tinker_2008(b):
    """
    theoretical profile for a halo of mass 1e12 at z = 0.6. Values are not correct. Just to see the shape...
    """
    A = 60
    M = 1e12
    Rg = 50
    ah = 0.2*Rg
    G0 = 1
    W = A*G0/((b**2 + ah**2)**0.5)*np.arctan(((Rg**2-b**2)/b**2 + ah**2)**0.5)
    return W

