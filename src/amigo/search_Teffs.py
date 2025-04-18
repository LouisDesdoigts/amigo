# import pyia
# import jax.numpy as np
# from astroquery.simbad import Simbad


# def get_simbad_spectral_type(source_id):
#     """Returns the spectral type and quality from Simbad for a given source ID."""
#     Simbad.add_votable_fields("sptype", "sp_qual")
#     table = Simbad.query_object(source_id)
#     if table is None:
#         return None, None
#     return table["SP_TYPE"].tolist()[0], table["SP_QUAL"].tolist()[0]


# def get_gaia_Teff(source_id, data_dr="dr3"):
#     """Returns the Teff from Gaia for a given source ID."""
#     result_table = Simbad.query_objectids(source_id)
#     if result_table is None:
#         return []
#     ids = []
#     for x in result_table:
#         if f"gaia {data_dr}" in x["ID"].lower():
#             ids.append(x["ID"].split(" ")[-1])

#     # All the potential keys with a Teff
#     teff_keys = [
#         "teff_gspphot",
#         "teff_gspspec",
#         "teff_msc1",
#         "teff_msc2",
#         "teff_esphs",
#         "teff_espucd",
#         "teff_val",
#     ]

#     Teffs_out = []
#     for obj_id in ids:
#         data = pyia.GaiaData.from_source_id(obj_id, source_id_dr="dr3", data_dr=data_dr)
#         for teff_type in teff_keys:
#             if hasattr(data, teff_type):
#                 val = np.squeeze(np.array(getattr(data, teff_type).value))
#                 if not np.isnan(val):
#                     Teffs_out.append(val)

#     return Teffs_out


# def get_Teff(targ_name):
#     """Returns the Teff for a given target name."""
#     # First check DR3
#     dr3_teffs = get_gaia_Teff(targ_name, data_dr="dr3")
#     if len(dr3_teffs) == 1:
#         return dr3_teffs[0]
#     elif len(dr3_teffs) > 1:
#         print(f"Multiple Teffs for {targ_name} in DR3, returning mean")
#         return np.array(dr3_teffs).mean()

#     # Then check Simbad -> Mamajeck?? table
#     # Skip this for now till I have time to make it work
#     if False:
#         spec_type, qual = get_simbad_spectral_type(targ_name)
#         if spec_type is not None:
#             return pyia.spectral_type_to_Teff(spec_type)
#         # TODO: Use `MeanStars` to get Teff from spectral type

#     # Finally, check DR2
#     dr2_teffs = get_gaia_Teff(targ_name, data_dr="dr2")
#     if len(dr2_teffs) == 1:
#         return dr2_teffs[0]
#     elif len(dr2_teffs) > 1:
#         print(f"Multiple Teffs for {targ_name} in DR2, returning mean")
#         return np.array(dr2_teffs).mean()

#     # Return -1 as a flag for 'not found'
#     return -1


# def get_Teffs(files, default=4500, skip_search=False, Teff_cache="files/Teffs"):
#     # # Check whether the specified cache directory exists
#     # if not os.path.exists(Teff_cache):
#     #     os.makedirs(Teff_cache)

#     Teffs = {}
#     for file in files:
#         prop_name = file[0].header["TARGPROP"]

#         # if os.exists(f"{Teff_cache}/{prop_name}.npy"):
#         try:
#             Teffs[prop_name] = np.load(f"{Teff_cache}/{prop_name}.npy")
#             continue
#         except FileNotFoundError:
#             pass

#         if prop_name in Teffs:
#             continue

#         # Temporary measure to get around gaia archive being dead
#         if skip_search:
#             Teffs[prop_name] = default
#             print("Warning using default Teff")
#             continue

#         # Teff = get_Teff(file[0].header["TARGNAME"])
#         Teff = np.array(5e3)

#         if Teff == -1:
#             print(f"No Teff found for {prop_name}, defaulting to 4500K")
#             Teffs[prop_name] = default
#         else:
#             Teffs[prop_name] = Teff
#             # np.save(f"{Teff_cache}/{prop_name}.npy", Teff)

#     return Teffs
