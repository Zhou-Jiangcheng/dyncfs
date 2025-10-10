from dyncfs.convert_input_format import *

if __name__ == "__main__":
    config = CfsConfig()
    config.read_config("wenchuan.ini")

    vp = 5.8000e3
    vs = 3.4600e3
    rho = 2.6000e3
    possion_ratio = ((vp / vs) ** 2 - 2) / (2 * (vp / vs) ** 2 - 1)
    youngs_modulus = 2 * rho * (1 + possion_ratio) * vs**2
    print("possion_ratio", possion_ratio)
    print("youngs_modulus", youngs_modulus)
    config.obs_delta_lat = 0.01
    config.obs_delta_lon = 0.01
    convert_source_csvs2coulomb3(
        config=config,
        obs_depth=15,
        possion_ratio=possion_ratio,
        youngs_modulus=youngs_modulus / 1e5,
    )
