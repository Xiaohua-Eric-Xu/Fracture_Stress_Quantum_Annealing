import numpy as np

def calc_stress(strike_m, dip_m, rake_m, ss, c=0.0):
    """
    Calculate shear and normal stress on faults

    Parameters
    ----------
    strike_m, dip_m, rake_m : (n,) array
        Fault geometry (degree)
    ss : (6, n) array
        Stress components:
        [Sxx, Syy, Szz, Syz, Sxz, Sxy]
    c : unused (kept for compatibility)

    Returns
    -------
    shear : (n,) array
        Shear stress (positive = right-lateral)
    normal : (n,) array
        Normal stress (positive = clamped)
    """

    strike_m = np.asarray(strike_m).flatten()
    dip_m    = np.asarray(dip_m).flatten()
    rake_m   = np.asarray(rake_m).flatten()

    n = strike_m.size

    # -------------------------------------------------
    # Coordinate adjustment (Aki & Richards)
    # -------------------------------------------------
    c1 = strike_m >= 180.0
    c2 = strike_m < 180.0

    strike = (strike_m - 180.0) * c1 + strike_m * c2
    dip    = (-1.0) * dip_m * c1 + dip_m * c2

    rake_m = rake_m - 90.0
    c1 = rake_m <= -180.0
    c2 = rake_m > -180.0
    rake = (360.0 + rake_m) * c1 + rake_m * c2

    # degree → radian
    strike = np.deg2rad(strike)
    dip    = np.deg2rad(dip)
    rake   = np.deg2rad(rake)

    # -------------------------------------------------
    # Rotation matrix for rake
    # -------------------------------------------------
    mtran = np.zeros((3, 3, n))
    for i in range(n):
        rsc = -rake[i]  # flipped
        c, s = np.cos(rsc), np.sin(rsc)
        mtran[:, :, i] = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s,  c]
        ])

    # -------------------------------------------------
    # Direction cosines
    # -------------------------------------------------
    ver = np.pi / 2.0

    c1 = strike >= 0.0
    c2 = strike < 0.0
    c3 = strike <= ver
    c4 = strike > ver

    d1 = dip >= 0.0
    d2 = dip < 0.0

    xbeta = -strike * d1 + (np.pi - strike) * d2
    ybeta = (np.pi - strike) * d1 + (-strike) * d2
    zbeta = (
        (ver - strike) * d1 +
        (-ver - strike) * d2 * c1 * c3 +
        (np.pi + ver - strike) * d2 * (c2 | c4)
    )

    xdel = ver - np.abs(dip)
    ydel = np.abs(dip)
    zdel = np.zeros(n)

    xl = np.cos(xdel) * np.cos(xbeta)
    xm = np.cos(xdel) * np.sin(xbeta)
    xn = np.sin(xdel)

    yl = np.cos(ydel) * np.cos(ybeta)
    ym = np.cos(ydel) * np.sin(ybeta)
    yn = np.sin(ydel)

    zl = np.cos(zdel) * np.cos(zbeta)
    zm = np.cos(zdel) * np.sin(zbeta)
    zn = np.sin(zdel)

    # -------------------------------------------------
    # Stress transformation tensor
    # -------------------------------------------------
    t = np.zeros((6, 6, n))

    t[0, 0, :] = xl * xl
    t[0, 1, :] = xm * xm
    t[0, 2, :] = xn * xn
    t[0, 3, :] = 2 * xm * xn
    t[0, 4, :] = 2 * xn * xl
    t[0, 5, :] = 2 * xl * xm

    t[1, 0, :] = yl * yl
    t[1, 1, :] = ym * ym
    t[1, 2, :] = yn * yn
    t[1, 3, :] = 2 * ym * yn
    t[1, 4, :] = 2 * yn * yl
    t[1, 5, :] = 2 * yl * ym

    t[2, 0, :] = zl * zl
    t[2, 1, :] = zm * zm
    t[2, 2, :] = zn * zn
    t[2, 3, :] = 2 * zm * zn
    t[2, 4, :] = 2 * zn * zl
    t[2, 5, :] = 2 * zl * zm

    t[3, 0, :] = yl * zl
    t[3, 1, :] = ym * zm
    t[3, 2, :] = yn * zn
    t[3, 3, :] = ym * zn + zm * yn
    t[3, 4, :] = yn * zl + zn * yl
    t[3, 5, :] = yl * zm + zl * ym

    t[4, 0, :] = zl * xl
    t[4, 1, :] = zm * xm
    t[4, 2, :] = zn * xn
    t[4, 3, :] = xm * zn + zm * xn
    t[4, 4, :] = xn * zl + zn * xl
    t[4, 5, :] = xl * zm + zl * xm

    t[5, 0, :] = xl * yl
    t[5, 1, :] = xm * ym
    t[5, 2, :] = xn * yn
    t[5, 3, :] = xm * yn + ym * xn
    t[5, 4, :] = xn * yl + yn * xl
    t[5, 5, :] = xl * ym + yl * xm

    # -------------------------------------------------
    # Apply stress tensor & rotation
    # -------------------------------------------------
    sn9 = np.zeros((3, 3, n))
    for k in range(n):
        sn = t[:, :, k] @ ss[:, k]

        sn9[:, :, k] = np.array([
            [sn[0], sn[5], sn[4]],
            [sn[5], sn[1], sn[3]],
            [sn[4], sn[3], sn[2]]
        ])

        sn9[:, :, k] = sn9[:, :, k] @ mtran[:, :, k]

    # -------------------------------------------------
    # Output
    # -------------------------------------------------
    shear  =  sn9[0, 1, :].reshape(n)
    normal =  sn9[0, 0, :].reshape(n)

    return shear, normal