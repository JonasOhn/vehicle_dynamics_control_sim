from casadi import *

N_refpath = 10
ref_path_s = np.linspace(0, 1, N_refpath)
p = MX.sym('p', N_refpath, 1)
x = MX.sym('x', 1, 1)

interpol_path_x = interpolant("interpol_spline_x", "bspline", [ref_path_s])
interp_exp = interpol_path_x(x, p)
interp_fun = Function('interp_fun', [x, p], [interp_exp])

# test interp_fun

x_test = MX.sym('x_test', 1, 1)
p_test = np.linspace(0, 1, N_refpath)
y_test = interp_fun(x_test, p_test)
print(y_test)

p_test = np.linspace(0, 2, N_refpath)
y_test = interp_fun(x_test, p_test)
print(y_test)

# generate C code
#interp_fun.generate() 