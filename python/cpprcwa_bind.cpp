// nanobind Python bindings for cpprcwa.
//
// Exposes the grcwa-compatible `obj` class plus the module-level helpers
// (Lattice_*, get_fft/get_ifft, Epsilon_fft) so that `import cpprcwa` can
// serve as a drop-in replacement for `import grcwa`.
#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/eigen/dense.h>

#include <cpprcwa/cpprcwa.h>

namespace nb = nanobind;
using namespace cpprcwa;

namespace {

[[noreturn]] void raise_notimplemented(const char* msg) {
    PyErr_SetString(PyExc_NotImplementedError, msg);
    throw nb::python_error();
}

// Convert a 2-element sequence (list/tuple/array) to Eigen::Vector2d.
Eigen::Vector2d to_vec2(const nb::object& o) {
    std::array<double, 2> v = nb::cast<std::array<double, 2>>(o);
    return Eigen::Vector2d(v[0], v[1]);
}

// Convert a numpy complex array / python sequence to std::vector<complex>.
std::vector<complex> to_complex_vec(const nb::handle& o) {
    nb::ndarray<complex, nb::c_contig> a;
    if (nb::try_cast(o, a)) {
        const complex* p = a.data();
        return std::vector<complex>(p, p + a.size());
    }
    nb::sequence s = nb::cast<nb::sequence>(o);
    std::vector<complex> v;
    v.reserve(nb::len(s));
    for (nb::handle item : s) v.push_back(nb::cast<complex>(item));
    return v;
}

// Is `o` a python complex / float / int (i.e. a scalar, not an array)?
bool is_scalar(const nb::handle& o) {
    return nb::isinstance<nb::float_>(o) || nb::isinstance<nb::int_>(o) ||
           PyComplex_Check(o.ptr());
}

// Build the grcwa-style [[ex,ey,ez],[hx,hy,hz]] list from one FieldFourier.
// Copies each Eigen vector into an OWNED numpy array: nanobind's Eigen caster
// creates a zero-copy view by default for a const&, which dangles once the
// temporary FieldFourier is destroyed (use-after-free — fields read as garbage
// after the caller allocates). Casting an rvalue Eigen copy forces a deep copy.
nb::object owned_vec(const Eigen::VectorXcd& v) {
    return nb::cast(Eigen::VectorXcd(v));
}

nb::object owned_mat(const Eigen::MatrixXcd& m) {
    return nb::cast(Eigen::MatrixXcd(m));
}

nb::object field_fourier_to_list(const FieldFourier& f) {
    nb::list e, h, eh;
    e.append(owned_vec(f.ex)); e.append(owned_vec(f.ey)); e.append(owned_vec(f.ez));
    h.append(owned_vec(f.hx)); h.append(owned_vec(f.hy)); h.append(owned_vec(f.hz));
    eh.append(e); eh.append(h);
    return nb::object(eh);
}

nb::object field_grid_to_list(const FieldGrid& f) {
    nb::list e, h, eh;
    e.append(owned_mat(f.ex)); e.append(owned_mat(f.ey)); e.append(owned_mat(f.ez));
    h.append(owned_mat(f.hx)); h.append(owned_mat(f.hy)); h.append(owned_mat(f.hz));
    eh.append(e); eh.append(h);
    return nb::object(eh);
}

std::vector<double> to_zoffsets(const nb::object& z_offset) {
    if (nb::isinstance<nb::float_>(z_offset) || nb::isinstance<nb::int_>(z_offset))
        return { nb::cast<double>(z_offset) };
    return nb::cast<std::vector<double>>(z_offset);
}

} // namespace

NB_MODULE(cpprcwa, m) {
    m.doc() = "cpprcwa: rigorous coupled-wave analysis (grcwa-compatible API).";

    // ── grcwa-compatible obj class ──────────────────────────────────────────
    nb::class_<RCWA>(m, "obj",
        "Rigorous coupled-wave analysis solver (drop-in for grcwa.obj).\n\n"
        "obj(nG, L1, L2, freq, theta, phi, verbose=1, quasi1d=False)\n"
        "  nG: truncation order; L1, L2: lattice vectors (x, y)\n"
        "  freq: frequency (1/lambda); theta, phi: incidence angles (rad)\n"
        "  quasi1d: restrict to the x-only harmonic set (exact for "
        "y-invariant structures)")
        .def("__init__",
             [](nb::pointer_and_handle<RCWA> v, int nG, const nb::object& L1,
                const nb::object& L2, complex freq, double theta, double phi,
                int verbose, bool quasi1d) {
                 RCWAConfig cfg;
                 cfg.nG = nG;
                 cfg.L1 = to_vec2(L1);
                 cfg.L2 = to_vec2(L2);
                 cfg.freq = freq;
                 cfg.theta = theta;
                 cfg.phi = phi;
                 cfg.verbose = verbose;
                 cfg.quasi1d = quasi1d;
                 new (v.p) RCWA(cfg);
             },
             nb::arg("nG"), nb::arg("L1"), nb::arg("L2"), nb::arg("freq"),
             nb::arg("theta"), nb::arg("phi"), nb::arg("verbose") = 1,
             nb::arg("quasi1d") = false)

        // ── layers ──
        .def("Add_LayerUniform", &RCWA::Add_LayerUniform,
             nb::arg("thickness"), nb::arg("epsilon"))
        .def("Add_LayerGrid", &RCWA::Add_LayerGrid,
             nb::arg("thickness"), nb::arg("Nx"), nb::arg("Ny"))
        .def("Add_LayerFourier",
             [](RCWA&, double, const nb::object&) {
                 raise_notimplemented(
                     "Add_LayerFourier is not implemented in cpprcwa");
             },
             nb::arg("thickness"), nb::arg("params"))

        .def("Init_Setup", &RCWA::Init_Setup,
             nb::arg("Pscale") = 1.0, nb::arg("Gmethod") = 0)

        .def("SetIncidence", &RCWA::SetIncidence,
             nb::arg("theta"), nb::arg("phi"))

        .def("MakeExcitationPlanewave",
             [](RCWA& self, double p_amp, double p_phase, double s_amp,
                double s_phase, int order, const std::string& direction) {
                 PlaneWaveExcitation exc;
                 exc.p_amp = p_amp;
                 exc.p_phase = p_phase;
                 exc.s_amp = s_amp;
                 exc.s_phase = s_phase;
                 exc.order = order;
                 exc.direction = direction == "backward" ? Direction::Backward
                                                         : Direction::Forward;
                 self.MakeExcitationPlanewave(exc);
             },
             nb::arg("p_amp"), nb::arg("p_phase"), nb::arg("s_amp"),
             nb::arg("s_phase"), nb::arg("order") = 0,
             nb::arg("direction") = "forward")

        .def("GridLayer_geteps",
             [](RCWA& self, const nb::object& ep_all) {
                 // grcwa: flat eps array, or [epsx, epsy, epsz] (anisotropic).
                 if (nb::isinstance<nb::list>(ep_all) ||
                     nb::isinstance<nb::tuple>(ep_all)) {
                     nb::sequence s = nb::cast<nb::sequence>(ep_all);
                     if (nb::len(s) > 0 && !is_scalar(s[0]))
                         raise_notimplemented(
                             "anisotropic GridLayer_geteps is not "
                             "implemented in cpprcwa");
                     self.GridLayer_geteps(to_complex_vec(s));
                 } else {
                     self.GridLayer_geteps(to_complex_vec(ep_all));
                 }
             },
             nb::arg("ep_all"))

        // ── solve ──
        .def("RT_Solve",
             [](RCWA& self, int normalize, int byorder) {
                 RTResult rt = self.RT_Solve(normalize != 0, byorder != 0);
                 return nb::make_tuple(rt.R, rt.T);
             },
             nb::arg("normalize") = 0, nb::arg("byorder") = 0)

        // ── amplitudes / fields ──
        .def("GetAmplitudes_noTranslate",
             [](RCWA& self, int which_layer) {
                 auto [ai, bi] = self.GetAmplitudes_noTranslate(which_layer);
                 return nb::make_tuple(owned_vec(ai), owned_vec(bi));
             },
             nb::arg("which_layer"))
        .def("GetAmplitudes",
             [](RCWA& self, int which_layer, double z_offset) {
                 auto [ai, bi] = self.GetAmplitudes(which_layer, z_offset);
                 return nb::make_tuple(owned_vec(ai), owned_vec(bi));
             },
             nb::arg("which_layer"), nb::arg("z_offset"))

        .def("Solve_FieldFourier",
             [](RCWA& self, int which_layer, const nb::object& z_offset) {
                 auto fields = self.Solve_FieldFourier(which_layer,
                                                       to_zoffsets(z_offset));
                 nb::list out;
                 for (const auto& f : fields) out.append(field_fourier_to_list(f));
                 return nb::object(out);
             },
             nb::arg("which_layer"), nb::arg("z_offset"))

        .def("Solve_FieldOnGrid",
             [](RCWA& self, int which_layer, const nb::object& z_offset,
                const nb::object& Nxy) {
                 std::optional<std::array<int, 2>> nxy;
                 if (!Nxy.is_none())
                     nxy = nb::cast<std::array<int, 2>>(Nxy);
                 auto fields = self.Solve_FieldOnGrid(which_layer,
                                                      to_zoffsets(z_offset), nxy);
                 nb::list out;
                 for (const auto& f : fields) out.append(field_grid_to_list(f));
                 return nb::object(out);
             },
             nb::arg("which_layer"), nb::arg("z_offset"), nb::arg("Nxy") = nb::none())

        .def("Solve_FieldFourierSelective",
             [](RCWA& self, int which_layer, const nb::object& z_offset,
                bool include_forward, bool include_backward) {
                 FieldSelection sel;
                 sel.include_forward = include_forward;
                 sel.include_backward = include_backward;
                 auto fields = self.Solve_FieldFourierSelective(
                     which_layer, to_zoffsets(z_offset), sel);
                 nb::list out;
                 for (const auto& f : fields) out.append(field_fourier_to_list(f));
                 return nb::object(out);
             },
             nb::arg("which_layer"), nb::arg("z_offset"),
             nb::arg("include_forward") = true,
             nb::arg("include_backward") = true)

        .def("Solve_FieldOnGridSelective",
             [](RCWA& self, int which_layer, const nb::object& z_offset,
                bool include_forward, bool include_backward, const nb::object& Nxy) {
                 FieldSelection sel;
                 sel.include_forward = include_forward;
                 sel.include_backward = include_backward;
                 std::optional<std::array<int, 2>> nxy;
                 if (!Nxy.is_none())
                     nxy = nb::cast<std::array<int, 2>>(Nxy);
                 auto fields = self.Solve_FieldOnGridSelective(
                     which_layer, to_zoffsets(z_offset), sel, nxy);
                 nb::list out;
                 for (const auto& f : fields) out.append(field_grid_to_list(f));
                 return nb::object(out);
             },
             nb::arg("which_layer"), nb::arg("z_offset"),
             nb::arg("include_forward") = true,
             nb::arg("include_backward") = true,
             nb::arg("Nxy") = nb::none())

        .def("Return_eps", &RCWA::Return_eps,
             nb::arg("which_layer"), nb::arg("Nx"), nb::arg("Ny"),
             nb::arg("component") = "xx")

        .def("Volume_integral",
             [](RCWA& self, int which_layer, const ComplexMatrix& Mx,
                const ComplexMatrix& My, const ComplexMatrix& Mz, int normalize) {
                 return nb::cast(self.Volume_integral(which_layer, Mx, My, Mz,
                                                      normalize != 0));
             },
             nb::arg("which_layer"), nb::arg("Mx"), nb::arg("My"),
             nb::arg("Mz"), nb::arg("normalize") = 0)

        .def("Solve_ZStressTensorIntegral", &RCWA::Solve_ZStressTensorIntegral,
             nb::arg("which_layer"))

        // ── attributes (grcwa parity) ──
        .def_prop_ro("nG", &RCWA::nG)
        .def_prop_ro("Layer_N", &RCWA::Layer_N)
        .def_prop_ro("normalization", &RCWA::normalization)
        .def_prop_ro("G", &RCWA::G)
        .def_prop_ro("kx", &RCWA::kx)
        .def_prop_ro("ky", &RCWA::ky)
        .def_prop_ro("thickness_list", &RCWA::thickness_list)
        // kp/q/phi are shared per distinct ε; expose as per-layer lists.
        .def_prop_ro("q_list", [](RCWA& self) {
            nb::list out;
            for (const auto& q : self.q_list()) out.append(nb::cast(*q));
            return nb::object(out);
        })
        .def_prop_ro("phi_list", [](RCWA& self) {
            nb::list out;
            for (const auto& p : self.phi_list()) out.append(nb::cast(*p));
            return nb::object(out);
        })
        .def_prop_ro("kp_list", [](RCWA& self) {
            nb::list out;
            for (const auto& k : self.kp_list()) out.append(nb::cast(*k));
            return nb::object(out);
        })
        .def_prop_ro("Uniform_ep_list", &RCWA::uniform_eps_list)
        .def_prop_ro("GridLayer_Nxy_list", &RCWA::grid_Nxy_list)
        .def_prop_ro("Patterned_epinv_list", &RCWA::patterned_epinv_list)
        .def_prop_ro("Patterned_ep2_list", &RCWA::patterned_ep2_list)
        .def_prop_ro("freq", &RCWA::freq)
        .def_prop_ro("omega", &RCWA::omega)
        .def_prop_ro("L1", &RCWA::L1)
        .def_prop_ro("L2", &RCWA::L2)
        .def_prop_ro("theta", &RCWA::theta)
        .def_prop_ro("phi", &RCWA::phi)
        .def_prop_ro("a0", &RCWA::a0)
        .def_prop_ro("bN", &RCWA::bN)
        .def_prop_ro("direction",
                     [](RCWA& self) { return std::string(self.direction_str()); })
        .def_prop_ro("id_list",
                     [](RCWA& self) {
                         nb::list out;
                         for (int li = 0; li < self.Layer_N(); ++li) {
                             nb::list row;
                             if (self.layer_types()[li] == LayerType::Uniform) {
                                 row.append(0); row.append(li);
                                 row.append(self.material_idx()[li]);
                             } else {
                                 row.append(1); row.append(li);
                                 row.append(self.material_idx()[li]);
                                 row.append(self.grid_idx()[li]);
                             }
                             out.append(row);
                         }
                         return nb::object(out);
                     })
        .def_prop_ro("quasi1d", &RCWA::quasi1d);

    // ── module-level helpers (grcwa.fft_funs / grcwa.kbloch parity) ────────
    m.def("Lattice_Reciprocate",
          [](const nb::object& L1, const nb::object& L2) {
              return Lattice_Reciprocate(to_vec2(L1), to_vec2(L2));
          },
          nb::arg("L1"), nb::arg("L2"));

    m.def("Lattice_getG",
          [](int nG, const nb::object& Lk1, const nb::object& Lk2, int method) {
              return Lattice_getG(nG, to_vec2(Lk1), to_vec2(Lk2), method);
          },
          nb::arg("nG"), nb::arg("Lk1"), nb::arg("Lk2"), nb::arg("method") = 0);

    m.def("Lattice_SetKs",
          [](const IntMatrix& G, complex kx0, complex ky0,
             const nb::object& Lk1, const nb::object& Lk2) {
              ComplexVector kx, ky;
              Lattice_SetKs(G, kx0, ky0, to_vec2(Lk1), to_vec2(Lk2), kx, ky);
              return nb::make_tuple(std::move(kx), std::move(ky));
          },
          nb::arg("G"), nb::arg("kx0"), nb::arg("ky0"),
          nb::arg("Lk1"), nb::arg("Lk2"));

    m.def("get_fft",
          [](double dN, const nb::ndarray<complex, nb::ndim<2>, nb::c_contig>& s_in,
             const IntMatrix& G) {
              int Nx = (int)s_in.shape(0);
              int Ny = (int)s_in.shape(1);
              const complex* p = s_in.data();
              std::vector<complex> flat(p, p + (size_t)Nx * Ny);
              return get_fft(dN, flat, Nx, Ny, G);
          },
          nb::arg("dN"), nb::arg("s_in"), nb::arg("G"));

    m.def("get_ifft",
          [](int Nx, int Ny, const ComplexVector& s_in, const IntMatrix& G) {
              double dN = 1.0 / ((double)Nx * Ny);
              return get_ifft(dN, Nx, Ny, s_in, G);
          },
          nb::arg("Nx"), nb::arg("Ny"), nb::arg("s_in"), nb::arg("G"));

    m.def("Epsilon_fft",
          [](double dN, const nb::object& eps_grid, const IntMatrix& G) {
              // isotropic: 2D array; anisotropic: [epsx, epsy, epsz].
              if (nb::isinstance<nb::list>(eps_grid) ||
                  nb::isinstance<nb::tuple>(eps_grid)) {
                  nb::sequence s = nb::cast<nb::sequence>(eps_grid);
                  if (nb::len(s) == 3 && !is_scalar(s[0])) {
                      std::vector<std::vector<complex>> grids;
                      for (nb::handle g : s)
                          grids.push_back(to_complex_vec(g));
                      auto g0 = nb::cast<nb::ndarray<complex, nb::ndim<2>, nb::c_contig>>(s[0]);
                      int Nx = (int)g0.shape(0);
                      int Ny = (int)g0.shape(1);
                      auto r = Epsilon_fft(dN, grids, Nx, Ny, G);
                      return nb::make_tuple(std::move(r.epsinv), std::move(r.eps2));
                  }
              }
              auto a = nb::cast<nb::ndarray<complex, nb::ndim<2>, nb::c_contig>>(eps_grid);
              int Nx = (int)a.shape(0);
              int Ny = (int)a.shape(1);
              auto r = Epsilon_fft(dN, to_complex_vec(eps_grid), Nx, Ny, G);
              return nb::make_tuple(std::move(r.epsinv), std::move(r.eps2));
          },
          nb::arg("dN"), nb::arg("eps_grid"), nb::arg("G"));
}
