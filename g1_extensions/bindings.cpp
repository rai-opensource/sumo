#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>

#include "rollout.h"
#include "g1_rollout.h"

namespace py = pybind11;

static std::vector<const mjModel*> getModelVector(const py::list& python_models) {
    std::vector<const mjModel*> model_vector;
    model_vector.reserve(python_models.size());
    for (auto&& item : python_models) {
        auto ptr = item.attr("_address").cast<std::uintptr_t>();
        model_vector.push_back(reinterpret_cast<const mjModel*>(ptr));
    }
    return model_vector;
}

static std::vector<mjData*> getDataVector(const py::list& python_data) {
    std::vector<mjData*> data_vector;
    data_vector.reserve(python_data.size());
    for (auto&& item : python_data) {
        auto ptr = item.attr("_address").cast<std::uintptr_t>();
        data_vector.push_back(reinterpret_cast<mjData*>(ptr));
    }
    return data_vector;
}

PYBIND11_MODULE(_g1_extensions, m) {
    // Function to shutdown persistent thread pool
    m.def("shutdown_thread_pool",
          []() {
              ThreadPoolManager::instance().shutdown();
          },
          R"doc(
Shutdown the persistent thread pool.

Call this function to clean up the persistent thread pool when done with rollouts.
The pool will be automatically recreated on the next call to persistent_cpp_rollout.
)doc");

    // Rollout
    m.def("rollout",
          [](const py::list& models,
             const py::list& data,
             const py::array_t<double>& x0,
             const py::array_t<double>& controls)
          {
              auto models_cpp = getModelVector(models);
              auto data_cpp   = getDataVector(data);
              return Rollout(models_cpp, data_cpp, x0, controls);
          },
          py::arg("models"),
          py::arg("data"),
          py::arg("x0"),
          py::arg("controls"),
          R"doc(
Run parallel MuJoCo rollouts.

    Args:
    models:                 length-B list of mujoco._structs.MjModel
    data:                   length-B list of mujoco._structs.MjData
    x0:                     2D array of shape (B, nq+nv), batched initial [qpos;qvel]
    controls:               3D array of shape (B, horizon, nu), batched control inputs

Returns:
    tuple of three np.ndarray:
    states  -> shape (B, horizon+1, nq+nv) - MuJoCo states (includes initial state)
    sensors -> shape (B, horizon, nsensordata) - sensor data
)doc");

    // Sim
    m.def("sim",
          [](py::object model,
             py::object data,
             const py::array_t<double>& x0,
             const py::array_t<double>& controls)
          {
              auto model_ptr = reinterpret_cast<const mjModel*>(model.attr("_address").cast<std::uintptr_t>());
              auto data_ptr  = reinterpret_cast<mjData*>(data.attr("_address").cast<std::uintptr_t>());
              Sim(model_ptr, data_ptr, x0, controls);
          },
          py::arg("model"),
          py::arg("data"),
          py::arg("x0"),
          py::arg("controls"),
          R"doc(
Run a single MuJoCo simulation step.

    Args:
    model:      mujoco._structs.MjModel
    data:       mujoco._structs.MjData
    x0:         1D array of shape (nq+nv), initial [qpos;qvel]
    controls:   1D array of shape (nu), control input

Returns:
    None
)doc");

    // G1 Sim
    m.def("sim_g1",
          [](py::object model,
             py::object data,
             const py::array_t<double>& x0,
             const py::array_t<double>& command,
             const py::array_t<float>& prev_policy)
          {
              auto model_ptr = reinterpret_cast<const mjModel*>(model.attr("_address").cast<std::uintptr_t>());
              auto data_ptr  = reinterpret_cast<mjData*>(data.attr("_address").cast<std::uintptr_t>());
              return SimG1(model_ptr, data_ptr, x0, command, prev_policy);
          },
          py::arg("model"),
          py::arg("data"),
          py::arg("x0"),
          py::arg("command"),
          py::arg("prev_policy"),
          R"doc(
Run a single MuJoCo step with G1 ONNX policy to generate controls.

Args:
    model:       mujoco._structs.MjModel
    data:        mujoco._structs.MjData
    x0:          1D array of shape (nq+nv), initial [qpos;qvel]
    command:     1D array of shape (17), command [vx, vy, wz, left_arm(7), right_arm(7)]
    prev_policy: 1D array of shape (29), previous policy output

Returns:
    1D array of shape (29): New policy output for next iteration
)doc");

    // G1Rollout class - mimicking mujoco.rollout.Rollout API
    py::class_<G1Rollout>(m, "G1Rollout")
        .def(py::init<int, double>(), py::arg("nthread") = 0, py::arg("cutoff_time") = 0.2,
             R"doc(
Create a G1Rollout object with thread pool for parallel rollouts.

Args:
    nthread: Number of threads in pool. If 0, runs single-threaded.
    cutoff_time: Maximum time allowed per rollout in seconds (default 0.2s).
)doc")
        .def("rollout",
             [](G1Rollout& self,
                const py::list& models,
                const py::list& data,
                const py::array_t<double>& initial_state,
                const py::array_t<double>& controls)
             {
                 auto models_cpp = getModelVector(models);
                 auto data_cpp = getDataVector(data);
                 return self.rollout(models_cpp, data_cpp, initial_state, controls);
             },
             py::arg("models"),
             py::arg("data"),
             py::arg("initial_state"),
             py::arg("controls"),
             R"doc(
Perform parallel rollouts with G1 ONNX policy.

Args:
    models: List of mujoco._structs.MjModel instances
    data: List of mujoco._structs.MjData instances
    initial_state: Array of shape (nbatch, nstate) with initial states
    controls: Array of shape (nbatch, nsteps, 17) with command sequences

Returns:
    Tuple of (states, sensordata) arrays matching mujoco.rollout API
)doc")
        .def("close", &G1Rollout::close,
             R"doc(Close the rollout object and cleanup resources.)doc")
        .def("__enter__", &G1Rollout::__enter__, py::return_value_policy::reference_internal)
        .def("__exit__", &G1Rollout::__exit__)
        .def_property_readonly("nthread", &G1Rollout::get_num_threads,
                              R"doc(Number of threads in the thread pool.)doc");
}
