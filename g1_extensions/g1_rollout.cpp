#include "g1_rollout.h"
#include <pybind11/stl.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <chrono>
#include <sstream>

namespace py = pybind11;

// ONNX Policy wrapper class
class OnnxPolicy {
public:
    explicit OnnxPolicy(const std::shared_ptr<Ort::Session>& session)
        : session_(session), memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
        Ort::Allocator allocator(*session_, memory_info_);
        input_name_  = session_->GetInputNameAllocated(0, allocator).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
        input_shape_  = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        output_shape_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        input_size_ = static_cast<int>(input_shape_[1]);
        output_size_ = static_cast<int>(output_shape_[1]);
    }

    std::vector<float> run(const std::vector<float>& observation) {
        if ((int)observation.size() != input_size_) {
            throw std::runtime_error("Observation size does not match ONNX input dimension");
        }
        std::array<int64_t, 2> ishape = { 1, static_cast<int64_t>(observation.size()) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float*>(observation.data()), observation.size(), ishape.data(), 2);
        const char* in_names[1] = { input_name_.c_str() };
        const char* out_names[1] = { output_name_.c_str() };
        auto outputs = session_->Run(run_options_, in_names, &input_tensor, 1, out_names, 1);
        auto& out = outputs[0];
        float* ptr = out.GetTensorMutableData<float>();
        auto info = out.GetTensorTypeAndShapeInfo();
        size_t n = info.GetElementCount();
        return std::vector<float>(ptr, ptr + n);
    }

    int input_size() const { return input_size_; }
    int output_size() const { return output_size_; }

private:
    std::shared_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    Ort::RunOptions run_options_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    int input_size_ = 0;
    int output_size_ = 0;
};

// Utility functions
py::array_t<double> make_array_owned_g1(std::vector<double>& buf, int B, int T, int D) {
    std::vector<ssize_t> shape   = { B, T, D };
    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(sizeof(double) * T * D),
        static_cast<ssize_t>(sizeof(double) *     D),
        static_cast<ssize_t>(sizeof(double))
    };
    auto heap_buf = new std::vector<double>(std::move(buf));
    py::capsule free_when_done(heap_buf, [](void *p) { delete reinterpret_cast<std::vector<double>*>(p); });
    return py::array_t<double>(shape, strides, heap_buf->data(), free_when_done);
}

// ONNX session allocator
static std::shared_ptr<Ort::Session> allocate_shared_session(const std::string& onnx_path) {
    static Ort::Env env;
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    return std::make_shared<Ort::Session>(env, onnx_path.c_str(), opts);
}

// Metadata structure for G1 policy
struct G1Metadata {
    int num_joints;
    int obs_dim;
    std::vector<double> default_joint_pos;
    std::vector<double> action_scale;
    std::vector<std::string> joint_names;
    std::vector<int> arm_joint_indices;

    G1Metadata() : num_joints(29), obs_dim(99) {}
};

// Global metadata storage
static G1Metadata g_g1_metadata;
static bool g_g1_metadata_loaded = false;

// Helper function to parse comma-separated values
static std::vector<double> parse_csv_floats(const std::string& csv) {
    std::vector<double> result;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stod(item));
    }
    return result;
}

static std::vector<std::string> parse_csv_strings(const std::string& csv) {
    std::vector<std::string> result;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(item);
    }
    return result;
}

// Parse metadata from ONNX model using ONNX Runtime session
static void parse_g1_metadata(const std::shared_ptr<Ort::Session>& session) {
    if (g_g1_metadata_loaded) return;

    try {
        // Get model metadata from session
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::ModelMetadata metadata = session->GetModelMetadata();

        // Parse metadata properties
        std::unordered_map<std::string, std::string> metadata_map;

        // Try to get custom metadata keys
        std::vector<std::string> keys_to_try = {
            "default_joint_pos", "action_scale", "joint_names"
        };

        for (const auto& key : keys_to_try) {
            try {
                Ort::AllocatedStringPtr value_ptr = metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator);
                if (value_ptr) {
                    metadata_map[key] = std::string(value_ptr.get());
                }
            } catch (...) {
                // Key not found, skip
            }
        }

        // Parse default_joint_pos
        if (metadata_map.count("default_joint_pos")) {
            g_g1_metadata.default_joint_pos = parse_csv_floats(metadata_map["default_joint_pos"]);
            g_g1_metadata.num_joints = static_cast<int>(g_g1_metadata.default_joint_pos.size());
        } else {
            // Fallback to zeros
            g_g1_metadata.default_joint_pos.assign(29, 0.0);
        }

        // Parse action_scale
        if (metadata_map.count("action_scale")) {
            g_g1_metadata.action_scale = parse_csv_floats(metadata_map["action_scale"]);
        } else {
            // Fallback to 0.25 for all joints
            g_g1_metadata.action_scale.assign(g_g1_metadata.num_joints, 0.25);
        }

        // Parse joint_names
        if (metadata_map.count("joint_names")) {
            g_g1_metadata.joint_names = parse_csv_strings(metadata_map["joint_names"]);
        }

    } catch (const std::exception& e) {
        // If metadata parsing fails, use defaults
        g_g1_metadata.default_joint_pos.assign(29, 0.0);
        g_g1_metadata.action_scale.assign(29, 0.25);
        g_g1_metadata.num_joints = 29;
    }

    // Arm joint indices (15-28) - hard-coded as in simulate_g1.py
    g_g1_metadata.arm_joint_indices = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};

    // Calculate observation dimension
    g_g1_metadata.obs_dim = 3 + 3 + 3 + g_g1_metadata.num_joints * 3 + 3;

    g_g1_metadata_loaded = true;
}

static std::string get_g1_policy_path() {
    // Check G1_EXTENSIONS_POLICY_DIR env var first (set by g1_extensions/__init__.py)
    const char* policy_dir = std::getenv("G1_EXTENSIONS_POLICY_DIR");
    if (policy_dir) {
        return std::string(policy_dir) + "/g1_velocity_policy.onnx";
    }
    // Fallback to relative path
    return std::string("g1_extensions/policy/g1_velocity_policy.onnx");
}

// =============================================================================
// G1ThreadPool Implementation
// =============================================================================

G1ThreadPool::G1ThreadPool(int num_threads)
    : num_threads_(num_threads), stop_(false), active_workers_(0), total_tasks_(0), completed_tasks_(0) {
    if (num_threads_ > 0) {
        threads_.reserve(num_threads_);
        for (int i = 0; i < num_threads_; ++i) {
            threads_.emplace_back(&G1ThreadPool::worker_thread, this);
        }
    }
}

G1ThreadPool::~G1ThreadPool() {
    if (num_threads_ > 0) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread &worker : threads_) {
            worker.join();
        }
    }
}

void G1ThreadPool::execute_parallel(std::function<void(int)> func, int total_work) {
    if (num_threads_ == 0) {
        // Single-threaded execution
        for (int i = 0; i < total_work; ++i) {
            func(i);
        }
    } else {
        // Multi-threaded execution
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            total_tasks_ = total_work;
            completed_tasks_ = 0;

            for (int i = 0; i < total_work; ++i) {
                tasks_.push([func, i]() { func(i); });
            }
        }
        condition_.notify_all();

        std::unique_lock<std::mutex> lock(queue_mutex_);
        finished_.wait(lock, [this]() { return completed_tasks_ == total_tasks_; });
    }
}

void G1ThreadPool::worker_thread() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty())
                return;

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        task();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            completed_tasks_++;
            if (completed_tasks_ == total_tasks_) {
                finished_.notify_one();
            }
        }
    }
}

// =============================================================================
// G1Rollout Implementation
// =============================================================================

G1Rollout::G1Rollout(int nthread, double cutoff_time) : num_threads_(nthread), cutoff_time_(cutoff_time) {
    initialize_policy();
    if (num_threads_ != 0) {
        thread_pool_ = std::make_unique<G1ThreadPool>(num_threads_);
    }
}

G1Rollout::~G1Rollout() {
    close();
}

void G1Rollout::close() {
    if (!closed_) {
        thread_pool_.reset();
        policy_.reset();
        onnx_session_.reset();
        closed_ = true;
    }
}

void G1Rollout::initialize_policy() {
    std::string policy_path = get_g1_policy_path();
    onnx_session_ = allocate_shared_session(policy_path);
    parse_g1_metadata(onnx_session_);
    policy_ = std::make_unique<OnnxPolicy>(onnx_session_);
}

py::tuple G1Rollout::rollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>& data,
    const py::array_t<double>& initial_state,
    const py::array_t<double>& controls
) {
    if (closed_) {
        throw std::runtime_error("Rollout requested after object was closed");
    }

    int B = (int)models.size();
    if (B == 0 || B != (int)data.size()) {
        throw std::runtime_error("models/data must have same non-zero length");
    }

    int horizon = (int)controls.shape(1);
    const mjModel* m0 = models[0];
    int nq = m0->nq;
    int nv = m0->nv;
    int nu = m0->nu;
    int nsens = m0->nsensordata;
    int nstate = nq + nv;

    if (initial_state.ndim() != 2 || initial_state.shape(0) != B || initial_state.shape(1) != nstate) {
        throw std::runtime_error("initial_state must be a 2D array of shape (B, nq+nv)");
    }

    // Controls should be (B, horizon, 17) for G1 - [vx, vy, wz, left_arm(7), right_arm(7)]
    // If controls are all zeros for arm indices, policy output will be used instead
    if (controls.ndim() != 3 || controls.shape(0) != B || controls.shape(2) != 17) {
        throw std::runtime_error("controls must be a 3D array of shape (B, horizon, 17)");
    }

    std::vector<double> states_buf(B * (horizon + 1) * nstate);
    std::vector<double> sens_buf(B * horizon * nsens);

    auto controls_unchecked = controls.unchecked<3>();
    const double* x0_ptr = initial_state.data();

    std::vector<std::vector<float>> prev_policy(B, std::vector<float>(g_g1_metadata.num_joints, 0.0f));

    {
        py::gil_scoped_release release;

        auto execute_work = [&](int i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            mjData* d = data[i];
            const mjModel* m = models[i];

            d->time = 0.0;
            const double* x0_i = x0_ptr + i * nstate;
            mj_setState(m, d, x0_i, mjSTATE_QPOS | mjSTATE_QVEL);
            mj_forward(m, d);
            mju_zero(d->qacc_warmstart, m->nv);

            double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
            double* se_ptr = &sens_buf[i * horizon * nsens];

            // Store initial state
            for (int j = 0; j < nq; j++) st_ptr[j] = d->qpos[j];
            for (int j = 0; j < nv; j++) st_ptr[nq + j] = d->qvel[j];

            int base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start;
            compute_indices(m, base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start);

            for (int t = 0; t < horizon; t++) {
                // Check timeout before each step
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration<double>(current_time - start_time).count();
                if (elapsed > cutoff_time_) {
                    // Timeout reached, fill remaining states with current state and return
                    for (int remaining_t = t; remaining_t < horizon; remaining_t++) {
                        for (int j = 0; j < nq; j++) st_ptr[(remaining_t + 1) * nstate + j] = d->qpos[j];
                        for (int j = 0; j < nv; j++) st_ptr[(remaining_t + 1) * nstate + nq + j] = d->qvel[j];
                        for (int j = 0; j < nsens; j++) se_ptr[remaining_t * nsens + j] = d->sensordata[j];
                    }
                    return;
                }

                // Get full control command [vx, vy, wz, left_arm(7), right_arm(7)]
                std::vector<float> obs;
                double cmd_vel_buf[3];
                double arm_cmd_buf[14];  // left_arm(7) + right_arm(7)

                // Extract velocity commands
                for (int j = 0; j < 3; j++) {
                    cmd_vel_buf[j] = static_cast<double>(controls_unchecked(i, t, j));
                }

                // Extract arm commands
                for (int j = 0; j < 14; j++) {
                    arm_cmd_buf[j] = static_cast<double>(controls_unchecked(i, t, 3 + j));
                }

                // Build observation for policy (using velocity commands only)
                build_observation(m, d, cmd_vel_buf, prev_policy[i],
                                base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start, obs);

                // Run policy
                auto policy_out_vec = policy_->run(obs);

                // Compute control from policy output and user arm commands
                std::vector<double> ctrl;
                compute_control_from_policy(policy_out_vec.data(), arm_cmd_buf, ctrl);

                if ((int)ctrl.size() != nu) {
                    throw std::runtime_error("Computed control size does not match model nu");
                }
                for (int j = 0; j < nu; j++) d->ctrl[j] = ctrl[j];

                // Step simulation
                mj_step(m, d);
                prev_policy[i] = std::move(policy_out_vec);

                // Store state and sensor data
                for (int j = 0; j < nq; j++) st_ptr[(t + 1) * nstate + j] = d->qpos[j];
                for (int j = 0; j < nv; j++) st_ptr[(t + 1) * nstate + nq + j] = d->qvel[j];
                for (int j = 0; j < nsens; j++) se_ptr[t * nsens + j] = d->sensordata[j];
            }
        };

        if (num_threads_ == 0) {
            // Single-threaded execution
            for (int i = 0; i < B; ++i) {
                execute_work(i);
            }
        } else {
            // Multi-threaded execution
            thread_pool_->execute_parallel(execute_work, B);
        }
    }

    auto states_arr = make_array_owned_g1(states_buf, B, horizon + 1, nstate);
    auto sens_arr = make_array_owned_g1(sens_buf, B, horizon, nsens);
    return py::make_tuple(states_arr, sens_arr);
}

G1Rollout* G1Rollout::__enter__() {
    return this;
}

void G1Rollout::__exit__(py::object exc_type, py::object exc_val, py::object exc_tb) {
    close();
}

int G1Rollout::get_num_threads() const {
    return num_threads_;
}

// Helper method implementations
void G1Rollout::compute_indices(const mjModel* m, int& base_qpos_start, int& base_qvel_start,
                                int& leg_qpos_start, int& leg_qvel_start) {
    // Find the free joint (root body)
    int free_joint_idx = -1;
    for (int j = 0; j < m->njnt; j++) {
        if (m->jnt_type[j] == mjJNT_FREE) {
            free_joint_idx = j;
            break;
        }
    }
    if (free_joint_idx == -1) {
        free_joint_idx = 0;
    }
    base_qpos_start = m->jnt_qposadr[free_joint_idx];
    base_qvel_start = m->jnt_dofadr[free_joint_idx];
    // G1 has 29 controlled joints starting after the free joint
    leg_qpos_start = base_qpos_start + 7;  // After pos(3) + quat(4)
    leg_qvel_start = base_qvel_start + 6;  // After lin_vel(3) + ang_vel(3)
}

void G1Rollout::build_observation(const mjModel* m, mjData* d, const double* command_ptr,
                                  const std::vector<float>& prev_policy,
                                  int base_qpos_start, int base_qvel_start,
                                  int leg_qpos_start, int leg_qvel_start,
                                  std::vector<float>& obs_out) {
    obs_out.resize(g_g1_metadata.obs_dim);
    int off = 0;

    // Get root body quaternion
    double quat[4];
    for (int i = 0; i < 4; i++) {
        quat[i] = d->qpos[base_qpos_start + 3 + i];
    }

    // Compute inverse quaternion for body frame transformations
    double invq[4];
    mju_negQuat(invq, quat);

    // Compute rotation matrix from quaternion
    double rot_mat[9];
    mju_quat2Mat(rot_mat, quat);

    // Base linear velocity in body frame (3)
    double base_lin_vel_world[3];
    for (int i = 0; i < 3; i++) {
        base_lin_vel_world[i] = d->qvel[base_qvel_start + i];
    }
    double base_lin_vel_body[3];
    mju_rotVecQuat(base_lin_vel_body, base_lin_vel_world, invq);
    for (int i = 0; i < 3; i++) obs_out[off++] = static_cast<float>(base_lin_vel_body[i]);

    // Base angular velocity in body frame (3)
    double base_ang_vel_world[3];
    for (int i = 0; i < 3; i++) {
        base_ang_vel_world[i] = d->qvel[base_qvel_start + 3 + i];
    }
    double base_ang_vel_body[3];
    mju_rotVecQuat(base_ang_vel_body, base_ang_vel_world, invq);
    for (int i = 0; i < 3; i++) obs_out[off++] = static_cast<float>(base_ang_vel_body[i]);

    // Projected gravity in body frame (3)
    double gvec[3] = {0.0, 0.0, -1.0};
    double gvec_rotated[3];
    mju_rotVecQuat(gvec_rotated, gvec, invq);
    for (int i = 0; i < 3; i++) obs_out[off++] = static_cast<float>(gvec_rotated[i]);

    // Joint positions relative to default
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        double joint_pos = d->qpos[leg_qpos_start + i] - g_g1_metadata.default_joint_pos[i];
        obs_out[off++] = static_cast<float>(joint_pos);
    }

    // Joint velocities
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        obs_out[off++] = static_cast<float>(d->qvel[leg_qvel_start + i]);
    }

    // Previous actions
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        obs_out[off++] = (i < (int)prev_policy.size() ? prev_policy[i] : 0.0f);
    }

    // Command (3): [vx, vy, wz]
    for (int i = 0; i < 3; i++) {
        obs_out[off++] = static_cast<float>(command_ptr[i]);
    }
}

void G1Rollout::compute_control_from_policy(const float* policy_out,
                                           const double* arm_commands,
                                           std::vector<double>& ctrl_out) {
    ctrl_out.resize(g_g1_metadata.num_joints);

    // Apply action scaling and add default positions for all joints
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        ctrl_out[i] = static_cast<double>(policy_out[i]) * g_g1_metadata.action_scale[i] + g_g1_metadata.default_joint_pos[i];
    }

    // Override arm joints with user commands if provided (non-zero)
    // arm_commands layout: [left_arm(7), right_arm(7)]
    // arm_joint_indices contains the indices of arm joints in the full joint array
    for (size_t i = 0; i < g_g1_metadata.arm_joint_indices.size(); i++) {
        int joint_idx = g_g1_metadata.arm_joint_indices[i];
        double user_cmd = arm_commands[i];

        // If user command is non-zero, use it; otherwise use policy output
        if (std::abs(user_cmd) > 1e-8) {
            ctrl_out[joint_idx] = user_cmd;
        }
        // else: keep policy output (already set above)
    }
}

// =============================================================================
// SimG1 - Single-step simulation with G1 policy
// =============================================================================

py::array_t<float> SimG1(
    const mjModel* model,
    mjData* data,
    const py::array_t<double>& x0,
    const py::array_t<double>& command,
    const py::array_t<float>& prev_policy
) {
    static std::shared_ptr<Ort::Session> onnx_session = nullptr;
    static std::unique_ptr<OnnxPolicy> policy = nullptr;

    // Initialize policy on first call
    if (!onnx_session || !policy) {
        std::string policy_path = get_g1_policy_path();
        onnx_session = allocate_shared_session(policy_path);
        parse_g1_metadata(onnx_session);
        policy = std::make_unique<OnnxPolicy>(onnx_session);
    }

    int nq = model->nq;
    int nv = model->nv;
    int nu = model->nu;

    if (x0.size() != nq + nv) {
        throw std::runtime_error("x0 size must equal nq + nv");
    }
    if (command.size() != 17) {
        throw std::runtime_error("command size must be 17 [vx, vy, wz, left_arm(7), right_arm(7)]");
    }
    if (prev_policy.size() != g_g1_metadata.num_joints) {
        throw std::runtime_error("prev_policy size must match num_joints from metadata");
    }

    // Set initial state
    const double* x0_ptr = x0.data();
    mj_setState(model, data, x0_ptr, mjSTATE_QPOS | mjSTATE_QVEL);
    mj_forward(model, data);

    // Convert prev_policy to vector
    std::vector<float> prev_policy_vec(g_g1_metadata.num_joints);
    const float* prev_policy_ptr = prev_policy.data();
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        prev_policy_vec[i] = prev_policy_ptr[i];
    }

    // Get command data - extract velocity commands and arm commands
    const double* cmd_ptr = command.data();
    double cmd_vel[3] = {cmd_ptr[0], cmd_ptr[1], cmd_ptr[2]};
    double arm_cmd[14];  // left_arm(7) + right_arm(7)
    for (int i = 0; i < 14; i++) {
        arm_cmd[i] = cmd_ptr[3 + i];
    }

    // Compute indices for root and joint positions/velocities
    int base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start;
    int free_joint_idx = -1;
    for (int j = 0; j < model->njnt; j++) {
        if (model->jnt_type[j] == mjJNT_FREE) {
            free_joint_idx = j;
            break;
        }
    }
    if (free_joint_idx == -1) {
        free_joint_idx = 0;
    }
    base_qpos_start = model->jnt_qposadr[free_joint_idx];
    base_qvel_start = model->jnt_dofadr[free_joint_idx];
    leg_qpos_start = base_qpos_start + 7;
    leg_qvel_start = base_qvel_start + 6;

    // Build observation vector
    std::vector<float> obs(g_g1_metadata.obs_dim);
    int off = 0;

    // Get root body quaternion
    double quat[4];
    for (int i = 0; i < 4; i++) {
        quat[i] = data->qpos[base_qpos_start + 3 + i];
    }

    // Compute inverse quaternion for body frame transformations
    double invq[4];
    mju_negQuat(invq, quat);

    // Base linear velocity in body frame (3)
    double base_lin_vel_world[3];
    for (int i = 0; i < 3; i++) {
        base_lin_vel_world[i] = data->qvel[base_qvel_start + i];
    }
    double base_lin_vel_body[3];
    mju_rotVecQuat(base_lin_vel_body, base_lin_vel_world, invq);
    for (int i = 0; i < 3; i++) obs[off++] = static_cast<float>(base_lin_vel_body[i]);

    // Base angular velocity in body frame (3)
    double base_ang_vel_world[3];
    for (int i = 0; i < 3; i++) {
        base_ang_vel_world[i] = data->qvel[base_qvel_start + 3 + i];
    }
    double base_ang_vel_body[3];
    mju_rotVecQuat(base_ang_vel_body, base_ang_vel_world, invq);
    for (int i = 0; i < 3; i++) obs[off++] = static_cast<float>(base_ang_vel_body[i]);

    // Projected gravity in body frame (3)
    double gvec[3] = {0.0, 0.0, -1.0};
    double gvec_rotated[3];
    mju_rotVecQuat(gvec_rotated, gvec, invq);
    for (int i = 0; i < 3; i++) obs[off++] = static_cast<float>(gvec_rotated[i]);

    // Joint positions relative to default
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        double joint_pos = data->qpos[leg_qpos_start + i] - g_g1_metadata.default_joint_pos[i];
        obs[off++] = static_cast<float>(joint_pos);
    }

    // Joint velocities
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        obs[off++] = static_cast<float>(data->qvel[leg_qvel_start + i]);
    }

    // Previous actions
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        obs[off++] = prev_policy_vec[i];
    }

    // Command (3): [vx, vy, wz] - use velocity commands only
    for (int i = 0; i < 3; i++) {
        obs[off++] = static_cast<float>(cmd_vel[i]);
    }

    // Run policy
    auto policy_out_vec = policy->run(obs);

    // Compute control from policy output
    std::vector<double> ctrl(g_g1_metadata.num_joints);

    // Apply action scaling and add default positions for all joints
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        ctrl[i] = static_cast<double>(policy_out_vec[i]) * g_g1_metadata.action_scale[i] + g_g1_metadata.default_joint_pos[i];
    }

    // Override arm joints with user commands if provided (non-zero)
    // arm_cmd layout: [left_arm(7), right_arm(7)]
    for (size_t i = 0; i < g_g1_metadata.arm_joint_indices.size(); i++) {
        int joint_idx = g_g1_metadata.arm_joint_indices[i];
        double user_cmd = arm_cmd[i];

        // If user command is non-zero, use it; otherwise use policy output
        if (std::abs(user_cmd) > 1e-8) {
            ctrl[joint_idx] = user_cmd;
        }
        // else: keep policy output (already set above)
    }

    // Apply control and step simulation
    if (nu != g_g1_metadata.num_joints) {
        throw std::runtime_error("Model nu does not match num_joints from metadata");
    }
    for (int j = 0; j < nu; j++) {
        data->ctrl[j] = ctrl[j];
    }
    mj_step(model, data);

    // Return new policy output as numpy array
    std::vector<ssize_t> shape = {g_g1_metadata.num_joints};
    std::vector<ssize_t> strides = {sizeof(float)};
    auto heap_buf = new std::vector<float>(std::move(policy_out_vec));
    py::capsule free_when_done(heap_buf, [](void *p) { delete reinterpret_cast<std::vector<float>*>(p); });
    return py::array_t<float>(shape, strides, heap_buf->data(), free_when_done);
}
