#include "RealtimeMotionBufferVla.hpp"
#include <zmq.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace RLController
{
    namespace
    {
        int64_t steady_ns()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        // Each existing SuperMimicDeployment::policy_from_buffer maps FROM
        // this Isaac buffer order, including for actors trained in JAKA order.
        constexpr std::array<int, 27> kWireIndexForBufferJoint = {
            0, 6, 12, 1, 7, 13, 25, 19, 2, 8, 14, 26, 20, 3,
            9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23, 18, 24};

        Eigen::Quaternionf orientation(const VlaPoseFrame &frame)
        {
            const auto &q = frame.quaternion;
            return Eigen::Quaternionf(q[0], q[1], q[2], q[3]);
        }

        InterpolatedFrame interpolate(const VlaPoseFrame &a, const VlaPoseFrame &b, float alpha)
        {
            InterpolatedFrame result;
            result.root_pos_w = (1.f - alpha) * Eigen::Map<const Eigen::Vector3f>(a.position.data())
                                + alpha * Eigen::Map<const Eigen::Vector3f>(b.position.data());
            result.root_quat_w = orientation(a).slerp(alpha, orientation(b)).normalized();
            result.dof_pos.resize(kWireIndexForBufferJoint.size());
            for (size_t j = 0; j < kWireIndexForBufferJoint.size(); ++j)
            {
                const auto wire = kWireIndexForBufferJoint[j];
                result.dof_pos[j] = (1.f - alpha) * a.joints[wire] + alpha * b.joints[wire];
            }
            return result;
        }
    }

    RealtimeMotionBufferVla::RealtimeMotionBufferVla(
        int64_t dt_ns, int64_t tolerance_ns, const Eigen::VectorXf &joints,
        const Eigen::Vector3f &pos, const Eigen::Quaternionf &quat, int64_t stale_ns)
        : dt_ns_(dt_ns), stale_ns_(stale_ns)
    {
        if (dt_ns != 20'000'000 || tolerance_ns < 0 ||
            tolerance_ns > (static_cast<int64_t>(kCapacity) - 2) * kFrame / kHz - 4 * dt_ns ||
            stale_ns <= 0)
            throw std::invalid_argument("Invalid 50 Hz SuperMimic VLA window or stale timeout");
        if (joints.size() != 27 || !joints.allFinite() || !pos.allFinite() ||
            !quat.coeffs().allFinite() || std::abs(quat.norm() - 1.f) > .01f)
            throw std::invalid_argument("Invalid default SuperMimic base_link pose");
        delay_units_ = (4 * dt_ns + tolerance_ns) * kHz;
        held_frame_ = {pos, quat.normalized(), joints}; // already in buffer order
    }

    RealtimeMotionBufferVla::RealtimeMotionBufferVla(
        const std::string &endpoint, int hwm, int64_t dt_ns, int64_t tolerance_ns,
        const Eigen::VectorXf &joints, const Eigen::Vector3f &pos,
        const Eigen::Quaternionf &quat, int64_t stale_ns)
        : RealtimeMotionBufferVla(dt_ns, tolerance_ns, joints, pos, quat, stale_ns)
    {
        if (endpoint.empty() || hwm < 1)
            throw std::invalid_argument("VLA ZMQ endpoint and positive HWM required");
        std::promise<void> started;
        auto startup = started.get_future();
        receiver_ = std::thread(&RealtimeMotionBufferVla::receive, this, endpoint, hwm, std::move(started));
        try { startup.get(); }
        catch (...) { receiver_.join(); throw; }
    }

    RealtimeMotionBufferVla::~RealtimeMotionBufferVla()
    {
        stop_ = true;
        if (receiver_.joinable()) receiver_.join();
    }

    void RealtimeMotionBufferVla::receive(std::string endpoint, int hwm, std::promise<void> started)
    {
        bool initialized = false;
        try
        {
            zmq::context_t context(1);
            zmq::socket_t sub(context, zmq::socket_type::sub);
            sub.set(zmq::sockopt::linger, 0);
            sub.set(zmq::sockopt::rcvhwm, hwm);
            sub.set(zmq::sockopt::rcvtimeo, 30);
            sub.set(zmq::sockopt::maxmsgsize, static_cast<int64_t>(kVlaMaxPacketBytes));
            sub.set(zmq::sockopt::subscribe, "pose");
            sub.connect(endpoint);
            started.set_value();
            initialized = true;
            int64_t last_reject_log = -1;
            bool multipart = false;
            while (!stop_)
            {
                zmq::message_t bytes;
                if (!sub.recv(bytes, zmq::recv_flags::none)) continue;
                const auto now = steady_ns();
                const bool more = sub.get(zmq::sockopt::rcvmore);
                const bool single = !multipart && !more;
                multipart = more;
                VlaPosePacket packet;
                if (!single || !decode_vla_pose(bytes.data(), bytes.size(), packet))
                {
                    if (last_reject_log < 0 || now - last_reject_log >= kFrame)
                    {
                        std::fprintf(stderr, "[VLA] Rejected pose: expected single-part JAKA v1, base_link, jaka order, stream_id.\n");
                        last_reject_log = now;
                    }
                    continue;
                }
                std::lock_guard<std::mutex> lock(mutex_);
                accept_locked(packet, now);
            }
        }
        catch (...)
        {
            if (!initialized) started.set_exception(std::current_exception());
            else
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!fault_) fault_ = "VLA receiver stopped";
            }
        }
    }

    void RealtimeMotionBufferVla::check_age_locked(int64_t now)
    {
        if (!fault_ && last_recv_ns_ >= 0 && now - last_recv_ns_ >= stale_ns_)
            fault_ = "VLA stream stale; re-enter SuperZmq to arm a new reference";
    }

    void RealtimeMotionBufferVla::accept_locked(const VlaPosePacket &packet, int64_t now)
    {
        check_age_locked(now); // A late packet cannot hide the timeout between two FSM ticks.
        if (fault_) return;
        if (frames_.empty()) stream_id_ = packet.stream_id;
        if (packet.stream_id != stream_id_)
        {
            fault_ = "VLA publisher session changed; reference origin may have changed";
            return;
        }
        auto first = packet.frames.begin();
        if (!frames_.empty())
        {
            const auto last = frames_.back().index;
            first = std::find_if(first, packet.frames.end(), [last](const auto &f) { return f.index > last; });
            if (first == packet.frames.end()) return; // duplicate packets never refresh liveness
            // A sliding packet may repeat an already buffered knot. The JAKA
            // publisher sends immutable references for a given session/index.
            // A conflicting overlap cannot define a continuous next segment.
            for (auto it = packet.frames.begin(); it != first; ++it)
            {
                if (it->index < frames_.front().index) continue;
                const auto &saved = frames_[static_cast<size_t>(it->index - frames_.front().index)];
                if (it->joints != saved.joints || it->position != saved.position ||
                    std::abs(orientation(*it).dot(orientation(saved))) < 1.f - 1.e-6f)
                {
                    fault_ = "Conflicting VLA reference at a repeated frame index";
                    return;
                }
            }
            if (first->index - last != 1)
            {
                fault_ = "Missing VLA reference after merging overlapping frame windows";
                return;
            }
        }
        if (frames_.size() + static_cast<size_t>(packet.frames.end() - first) > kCapacity)
        {
            // The consumer stopped advancing. Avoid skipping queued commands to jump to the head.
            fault_ = "VLA playback backlog exceeded buffer capacity";
            return;
        }
        frames_.insert(frames_.end(), first, packet.frames.end());
        last_recv_ns_ = now;
    }

    VlaMotionWindow RealtimeMotionBufferVla::sample()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return sample_locked(steady_ns());
    }

    VlaMotionWindow RealtimeMotionBufferVla::sample_locked(int64_t now)
    {
        check_age_locked(now);
        VlaMotionWindow out;
        out.fault_reason = fault_;
        out.fault = fault_ != nullptr;
        if (!frames_.empty())
        {
            out.last_frame_index = frames_.back().index;
            out.receive_age_ns = now - last_recv_ns_;
        }
        const int64_t cap = (static_cast<int64_t>(frames_.size()) - 1) * kFrame - delay_units_;
        if (out.fault || cap < 0)
        {
            // Collapse the lookahead into the last *displayed* base/joint
            // target. Repeating the previous five-point moving window would
            // keep commanding its future displacement indefinitely.
            out.frames.fill(held_frame_);
            out.ready = last_sample_ns_ >= 0;
            return out;
        }

        if (last_sample_ns_ < 0) cursor_ = cap;
        else
        {
            // Repeated reads at the same instant do not advance playback. A
            // delayed control thread cannot execute an unbounded catch-up step.
            const auto elapsed = std::clamp<int64_t>(now - last_sample_ns_, 0, 2 * dt_ns_);
            cursor_ = std::min(cap, cursor_ + elapsed * kHz);
        }
        last_sample_ns_ = now;
        for (size_t i = 0; i < out.frames.size(); ++i)
        {
            const int64_t target = cursor_ + static_cast<int64_t>(i) * dt_ns_ * kHz;
            const auto left = static_cast<size_t>(target / kFrame);
            const auto right = std::min(left + 1, frames_.size() - 1);
            const float alpha = static_cast<float>(static_cast<double>(target % kFrame) / kFrame);
            out.frames[i] = interpolate(frames_[left], frames_[right], alpha);
        }
        // Keep the left interpolation knot and all unread frames.
        while (cursor_ >= kFrame)
        {
            frames_.pop_front();
            cursor_ -= kFrame;
        }
        held_frame_ = out.frames.front();
        out.ready = true;
        return out;
    }
}
