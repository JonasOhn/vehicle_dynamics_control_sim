#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "rosgraph_msgs/msg/clock.hpp"

using namespace std::chrono_literals;


class SimTimeNode : public rclcpp::Node
{
    public:
        SimTimeNode()
        : Node("sim_time_node",
                rclcpp::NodeOptions()
                    .allow_undeclared_parameters(true)
                    .automatically_declare_parameters_from_overrides(true))
        {
            this->rtf_ = this->get_parameter("real_time_factor").as_double();

            this->dt_ms_ = (int) (1.0 / this->rtf_);

            this->dt_ = std::chrono::milliseconds(dt_ms_);

            time_publisher_ = this->create_publisher<rosgraph_msgs::msg::Clock>("clock", 10);

            sim_timer_ = this->create_wall_timer(this->dt_, std::bind(&SimTimeNode::publish_time, this));

            RCLCPP_INFO_STREAM(this->get_logger(), "Node " << this->get_name() << " initialized.");
        }

    private:

        void publish_time()
        {
            auto time_msg = builtin_interfaces::msg::Time();
            auto clock_msg = rosgraph_msgs::msg::Clock();

            this->current_sim_time_ns_ += 1e6;
            
            if (this->current_sim_time_ns_ >= 1e9)
            {
                RCLCPP_DEBUG_STREAM(this->get_logger(), "Resetting ns.");
                this->current_sim_time_s_ = this->current_sim_time_s_ + 1;
                this->current_sim_time_ns_ = this->current_sim_time_ns_ - 1e9;
            }

            time_msg.nanosec = this->current_sim_time_ns_;
            time_msg.sec = this->current_sim_time_s_;

            clock_msg.clock = time_msg;

            time_publisher_->publish(clock_msg);
        }

        rclcpp::TimerBase::SharedPtr sim_timer_;

        rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr time_publisher_;

        double rtf_ = 1.0;
        int dt_ms_ = 1.0;
        int32_t current_sim_time_s_ = 0;
        uint32_t current_sim_time_ns_ = 0;

        std::chrono::milliseconds dt_{std::chrono::milliseconds(dt_ms_)};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimTimeNode>());
  rclcpp::shutdown();
  return 0;
}