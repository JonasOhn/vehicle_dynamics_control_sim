#include <functional>
#include <memory>
#include <sstream>
#include <string>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include "sim_backend/msg/vehicle_state.hpp"


class VehicleFramePublisher : public rclcpp::Node
{
public:
  VehicleFramePublisher()
  : Node("vehicle_frame_publisher",
                rclcpp::NodeOptions()
                    .allow_undeclared_parameters(true)
                    .automatically_declare_parameters_from_overrides(true))
  {
    // acquire `tf_name` parameter
    frame_name_ = this->get_parameter("frame_name").as_string();

    // Initialize the transform broadcaster
    tf_broadcaster_ =
      std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    subscription_ = this->create_subscription<sim_backend::msg::VehicleState>(
      "vehicle_state", 10,
      std::bind(&VehicleFramePublisher::handle_vehicle_state, this, std::placeholders::_1));
  }

private:
  void handle_vehicle_state(const std::shared_ptr<sim_backend::msg::VehicleState> msg)
  {
    geometry_msgs::msg::TransformStamped t;

    // Read message content and assign it to
    // corresponding tf variables
    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "world";
    t.child_frame_id = frame_name_.c_str();

    t.transform.translation.x = msg->x_c;
    t.transform.translation.y = msg->y_c;
    t.transform.translation.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, msg->psi);
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();

    // Send the transformation
    tf_broadcaster_->sendTransform(t);
  }

  rclcpp::Subscription<sim_backend::msg::VehicleState>::SharedPtr subscription_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::string frame_name_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VehicleFramePublisher>());
  rclcpp::shutdown();
  return 0;
}