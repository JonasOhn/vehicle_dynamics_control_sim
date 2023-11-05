#include <functional>
#include <memory>
#include <sstream>
#include <string>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/pose.hpp"

class VehicleFramePublisher : public rclcpp::Node
{
public:
  VehicleFramePublisher()
  : Node("vehicle_frame_publisher")
  {
    // Declare and acquire `tf_name` parameter
    frame_name_ = this->declare_parameter<std::string>("tf_name", "vehicle_frame");

    // Initialize the transform broadcaster
    tf_broadcaster_ =
      std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    subscription_ = this->create_subscription<geometry_msgs::msg::Pose>(
      "vehicle_pose", 10,
      std::bind(&VehicleFramePublisher::handle_vehicle_pose, this, std::placeholders::_1));
  }

private:
  void handle_vehicle_pose(const std::shared_ptr<geometry_msgs::msg::Pose> msg)
  {
    geometry_msgs::msg::TransformStamped t;

    // Read message content and assign it to
    // corresponding tf variables
    t.header.stamp = this->now();
    t.header.frame_id = "world";
    t.child_frame_id = frame_name_.c_str();

    t.transform.translation.x = msg->position.x;
    t.transform.translation.y = msg->position.y;

    t.transform.rotation = msg->orientation;

    // Send the transformation
    tf_broadcaster_->sendTransform(t);
  }

  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr subscription_;
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