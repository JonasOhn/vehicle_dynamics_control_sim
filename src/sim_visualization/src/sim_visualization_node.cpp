#include "rclcpp/rclcpp.hpp"
#include "sim_backend/msg/vehicle_state.hpp"
#include "sim_backend/msg/point2_d.hpp"
#include "sim_backend/msg/point2_d_array.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/msg/marker_array.hpp>

using namespace std::chrono_literals;


class SimVisualization : public rclcpp::Node
{
  public:
    SimVisualization()
    : Node("sim_visualization",
            rclcpp::NodeOptions()
                .allow_undeclared_parameters(true)
                .automatically_declare_parameters_from_overrides(true))
    {
      /* ====================
        SUBSCRIBERS
      */

      // Subscriber for vehicle state
      state_subscription_ = this->create_subscription<sim_backend::msg::VehicleState>(
          "vehicle_state", 10, std::bind(&SimVisualization::visualize_velvec, this, std::placeholders::_1));

      // Subscriber for track middle points
      middle_path_subscription_ = this->create_subscription<sim_backend::msg::Point2DArray>(
          "track_points2d", 10, std::bind(&SimVisualization::update_middle_path_visual, this, std::placeholders::_1));

      // Subscriber for track left boundary points
      trackbounds_left_subscription_ = this->create_subscription<sim_backend::msg::Point2DArray>(
          "trackbounds_left_points2d", 10, std::bind(&SimVisualization::update_trackbounds_left_visual, this, std::placeholders::_1));
      
      // Subscriber for track right boundary points
      trackbounds_right_subscription_ = this->create_subscription<sim_backend::msg::Point2DArray>(
          "trackbounds_right_points2d", 10, std::bind(&SimVisualization::update_trackbounds_right_visual, this, std::placeholders::_1));
      
      // Subscriber for reference path ahead of vehicle
      ref_path_subscription_ = this->create_subscription<sim_backend::msg::Point2DArray>(
          "reference_path_points2d", 10, std::bind(&SimVisualization::update_ref_path_visual, this, std::placeholders::_1));

      // Subscriber for MPC spline fit
      mpc_spline_subscription_ = this->create_subscription<sim_backend::msg::Point2DArray>(
          "mpc_spline_points2d", 10, std::bind(&SimVisualization::update_mpc_spline_visual, this, std::placeholders::_1));

      // Subscriber for MPC prediction trajectory (to be published as marker array with purple cubes)
      mpc_xy_predict_subscription_ = this->create_subscription<sim_backend::msg::Point2DArray>(
          "mpc_xy_predict_trajectory", 10, std::bind(&SimVisualization::update_mpc_xy_prediction_visual, this, std::placeholders::_1));


      /* ====================
        PUBLISHERS
      */

      // === PointCloud2 Publishers ===
      // track/reference path (small black dots)
      track_publisher_PCL_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("track_pcl", 10);
      // reference points from path planning (red dots)
      ref_path_publisher_PCL_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("reference_path_pcl", 10);

      // Track Boundary Publishers (blue and yellow vertical cylinders)
      trackbounds_left_publisher_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("trackbounds_left_markers", 10);
      trackbounds_right_publisher_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("trackbounds_right_markers", 10);

      // Velocity Vector Publisher (green arrow with alpha channel for transparency)
      velocity_vector_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("velocity_vector_marker", 10);

      // MPC Spline Fit Publisher (green dots)
      mpc_spline_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("mpc_spline_pcl", 10);

      // MPC x,y prediction Publisher (purple cubes)
      mpc_xy_prediction_publisher_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("mpc_xy_prediction_markers", 10);


      RCLCPP_INFO_STREAM(this->get_logger(), "Node " << this->get_name() << " initialized.");
    }

  private:

    void update_mpc_xy_prediction_visual(const sim_backend::msg::Point2DArray & point2d_array)
    {
      // Create a marker array message
      visualization_msgs::msg::MarkerArray mpc_xy_prediction_markers;

      // Fill in the message
      for (size_t i = 0; i < point2d_array.points.size(); i++)
      {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "vehicle_frame";
        marker.header.stamp = this->now();
        marker.ns = "mpc_xy_prediction";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = point2d_array.points[i].x;
        marker.pose.position.y = point2d_array.points[i].y;
        marker.pose.position.z = 0.5;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 0.8;
        mpc_xy_prediction_markers.markers.push_back(marker);
      }

      mpc_xy_prediction_publisher_markers_->publish(mpc_xy_prediction_markers);
    }

    void update_ref_path_visual(const sim_backend::msg::Point2DArray & point2d_array)
    {
      // Create a point cloud message
      sensor_msgs::msg::PointCloud2 ref_path_pcl;
      pcl::PointCloud<pcl::PointXYZRGB> ref_path_cloud;

      // Fill in the message
      ref_path_cloud.header.frame_id = "world";
      ref_path_cloud.height = 1;
      ref_path_cloud.width = point2d_array.points.size();
      ref_path_cloud.points.resize(ref_path_cloud.height * ref_path_cloud.width);

      for (size_t i = 0; i < point2d_array.points.size(); i++)
      {
        ref_path_cloud.points[i].x = point2d_array.points[i].x;
        ref_path_cloud.points[i].y = point2d_array.points[i].y;
        ref_path_cloud.points[i].z = 0.0;
        ref_path_cloud.points[i].r = 255;
        ref_path_cloud.points[i].g = 0;
        ref_path_cloud.points[i].b = 0;
      }

      pcl::toROSMsg(ref_path_cloud, ref_path_pcl);
      ref_path_publisher_PCL_->publish(ref_path_pcl);
    }

    void update_mpc_spline_visual(const sim_backend::msg::Point2DArray & point2d_array)
    {
      // Create a point cloud message
      sensor_msgs::msg::PointCloud2 mpc_spline_pcl;
      pcl::PointCloud<pcl::PointXYZRGB> mpc_spline_cloud;

      // Fill in the message
      mpc_spline_cloud.header.frame_id = "world";
      mpc_spline_cloud.height = 1;
      mpc_spline_cloud.width = point2d_array.points.size();
      mpc_spline_cloud.points.resize(mpc_spline_cloud.height * mpc_spline_cloud.width);

      for (size_t i = 0; i < point2d_array.points.size(); i++)
      {
        mpc_spline_cloud.points[i].x = point2d_array.points[i].x;
        mpc_spline_cloud.points[i].y = point2d_array.points[i].y;
        mpc_spline_cloud.points[i].z = 0.0;
        mpc_spline_cloud.points[i].r = 0;
        mpc_spline_cloud.points[i].g = 255;
        mpc_spline_cloud.points[i].b = 0;
      }

      pcl::toROSMsg(mpc_spline_cloud, mpc_spline_pcl);
      mpc_spline_publisher_->publish(mpc_spline_pcl);
    }

    void update_middle_path_visual(const sim_backend::msg::Point2DArray & point2d_array)
    {
      // Create a point cloud message
      sensor_msgs::msg::PointCloud2 track_pcl;
      pcl::PointCloud<pcl::PointXYZRGB> track_cloud;

      // Fill in the message
      track_cloud.header.frame_id = "world";
      track_cloud.height = 1;
      track_cloud.width = point2d_array.points.size();
      track_cloud.points.resize(track_cloud.height * track_cloud.width);

      for (size_t i = 0; i < point2d_array.points.size(); i++)
      {
        track_cloud.points[i].x = point2d_array.points[i].x;
        track_cloud.points[i].y = point2d_array.points[i].y;
        track_cloud.points[i].z = 0.0;
        track_cloud.points[i].r = 0;
        track_cloud.points[i].g = 0;
        track_cloud.points[i].b = 0;
      }

      pcl::toROSMsg(track_cloud, track_pcl);
      track_publisher_PCL_->publish(track_pcl);
    }

    void update_trackbounds_left_visual(const sim_backend::msg::Point2DArray & point2d_array)
    {
      // Create a marker array message
      visualization_msgs::msg::MarkerArray trackbounds_left_markers;

      // Fill in the message
      for (size_t i = 0; i < point2d_array.points.size(); i++)
      {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = this->now();
        marker.ns = "trackbounds_left";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = point2d_array.points[i].x;
        marker.pose.position.y = point2d_array.points[i].y;
        marker.pose.position.z = 0.5;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 0.8;
        trackbounds_left_markers.markers.push_back(marker);
      }

      trackbounds_left_publisher_markers_->publish(trackbounds_left_markers);
    }

    void update_trackbounds_right_visual(const sim_backend::msg::Point2DArray & point2d_array)
    {
      // Create a marker array message
      visualization_msgs::msg::MarkerArray trackbounds_right_markers;

      // Fill in the message
      for (size_t i = 0; i < point2d_array.points.size(); i++)
      {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = this->now();
        marker.ns = "trackbounds_right";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = point2d_array.points[i].x;
        marker.pose.position.y = point2d_array.points[i].y;
        marker.pose.position.z = 0.5;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
        trackbounds_right_markers.markers.push_back(marker);
      }

      trackbounds_right_publisher_markers_->publish(trackbounds_right_markers);
    }

    void visualize_velvec(const sim_backend::msg::VehicleState::SharedPtr msg)
    {
      // Create a marker message
      visualization_msgs::msg::Marker velocity_vector_marker;

      // Fill in the message
      velocity_vector_marker.header.frame_id = "vehicle_frame";
      velocity_vector_marker.header.stamp = this->now();
      velocity_vector_marker.ns = "velocity_vector";
      velocity_vector_marker.id = 0;
      velocity_vector_marker.type = visualization_msgs::msg::Marker::ARROW;
      velocity_vector_marker.action = visualization_msgs::msg::Marker::ADD;
      velocity_vector_marker.pose.position.x = 0.0;
      velocity_vector_marker.pose.position.y = 0.0;
      velocity_vector_marker.pose.position.z = 0.0;
      double angle = atan2(msg->dy_c_v, msg->dx_c_v);
      tf2::Quaternion q;
      q.setRPY(0, 0, angle);
      velocity_vector_marker.pose.orientation = tf2::toMsg(q);
      velocity_vector_marker.scale.x = hypot(msg->dx_c_v, msg->dy_c_v);
      velocity_vector_marker.scale.y = 0.1;
      velocity_vector_marker.scale.z = 0.1;
      velocity_vector_marker.color.r = 0.0;
      velocity_vector_marker.color.g = 1.0;
      velocity_vector_marker.color.b = 0.0;
      velocity_vector_marker.color.a = 0.5;
      velocity_vector_publisher_->publish(velocity_vector_marker);
    }

    // === Member variables ===
    rclcpp::Subscription<sim_backend::msg::VehicleState>::SharedPtr state_subscription_;

    rclcpp::Subscription<sim_backend::msg::Point2DArray>::SharedPtr middle_path_subscription_;

    rclcpp::Subscription<sim_backend::msg::Point2DArray>::SharedPtr trackbounds_left_subscription_;

    rclcpp::Subscription<sim_backend::msg::Point2DArray>::SharedPtr trackbounds_right_subscription_;

    rclcpp::Subscription<sim_backend::msg::Point2DArray>::SharedPtr ref_path_subscription_;

    rclcpp::Subscription<sim_backend::msg::Point2DArray>::SharedPtr mpc_xy_predict_subscription_;

    rclcpp::Subscription<sim_backend::msg::Point2DArray>::SharedPtr mpc_spline_subscription_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr track_publisher_PCL_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ref_path_publisher_PCL_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trackbounds_left_publisher_markers_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trackbounds_right_publisher_markers_;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr velocity_vector_publisher_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mpc_spline_publisher_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr mpc_xy_prediction_publisher_markers_;

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimVisualization>());
  rclcpp::shutdown();
  return 0;
}