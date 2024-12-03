# import deepq
from deepq import *
import gym
from gym import spaces
from additional_lib import *
from reward import *

class CarlaEnv(gym.Env):
    """A custom Gym environment wrapper for the CARLA simulator."""
    
    def __init__(self, host='localhost', port=2000, map_name='Town02', width=240, height=320, spawn_point_index=79, route=None):
        super(CarlaEnv, self).__init__()

        self.route = route

        self.spawn_index = spawn_point_index

        # CARLA setup
        self.client = carla.Client(host, port)
        # self.client.set_timeout(10.0)  # seconds
        self.world = self.client.load_world(map_name)

        # Image dimensions (for RGB camera)
        self.width = height
        self.height = width

        self.collision_sensor = None
        self.lane_invasion_sensor = None

    

        # Create a blueprint for the vehicle
        self.bp_lib = self.world.get_blueprint_library() 
        self.vehicle_bp = self.bp_lib.find('vehicle.audi.etron')
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.ego_vehicle = self.world.try_spawn_actor(self.vehicle_bp, self.spawn_points[self.spawn_index ])

        # transform = self.world.get_map().get_spawn_points()[0]
        # self.ego_vehicle = self.world.spawn_actor(self.bp_lib, transform)

        self.transform = carla.Transform(self.ego_vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),self.ego_vehicle.get_transform().rotation)
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(self.transform)


        self.camera_bp = self.bp_lib.find('sensor.camera.rgb') 
        # # set attributes of the camera
        self.camera_bp.set_attribute('image_size_x', str(self.width))
        self.camera_bp.set_attribute('image_size_y', str(self.height))
        self.camera_init_trans = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)) 
        self.camera = self.world.spawn_actor(
                        self.camera_bp, 
                        self.camera_init_trans, 
                        attach_to=self.ego_vehicle, 
                        attachment_type=carla.AttachmentType.SpringArm)


        self.camera_init_trans = carla.Transform(carla.Location(x=1.6, z=1.7)) 
        self.camera = self.world.spawn_actor(
                        self.camera_bp, 
                        self.camera_init_trans, 
                        attach_to= self.ego_vehicle, 
                        attachment_type=carla.AttachmentType.Rigid)

        # # We'll add all the other sensors' data into this dictionary later.
        # # For now, we've added the camera feed 
        self.sensor_data = {'rgb_image': np.zeros((self.height, self.width, 4))}
        self.camera.listen(lambda image: self.rgb_callback(image, self.sensor_data))


        # Define action space (steering, throttle, brake)
        # Action: [steering (continuous), throttle (continuous), brake (continuous)]
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0, 0.0]), 
                                       high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

        # Define observation space (RGB image)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        
        self.observation = np.zeros((self.height, self.width, 3))

    # RGB Camera Callback
    def rgb_callback(self, image, data_dict):
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) #Reshaping with alpha channel
        img[:,:,3] = 255 #Setting the alpha to 255 
        data_dict['rgb_image'] = img
        self.observation = img
            
    def reset(self):
        """Resets the environment and spawns a new vehicle."""
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()

        if self.collision_sensor is not None:
            self.collision_sensor.destroy()

        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()


        self.route_index = 1
        self.previous_distance = 0

        self.vehicle_bp = self.bp_lib.find('vehicle.audi.etron')

        self.ego_vehicle = self.world.try_spawn_actor(self.vehicle_bp, self.spawn_points[self.spawn_index])


        self.spectator = self.world.get_spectator()
        self.transform = carla.Transform(self.ego_vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),self.ego_vehicle.get_transform().rotation)
        self.spectator.set_transform(self.transform)


        self.camera_bp = self.bp_lib.find('sensor.camera.rgb') 
        # set attributes of the camera
        self.camera_bp.set_attribute('image_size_x', str(self.width))
        self.camera_bp.set_attribute('image_size_y', str(self.height))
        self.camera_init_trans = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)) 
        self.camera = self.world.spawn_actor(
                        self.camera_bp, 
                        self.camera_init_trans, 
                        attach_to=self.ego_vehicle, 
                        attachment_type=carla.AttachmentType.SpringArm)


        self.camera_init_trans = carla.Transform(carla.Location(x=1.6, z=1.7)) 
        self.camera = self.world.spawn_actor(
                        self.camera_bp, 
                        self.camera_init_trans, 
                        attach_to= self.ego_vehicle, 
                        attachment_type=carla.AttachmentType.Rigid)

        # We'll add all the other sensors' data into this dictionary later.
        # For now, we've added the camera feed 
        self.sensor_data = {'rgb_image': np.zeros((self.height, self.width, 4))}
        self.camera.listen(lambda image: self.rgb_callback(image, self.sensor_data))

        collision_bp = self.bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(self.process_collision)

        self.lane_invasion_sensor = self.world.spawn_actor(self.bp_lib.find('sensor.other.lane_invasion'), carla.Transform(), attach_to=self.ego_vehicle)
        self.lane_invasion_sensor.listen(self.process_lane_invasion)

        # Reset other environment states
        self.lane_invas = False
        self.done = False
        self.steps = 0

        # # Process the initial image
        # self.process_image(self.sensor_data['rgb_image'])

        return self.observation  # Return the initial observation

    def process_collision(self, event):
        """Process the collision event."""
        self.done = True

    def process_lane_invasion(self, event):
        """Process the lane invasion event."""
        self.lane_invas = True

    def process_image(self, image):
        """Process the image received from the camera sensor."""
        image_array = np.array(image.raw_data)
        image_array = image_array.reshape((self.height, self.width, 4))  # RGBA
        image_array = image_array[:, :, :3]  # Discard the alpha channel
        self.observation = image_array

    def step(self, action):
        """Perform a step in the environment."""
        steering, throttle, brake = action
        
        control = carla.VehicleControl(throttle=float(1), brake=float(0), steer=float(steering))
        # Control the vehicle
        self.ego_vehicle.apply_control(control)

        # Simulate for some time (e.g., 1 second)
        self.world.tick()

        # find the velocity of the ego vehicle
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        position = self.ego_vehicle.get_location()
        current_position = (position.x, position.y)

        waypoint = ((self.route[self.route_index])[0]).transform.location
        next_waypoint = (waypoint.x, waypoint.y)


        # Calculate the reward (for simplicity, we use a dummy reward)
        output = reward_function(self.done, speed, self.lane_invas, current_position, self.previous_distance, next_waypoint, float(5.0))  # Replace with your reward logic

        reward = output[0]
        self.previous_distance = output[1]
        

        self.route_index += 1

        # Check if the episode is done (for example, if the car crashes or exceeds time limit)
        if self.done:
            return self.observation, reward, self.done, {}

        # Increase the step counter
        self.steps += 1

        # Return the observation, reward, done, and info
        return self.observation, reward, self.done, {}

    def render(self, mode='human'):
        """Render the environment (optional)."""
        # You can use this method to show the image using OpenCV, matplotlib, or any other method.
        # For now, we'll just use OpenCV to display the image:
        import cv2
        # Process the initial image
        self.observation = self.sensor_data['rgb_image']
        cv2.imshow('Carla Environment', self.observation)
        cv2.waitKey(1)

    def close(self):
        """Clean up the environment (stop sensors and actors)."""
        if self.camera is not None:
            self.camera.stop()
        if self.vehicle is not None:
            self.vehicle.destroy()


        

#The default parameters for training a agent can be found in deepq.py
def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    start_index = 79
    end_index = start_index - 1

    route_ideal  = generate_route(start_index, end_index, map_name='Town02', sampling_resolution=2)

    #Initialize your carla env above and train the Deep Q-Learning agent
    carla_obj = CarlaEnv(spawn_point_index=start_index, route=route_ideal)


    # Train the agent
    learn(carla_obj)

if __name__ == '__main__':

    main()

