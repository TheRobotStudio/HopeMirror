#!/usr/bin/env python3

from HandTrackerRenderer import HandTrackerRenderer
import argparse

import numpy as np
import cv2
import requests
import time
from sklearn.neighbors import KDTree

# ARC Server
arc_server = '192.168.1.17'
arc_passwd = 'admin'

# Define gesture-to-action mapping with auto position name
gesture_action_map = {
    # "Gesture": ("Action", "AutoPositionName")
    #"FIST": ("Hey", "Auto Position"),
    "WAVE": ("Hey", "Auto Position"),
    #"OK": ("ThumbsUp", "Position3"),
    #"PEACE": ("PeaceSign", "Position4"),
    #"ONE": ("Point", "Position5"),
    #"TWO": ("Salute", "Position6"),
    # Add additional mappings as needed
}

# Cooldown settings
cooldown_period = 15  # Trigger an action only every 5 seconds

# Movement tracking variables
wave_threshold = 100  # Minimum pixel movement to detect waving
wave_frames = 5       # Number of frames to confirm a wave
movement_history = []  # Stores wrist/palm positions to detect movement

# Define angles and associated actions with auto position names in a list of tuples
# Format: (angle_5_7, angle_7_9, "ARC Action", "AutoPositionName")
angle_action_grid = [
    (10, 10, "action_10_10", "Arm_L"),
    (10, 90, "action_10_90", "Arm_L"),
    #(30, 10, "action_30_10", "Auto Position 2"),
    #(40, 10, "action_40_10", "Auto Position 2"),
    #(50, 10, "action_50_10", "Auto Position 2"),
    #(60, 10, "action_60_10", "Auto Position 2"),
    #(70, 10, "action_70_10", "Auto Position 2"),
    #(10, 70, "action_10_70", "Auto Position 2"),
    # Add up to 47 or more predefined angles and actions as needed
]

# Separate angles, actions, and auto position names for KD-tree construction
angles = [(a[0], a[1]) for a in angle_action_grid]
actions = [(a[2], a[3]) for a in angle_action_grid]  # (action, auto_position_name) pairs

# Create a KD-tree from the angles
tree = KDTree(angles)

last_triggered_time = 0
last_gesture = ""

def calculate_angle(point1, point2, point3):
    # Convert points to vectors
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    
    # Calculate the cosine of the angle
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_angle_with_down_vector(point1, point2):
    # Vector from point1 to point2
    segment_vector = np.array(point2) - np.array(point1)
    down_vector = np.array([0, 1])  # Downward vector

    # Calculate the cosine of the angle
    cosine_angle = np.dot(segment_vector, down_vector) / (np.linalg.norm(segment_vector) * np.linalg.norm(down_vector))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def trigger_arc_action(action, auto_position_name):
    url = f'http://{arc_server}:80/Exec?password={arc_passwd}&script=ControlCommand("{auto_position_name}", "AutoPositionAction", "{action}")'
    response = requests.get(url)

    if response.status_code == 200:
        print(f"Action '{action}' triggered successfully!")
    else:
        print(f"Failed to trigger action '{action}'.")

def detect_wave(hand):
    global movement_history
    wrist_position = hand.landmarks[0][0]  # Assuming landmark 0 is wrist; adjust if needed

    # Add the current wrist position to the movement history
    movement_history.append(wrist_position)

    # Limit history to a certain number of frames
    if len(movement_history) > wave_frames:
        movement_history.pop(0)

    # Calculate movement by comparing distances across history
    if len(movement_history) == wave_frames:
        start_pos = movement_history[0]
        end_pos = movement_history[-1]
        distance = np.abs(start_pos - end_pos)

        # Detect wave if the hand moved back and forth
        if distance > wave_threshold:
            print("Wave detected!")
            return True

    return False

def process_gestures():
    global last_triggered_time
    global last_gesture
    current_time = time.time()
    
    # Trigger only if cooldown period has passed
    if current_time - last_triggered_time > cooldown_period:
        cv2.putText(frame, f"NO COOLDOWN - send action enabled", (50, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        for hand in hands:
            if not hasattr(hand, "gesture"):
                continue
            if hand.gesture == "FIVE" and hand.label == 'left':  # Check waving if hand is open
                hand.gesture = "FIVE or WAVE?"
                if detect_wave(hand):
                    action, auto_position_name = gesture_action_map["WAVE"]
                    print(f"Detected gesture: {hand.gesture} -> Triggering action: {action}")
                    last_gesture = "WAVE"
                    hand.gesture = "WAVE"
                    trigger_arc_action(action, auto_position_name)
                    last_triggered_time = current_time
                    break

            if hand.gesture and hand.gesture in gesture_action_map:
                action, auto_position_name = gesture_action_map[hand.gesture]
                print(f"Detected gesture: {hand.gesture} -> Triggering action: {action}")
                last_gesture = hand.gesture
                trigger_arc_action(action, auto_position_name)
                last_triggered_time = current_time
                break
    else:
        cv2.putText(frame, f"COOLDOWN - send action disabled", (50, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Last Detected gesture: {last_gesture} -> Triggering action: {gesture_action_map[last_gesture][0]}", (50, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

def find_nearest_action(angle_5_7, angle_7_9):
    dist, index = tree.query([[angle_5_7, angle_7_9]], k=1)
    nearest_action, auto_position_name = actions[index[0][0]]
    return nearest_action, auto_position_name

def mirror_to_discrete_actions():
    current_time = time.time()

    keypoint_5 = bag["body"].keypoints[5]   # Left shoulder
    keypoint_7 = bag["body"].keypoints[7]  # Left elbow
    keypoint_9 = bag["body"].keypoints[9]  # Left wrist

    # Calculate angles
    angle_5_7 = calculate_angle_with_down_vector(keypoint_5, keypoint_7)
    angle_7_9 = calculate_angle_with_down_vector(keypoint_7, keypoint_9)

    # Trigger only if cooldown period has passed
    if current_time - last_triggered_time > cooldown_period:
        # Find the nearest action and auto position name based on the current angles
        action, auto_position_name = find_nearest_action(angle_5_7, angle_7_9)
        if action:
            cv2.putText(frame, f"Triggering action '{action}' with '{auto_position_name}' for angles: 5-7 = {angle_5_7}, 7-9 = {angle_7_9}", (50, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            trigger_arc_action(action, auto_position_name)
       
# Setup and main loop
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true", help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str, help="Path to a blob file for palm detection model")
parser_tracker.add_argument('--no_lm', action="store_true", help="Only the palm detection model is run (no hand landmark model)")
parser_tracker.add_argument("--lm_model", type=str, help="Landmark model 'full', 'lite', 'sparse' or path to a blob file")
parser_tracker.add_argument('--use_world_landmarks', action="store_true", help="Fetch landmark 3D coordinates in meter")
parser_tracker.add_argument('-s', '--solo', action="store_true", help="Solo mode: detect one hand max. If not used, detect 2 hands max (Duo mode)")                    
parser_tracker.add_argument('-xyz', "--xyz", action="store_true", help="Enable spatial location measure of palm centers")
parser_tracker.add_argument('-g', '--gesture', action="store_true", help="Enable gesture recognition")
parser_tracker.add_argument('-c', '--crop', action="store_true", help="Center crop frames to a square shape")
parser_tracker.add_argument('-f', '--internal_fps', type=int, help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument("-r", "--resolution", choices=['full', 'ultra'], default='full', help="Sensor resolution: 'full' (1920x1080) or 'ultra' (3840x2160) (default=%(default)s)")
parser_tracker.add_argument('--internal_frame_height', type=int, help="Internal color camera frame height in pixels") 
parser_tracker.add_argument("-bpf", "--body_pre_focusing", default='higher', choices=['right', 'left', 'group', 'higher'], help="Enable Body Pre Focusing")      
parser_tracker.add_argument('-ah', '--all_hands', action="store_true", help="In Body Pre Focusing mode, consider all hands (not only the hands up)")                                     
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=10, help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")
parser_tracker.add_argument('--dont_force_same_image', action="store_true", help="(Edge Duo mode only) Don't force the use the same image when inferring the landmarks of the 2 hands (slower but skeleton less shifted)")
parser_tracker.add_argument('-lmt', '--lm_nb_threads', type=int, choices=[1,2], default=2, help="Number of the landmark model inference threads (default=%(default)i)")  
parser_tracker.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0, help="Print some debug infos. The type of info depends on the optional argument.")                
parser_renderer = parser.add_argument_group("Renderer arguments") 
parser_renderer.add_argument('-o', '--output', help="Path to output video file")
args = parser.parse_args()
dargs = vars(args)
tracker_args = {a: dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

if args.edge:
    from HandTrackerBpfEdge import HandTrackerBpf
    tracker_args['use_same_image'] = not args.dont_force_same_image
else:
    from HandTrackerBpf import HandTrackerBpf

tracker = HandTrackerBpf(
    input_src=args.input, 
    use_lm=not args.no_lm, 
    use_world_landmarks=args.use_world_landmarks,
    use_gesture=args.gesture,
    xyz=args.xyz,
    solo=args.solo,
    crop=args.crop,
    resolution=args.resolution,
    body_pre_focusing=args.body_pre_focusing,
    hands_up_only=not args.all_hands,
    single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
    lm_nb_threads=args.lm_nb_threads,
    stats=True,
    trace=args.trace,
    **tracker_args
)

renderer = HandTrackerRenderer(
    tracker=tracker,
    output=args.output
)

if not args.gesture:
    input("Run `python demo_bpf.py --gesture` to enable gesture detection, or press Enter to continue.")

while True:
    tracker.use_previous_landmarks = False
    frame, hands, bag = tracker.next_frame()
    if frame is None: break

    process_gestures()
    mirror_to_discrete_actions()

    # Draw hands
    frame = renderer.draw(frame, hands, bag)
    time.sleep(0.1)  

    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break

renderer.exit()
tracker.exit()
